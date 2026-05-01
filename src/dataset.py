import os
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def _build_transforms(image_size: int, train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize(image_size + 32),
        transforms.CenterCrop(image_size),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


class OpenIDataset(Dataset):
    """Indiana University Chest X-ray dataset with radiology report pairs."""

    def __init__(self, root: str, tokenizer, split: str = "train",
                 image_size: int = 224, max_length: int = 128):
        self.root = Path(root)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transform = _build_transforms(image_size, train=(split == "train"))
        self.samples = self._load_samples(split)

    def _load_samples(self, split: str) -> list[dict]:
        # TODO: parse OpenI XML reports → build (image_path, caption) list
        # Expected XML fields: findings, impression
        # Placeholder — implement after downloading data
        raise NotImplementedError("Implement after downloading Indiana dataset")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        image = Image.open(s["image_path"]).convert("RGB")
        image = self.transform(image)
        enc = self.tokenizer(
            s["caption"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "image":          image,
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }


NIH_CLASSES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Infiltration", "Pneumonia", "Pneumothorax",
]


class NIHDataset(Dataset):
    """NIH ChestX-ray14 dataset for zero-shot evaluation."""

    def __init__(self, root: str, classes: list[str] = NIH_CLASSES, image_size: int = 224):
        self.root = Path(root)
        self.classes = classes
        self.transform = _build_transforms(image_size, train=False)
        self.df = self._load_metadata()

    def _load_metadata(self) -> pd.DataFrame:
        csv_path = self.root / "Data_Entry_2017.csv"
        df = pd.read_csv(csv_path)
        df = df.rename(columns={"Image Index": "image", "Finding Labels": "labels"})
        for cls in self.classes:
            df[cls] = df["labels"].str.contains(cls).astype(int)
        # Keep only rows that have at least one of our target classes or are Normal
        mask = df[self.classes].any(axis=1) | df["labels"].eq("No Finding")
        return df[mask].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self.root / "images" / row["image"]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        labels = row[self.classes].values.astype("float32")
        return {"image": image, "labels": labels}
