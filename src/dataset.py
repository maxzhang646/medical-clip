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
        reports = pd.read_csv(self.root / "indiana_reports.csv")
        projections = pd.read_csv(self.root / "indiana_projections.csv")

        # Keep frontal view only and join with reports
        frontal = projections[projections["projection"] == "Frontal"][["uid", "filename"]]
        df = reports.merge(frontal, on="uid", how="inner")

        # Drop rows with empty findings or impression
        df = df[df["findings"].notna() & df["impression"].notna()]
        df = df[df["findings"].str.strip().ne("") & df["impression"].str.strip().ne("")]
        df = df.drop_duplicates(subset="uid").reset_index(drop=True)

        # Patient-level train/val/test split (80/10/10)
        uids = df["uid"].unique()
        n = len(uids)
        rng = pd.Series(uids).sample(frac=1, random_state=42).values
        cuts = [int(n * 0.8), int(n * 0.9)]
        split_uids = {"train": rng[:cuts[0]], "val": rng[cuts[0]:cuts[1]], "test": rng[cuts[1]:]}
        df = df[df["uid"].isin(split_uids[split])].reset_index(drop=True)

        img_dir = self.root / "images" / "images_normalized"
        samples = []
        for _, row in df.iterrows():
            img_path = img_dir / row["filename"]
            if not img_path.exists():
                continue
            caption = f"{row['findings'].strip()} {row['impression'].strip()}"
            samples.append({"image_path": str(img_path), "caption": caption, "uid": row["uid"]})
        return samples

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
        self.image_index = self._build_image_index()
        self.df = self._load_metadata()

    def _build_image_index(self) -> dict:
        img_dir = self.root / "images"
        index = {p.name: p for p in img_dir.rglob("*.png")}
        if not index:
            # fallback: images directly in root
            index = {p.name: p for p in self.root.rglob("*.png")}
        print(f"NIHDataset: indexed {len(index)} images")
        return index

    def _load_metadata(self) -> pd.DataFrame:
        csv_path = self.root / "Data_Entry_2017.csv"
        df = pd.read_csv(csv_path)
        df = df.rename(columns={"Image Index": "image", "Finding Labels": "labels"})
        for cls in self.classes:
            df[cls] = df["labels"].str.contains(cls).astype(int)
        mask = df[self.classes].any(axis=1) | df["labels"].eq("No Finding")
        df = df[mask].reset_index(drop=True)
        # Keep only rows whose image file exists
        df = df[df["image"].isin(self.image_index)].reset_index(drop=True)
        return df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self.image_index[row["image"]]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        labels = row[self.classes].values.astype("float32")
        return {"image": image, "labels": labels}
