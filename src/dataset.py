from pathlib import Path
from typing import Callable, Optional

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


NORMALIZATION_STATS = {
    "imagenet": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
    "clip": {
        "mean": [0.48145466, 0.4578275, 0.40821073],
        "std": [0.26862954, 0.26130258, 0.27577711],
    },
}


def _normalization_stats(image_normalization: str) -> tuple[list[float], list[float]]:
    if image_normalization not in NORMALIZATION_STATS:
        valid = ", ".join(sorted(NORMALIZATION_STATS))
        raise ValueError(f"Unknown image_normalization: {image_normalization}. Valid options: {valid}")
    stats = NORMALIZATION_STATS[image_normalization]
    return stats["mean"], stats["std"]


def _build_transforms(image_size: int, train: bool,
                      image_normalization: str = "imagenet") -> transforms.Compose:
    mean, std = _normalization_stats(image_normalization)
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return transforms.Compose([
        transforms.Resize(image_size + 32),
        transforms.CenterCrop(image_size),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


class OpenIDataset(Dataset):
    """Indiana University Chest X-ray dataset with radiology report pairs."""

    def __init__(self, root: str, tokenizer, split: str = "train",
                 image_size: int = 224, max_length: int = 128,
                 image_normalization: str = "imagenet",
                 split_dir: Optional[str] = None,
                 split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
                 split_seed: int = 42,
                 transform: Optional[Callable] = None,
                 tokenize_fn: Optional[Callable[[str], dict]] = None):
        """`transform` / `tokenize_fn` let another backbone (e.g. BioMedCLIP) inject
        its own preprocessing and tokenizer. When given they take precedence over
        `image_size` / `image_normalization` and `tokenizer` / `max_length`.
        """
        if tokenizer is None and tokenize_fn is None:
            raise ValueError("OpenIDataset needs either a tokenizer or a tokenize_fn.")
        self.root = Path(root)
        self.tokenizer = tokenizer
        self.tokenize_fn = tokenize_fn
        self.max_length = max_length
        self.image_normalization = image_normalization
        self.split_dir = Path(split_dir) if split_dir else None
        self.split_ratios = split_ratios
        self.split_seed = split_seed
        self.transform = transform if transform is not None else _build_transforms(
            image_size,
            train=(split == "train"),
            image_normalization=image_normalization,
        )
        self.samples = self._load_samples(split)

    def _load_samples(self, split: str) -> list[dict]:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown OpenI split: {split}")
        required = [
            self.root / "indiana_reports.csv",
            self.root / "indiana_projections.csv",
            self.root / "images" / "images_normalized",
        ]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "OpenI dataset is incomplete. Missing:\n"
                + "\n".join(f"  - {p}" for p in missing)
                + "\nExpected structure: indiana_reports.csv, "
                  "indiana_projections.csv, images/images_normalized/*.png"
            )

        reports = pd.read_csv(self.root / "indiana_reports.csv")
        projections = pd.read_csv(self.root / "indiana_projections.csv")

        # Keep frontal view only and join with reports
        frontal = projections[projections["projection"] == "Frontal"][["uid", "filename"]]
        df = reports.merge(frontal, on="uid", how="inner")

        # Drop rows with empty findings or impression
        df = df[df["findings"].notna() & df["impression"].notna()]
        df = df[df["findings"].str.strip().ne("") & df["impression"].str.strip().ne("")]
        df = df.drop_duplicates(subset="uid").reset_index(drop=True)

        uids = df["uid"].unique()
        split_uids = self._load_split_uids(split, uids)
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

    def _load_split_uids(self, split: str, uids) -> dict[str, list]:
        if self.split_dir:
            split_path = self.split_dir / f"openi_{split}_uids.txt"
            if split_path.exists():
                return {split: [int(line.strip()) for line in split_path.read_text().splitlines() if line.strip()]}

        train_ratio, val_ratio, test_ratio = self.split_ratios
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"OpenI split ratios must sum to 1.0, got {total}")

        n = len(uids)
        rng = pd.Series(uids).sample(frac=1, random_state=self.split_seed).values
        train_cut = int(n * train_ratio)
        val_cut = int(n * (train_ratio + val_ratio))
        return {
            "train": rng[:train_cut],
            "val": rng[train_cut:val_cut],
            "test": rng[val_cut:],
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        image = Image.open(s["image_path"]).convert("RGB")
        image = self.transform(image)
        if self.tokenize_fn is not None:
            enc = self.tokenize_fn(s["caption"])
            return {
                "image":          image,
                "input_ids":      enc["input_ids"],
                "attention_mask": enc["attention_mask"],
            }
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

    def __init__(self, root: str, classes: list[str] = NIH_CLASSES,
                 image_size: int = 224, image_normalization: str = "imagenet",
                 transform: Optional[Callable] = None):
        self.root = Path(root)
        self.classes = classes
        self.image_normalization = image_normalization
        self.transform = transform if transform is not None else _build_transforms(
            image_size,
            train=False,
            image_normalization=image_normalization,
        )
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
