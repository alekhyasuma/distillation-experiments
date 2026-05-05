"""
=============================================================================
ML2 Homework 2 — train_teacher.py
=============================================================================

Train a "teacher" model that you'll later distill into your submittable
student. The teacher is too big to submit (it has way more than 500K
parameters); its only role is to produce soft labels on the unlabeled
images that distill.py uses to teach a smaller student.

This script:
  1. Loads a pretrained EfficientNet-B0 (~5M params, ImageNet weights).
  2. Replaces its classifier with a 7-class head.
  3. Fine-tunes on the 399 labeled training images.
  4. Caches the trained teacher to teacher_state.pth (so you don't
     retrain on every run; delete the file to force a retrain).
  5. Runs inference on the 798 unlabeled images and saves the resulting
     pre-softmax logits to teacher_soft_labels.npy + a matching filename
     index in teacher_filenames.txt.

Then run distill.py.

You can swap to a different backbone (resnet50, efficientnet_b3,
convnext_tiny, ...) by editing build_teacher().
=============================================================================
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset, random_split

TRAIN_ROOT = Path(__file__).parent / "train"
UNLABELED_ROOT = Path(__file__).parent / "unlabeled"
# Cache file is named after the backbone so swapping build_teacher() to a
# different model (resnet50, convnext_tiny, ...) won't try to load the
# wrong checkpoint. If you change BACKBONE, the next run trains fresh.
BACKBONE = "efficientnet_b0"
TEACHER_STATE_PATH = Path(__file__).parent / f"teacher_{BACKBONE}.pth"
NUM_CLASSES = 7
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
EPOCHS = 10


class LabeledDataset(Dataset):
    def __init__(self, root: Path, size: int = 224):
        self.root = root
        self.df = pd.read_csv(root / "labels.csv")
        self.size = size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = torchvision.io.read_image(str(self.root / row["filename"]))
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        elif img.shape[0] == 4:
            img = img[:3]
        img = img.float() / 255.0
        img = F.interpolate(img.unsqueeze(0), size=self.size,
                            mode="bilinear", align_corners=False).squeeze(0)
        img = (img - IMAGENET_MEAN.squeeze(0)) / IMAGENET_STD.squeeze(0)
        return img, int(row["label"])


class UnlabeledDataset(Dataset):
    def __init__(self, root: Path, size: int = 224):
        self.root = root
        self.filenames = sorted(p.name for p in root.glob("*.jpg"))
        self.size = size

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, i):
        fn = self.filenames[i]
        img = torchvision.io.read_image(str(self.root / fn))
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        elif img.shape[0] == 4:
            img = img[:3]
        img = img.float() / 255.0
        img = F.interpolate(img.unsqueeze(0), size=self.size,
                            mode="bilinear", align_corners=False).squeeze(0)
        img = (img - IMAGENET_MEAN.squeeze(0)) / IMAGENET_STD.squeeze(0)
        return img, fn


def build_teacher(num_classes: int = NUM_CLASSES) -> nn.Module:
    m = torchvision.models.efficientnet_b0(
        weights=torchvision.models.EfficientNet_B0_Weights.DEFAULT
    )
    # EfficientNet's classifier is Sequential(Dropout, Linear); replace
    # the final Linear with a 7-class head and keep the dropout.
    in_features = m.classifier[1].in_features
    m.classifier[1] = nn.Linear(in_features, num_classes)
    return m


def fine_tune(model, train_loader, val_loader, epochs: int, device):
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item() * x.size(0); n += x.size(0)
        model.eval()
        correct, total = 0, 0
        with torch.inference_mode():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(1) == y).sum().item()
                total += x.size(0)
        print(f"Epoch {epoch:2d}  train_loss={loss_sum / n:.3f}  val_acc={correct / total:.3f}")


@torch.inference_mode()
def dump_soft_labels(model, unlabeled_loader, device, out_path: Path, filenames_out: Path):
    model.eval()
    all_logits, all_filenames = [], []
    for x, fns in unlabeled_loader:
        x = x.to(device)
        logits = model(x).cpu().numpy()
        all_logits.append(logits)
        all_filenames.extend(fns)
    np.save(out_path, np.concatenate(all_logits, axis=0))
    filenames_out.write_text("\n".join(all_filenames) + "\n")
    print(f"Saved {out_path.name} shape={np.load(out_path).shape}")
    print(f"Saved filename index → {filenames_out.name}")


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 80/20 train/val split (seed=0, reproducible).
    labeled = LabeledDataset(TRAIN_ROOT)
    n_val = max(1, len(labeled) // 5)
    n_train = len(labeled) - n_val
    train_ds, val_ds = random_split(
        labeled, [n_train, n_val],
        generator=torch.Generator().manual_seed(0),
    )
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    teacher = build_teacher().to(device)
    print(f"Teacher parameters: {sum(p.numel() for p in teacher.parameters()):,}")

    if TEACHER_STATE_PATH.exists():
        teacher.load_state_dict(torch.load(TEACHER_STATE_PATH, map_location=device))
        print(f"Loaded teacher weights from {TEACHER_STATE_PATH.name}")
    else:
        fine_tune(teacher, train_loader, val_loader, epochs=EPOCHS, device=device)
        torch.save(teacher.state_dict(), TEACHER_STATE_PATH)
        print(f"Saved teacher weights → {TEACHER_STATE_PATH.name}")

    unlabeled = UnlabeledDataset(UNLABELED_ROOT)
    unlabeled_loader = DataLoader(unlabeled, batch_size=16, shuffle=False)
    dump_soft_labels(
        teacher, unlabeled_loader, device,
        out_path=Path(__file__).parent / "teacher_soft_labels.npy",
        filenames_out=Path(__file__).parent / "teacher_filenames.txt",
    )


if __name__ == "__main__":
    main()
