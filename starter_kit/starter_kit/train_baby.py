"""
=============================================================================
ML2 Homework 2 — train_baby.py
=============================================================================

Simplest starting point. Trains a small CNN from scratch on the 399 labeled
images for 20 epochs and saves a submittable model.pt.

This baseline is intentionally weak (~95K params). You should:
  1. Run it as-is to confirm everything works end-to-end and submit it.
  2. Make the network bigger (up to 500K params).
  3. Add regularization (more dropout, weight decay, augmentation).
  4. Use the 798 unlabeled images via knowledge distillation
     (run train_teacher.py first, then distill.py).

You are given:
  - train/        : 399 labeled 256x256 RGB JPEGs + labels.csv
  - unlabeled/    : 798 unlabeled images (use these with a teacher)

THE CONTRACT (very important):
  - The leaderboard server calls your model with x of shape
    (B, 3, 256, 256), float32 in [0, 1].
  - Your model must return (B, 7) float logits.
  - Preprocessing (resize, normalize) MUST be inside your submitted module.
  - Total parameters must be <= 500,000.

Allowed layers: Conv2d, BatchNorm*, LayerNorm, Dropout*, MaxPool*, AvgPool*,
any activation, Linear, Flatten.

Pretrained models may be used ONLY as teachers during training
(see train_teacher.py). Your submitted model must be your own architecture.
=============================================================================
"""
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader, random_split

DATA_ROOT = Path(__file__).parent / "train"
NUM_CLASSES = 7
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# =============================================================================
# Dataset
# =============================================================================
class ImageDataset(Dataset):
    def __init__(self, root: Path):
        self.root = Path(root)
        self.df = pd.read_csv(self.root / "labels.csv")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        img = torchvision.io.read_image(str(self.root / row["filename"]))  # uint8 (3,H,W)
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        elif img.shape[0] == 4:
            img = img[:3]
        img = img.float() / 255.0  # float in [0,1] — matches the server contract
        return img, int(row["label"])


# =============================================================================
# Preprocess wrapper — enforces the (B, 3, 256, 256) server contract
# =============================================================================
class Preprocess(nn.Module):
    """Wraps your network. Resizes the server's 256x256 input to `size`
    and normalizes with ImageNet mean/std before forwarding."""

    def __init__(self, net: nn.Module, size: int = 64,
                 mean=IMAGENET_MEAN, std=IMAGENET_STD):
        super().__init__()
        self.net = net
        self.size = size
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, 256, 256) float in [0, 1]
        x = F.interpolate(x, size=self.size, mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        return self.net(x)


# =============================================================================
# A tiny baseline CNN — ~95K params. Lots of room to grow under the 500K cap.
# =============================================================================
class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        # TODO: Try wider channels, deeper stacks, more dropout, weight
        # decay, etc. The param cap is 500,000.
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                          # 32x32
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                          # 16x16
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),                                  # 1x1
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# =============================================================================
# Training loop
# =============================================================================
def train_one_epoch(model, loader, opt, device):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_sum += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    total, correct = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return correct / total


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# =============================================================================
# Main
# =============================================================================
def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 80/20 train/val split (seed=0, reproducible).
    ds = ImageDataset(DATA_ROOT)
    n_val = max(1, len(ds) // 5)
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(
        ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(0),
    )
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)

    inner = SmallCNN()
    model = Preprocess(inner, size=64).to(device)

    n_params = count_params(model)
    print(f"Total parameters: {n_params:,}")
    assert n_params <= 500_000, f"Over cap: {n_params:,}"

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(1, 21):
        train_loss, train_acc = train_one_epoch(model, train_loader, opt, device)
        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:2d}  train_loss={train_loss:.3f}  "
              f"train_acc={train_acc:.3f}  val_acc={val_acc:.3f}")

    # Save — move to cpu + eval mode for the server.
    model_cpu = model.cpu().eval()
    # Sanity check: the saved model must forward correctly on the server-shaped input.
    with torch.inference_mode():
        dummy = torch.rand(2, 3, 256, 256)
        out = model_cpu(dummy)
        assert out.shape == (2, 7), f"Output shape mismatch: {tuple(out.shape)}"

    # TorchScript before saving so the server can load without your class
    # definitions on its import path.
    scripted = torch.jit.script(model_cpu)
    torch.jit.save(scripted, "model.pt")
    print("Saved model.pt — upload this to the leaderboard.")


if __name__ == "__main__":
    main()
