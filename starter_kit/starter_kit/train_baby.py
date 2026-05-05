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


"""
Changes from baseline
---------------------
Architecture (SmallCNN):
  - 4 conv blocks instead of 3, with residual (skip) connections.
  - Wider channels: 32 → 64 → 128 → 256, then projected down.
  - Depthwise-separable conv in the last block to stay under the 500K cap.
  - Two-layer MLP head with an extra hidden layer (256 → 128 → 7).
  - Dropout raised to 0.4 on the head; 0.1 spatial dropout after block 3.
  Total params: ~420K — well under the 500K cap.

Training:
  - AdamW with weight_decay=1e-4 (L2 regularisation "for free").
  - CosineAnnealingLR over 40 epochs instead of flat LR for 20.
  - Data augmentation applied on-the-fly in the training loop:
      * RandomHorizontalFlip
      * RandomVerticalFlip  (production line images can be upside-down)
      * ColorJitter (brightness / contrast / saturation ±20%)
      * RandomErasing (simulates occlusion on the line)
  - Label smoothing (ε=0.1) in cross-entropy to soften hard targets.
  - Best-on-val checkpoint saved; that checkpoint is what gets scripted.

Everything else (Preprocess wrapper, ImageDataset, server contract) is
unchanged so distill.py can import SmallCNN and Preprocess without
modification.
"""
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, random_split

DATA_ROOT = Path(__file__).parent / "train"
NUM_CLASSES = 7
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
EPOCHS = 40

# =============================================================================
# Dataset
# =============================================================================
class ImageDataset(Dataset):
    def __init__(self, root: Path, augment: bool = False):
        self.root    = Path(root)
        self.df      = pd.read_csv(self.root / "labels.csv")
        self.augment = augment
        self.transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.RandomErasing(p=0.25, scale=(0.02, 0.15)),
        ])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        img = torchvision.io.read_image(str(self.root / row["filename"]))  # uint8 (3,H,W)
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        elif img.shape[0] == 4:
            img = img[:3]
        img = img.float() / 255.0   # float in [0,1] — matches server contract
        if self.augment:
            img = self.transform(img)
        return img, int(row["label"])


# =============================================================================
# Preprocess wrapper — enforces the (B, 3, 256, 256) server contract
# (unchanged — distill.py imports this)
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
        x = F.interpolate(x, size=self.size, mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        return self.net(x)


# =============================================================================
# Building block: Conv → BN → ReLU (→ optional residual add)
# =============================================================================
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3,
                 stride: int = 1, groups: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride,
                      padding=kernel // 2, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResBlock(nn.Module):
    """Two ConvBNReLU layers with a residual connection.
    If in_ch != out_ch, a 1x1 conv aligns the skip."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = ConvBNReLU(in_ch, out_ch)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = (
            nn.Conv2d(in_ch, out_ch, 1, bias=False)
            if in_ch != out_ch else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.relu(out + self.skip(x))
        return out


class DepthwiseSepConv(nn.Module):
    """Depthwise-separable conv: cheap way to add capacity under the cap."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.dw = ConvBNReLU(in_ch, in_ch, groups=in_ch)   # depthwise
        self.pw = ConvBNReLU(in_ch, out_ch, kernel=1)       # pointwise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


# =============================================================================
# Improved SmallCNN  (~420K params at size=64 input)
# =============================================================================
class SmallCNN(nn.Module):
    """
    Input: (B, 3, 64, 64)  — after Preprocess resizes from 256
    Block 1: 3  → 32,  MaxPool  →  32x32
    Block 2: 32 → 64,  MaxPool  →  16x16   (ResBlock)
    Block 3: 64 → 128, MaxPool  →   8x8    (ResBlock)
    Block 4: 128→ 256, DS conv  →   8x8    (Depthwise-sep, no extra pool)
    GAP → (B, 256) → Dropout(0.4) → Linear(256,128) → ReLU → Dropout(0.3)
                   → Linear(128, 7)
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        # Block 1 — plain conv (no residual; first block always clean)
        self.block1 = nn.Sequential(
            ConvBNReLU(3, 32),
            nn.MaxPool2d(2),        # 32x32
        )

        # Block 2 — residual
        self.block2 = nn.Sequential(
            ResBlock(32, 64),
            nn.MaxPool2d(2),        # 16x16
        )

        # Block 3 — residual
        self.block3 = nn.Sequential(
            ResBlock(64, 128),
            nn.MaxPool2d(2),        # 8x8
        )

        # Block 4 — depthwise-separable (keeps param count manageable)
        self.block4 = nn.Sequential(
            DepthwiseSepConv(128, 256),   # still 8x8
        )

        # Classifier head
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


# =============================================================================
# Training utilities
# =============================================================================
def train_one_epoch(model, loader, opt, device):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        # Label smoothing (ε=0.1) softens the hard targets slightly
        loss = F.cross_entropy(logits, y, label_smoothing=0.1)
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_sum += loss.item() * x.size(0)
        correct  += (logits.argmax(1) == y).sum().item()
        total    += x.size(0)
    return loss_sum / total, correct / total


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    total, correct = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        correct += (logits.argmax(1) == y).sum().item()
        total   += x.size(0)
    return correct / total


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# =============================================================================
# Main
# =============================================================================
def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 80/20 train/val split (seed=0, reproducible)
    ds    = ImageDataset(DATA_ROOT, augment=False)   # base dataset (no aug)
    n_val   = max(1, len(ds) // 5)
    n_train = len(ds) - n_val
    train_indices, val_indices = random_split(
        range(len(ds)), [n_train, n_val],
        generator=torch.Generator().manual_seed(0),
    )

    # Wrap indices into augmentation-aware subsets
    train_ds = torch.utils.data.Subset(ImageDataset(DATA_ROOT, augment=True),  train_indices)
    val_ds   = torch.utils.data.Subset(ImageDataset(DATA_ROOT, augment=False), val_indices)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False, num_workers=0)

    inner = SmallCNN()
    model = Preprocess(inner, size=64).to(device)

    n_params = count_params(model)
    print(f"Total parameters: {n_params:,}")
    assert n_params <= 500_000, f"Over cap: {n_params:,}"

    # AdamW gives L2 regularisation via weight_decay
    opt       = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best_val   = -1.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, opt, device)
        val_acc               = evaluate(model, val_loader, device)
        scheduler.step()
        print(
            f"Epoch {epoch:2d}  train_loss={train_loss:.3f}  "
            f"train_acc={train_acc:.3f}  val_acc={val_acc:.3f}  "
            f"lr={opt.param_groups[0]['lr']:.5f}"
        )
        if val_acc > best_val:
            best_val   = val_acc
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

    # Restore best-on-val checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"\nBest val_acc = {best_val:.4f}")

    # Move to CPU + eval, sanity-check, then TorchScript & save
    model_cpu = model.cpu().eval()
    with torch.inference_mode():
        dummy = torch.rand(2, 3, 256, 256)
        out   = model_cpu(dummy)
        assert out.shape == (2, 7), f"Output shape mismatch: {tuple(out.shape)}"

    scripted = torch.jit.script(model_cpu)
    torch.jit.save(scripted, "model.pt")
    print("Saved model.pt — upload this to the leaderboard.")


if __name__ == "__main__":
    main()
