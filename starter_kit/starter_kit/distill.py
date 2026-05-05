"""
=============================================================================
ML2 Homework 2 — distill.py
=============================================================================

Knowledge distillation training. Trains the SmallCNN defined in
train_baby.py on:
  - Hard cross-entropy loss on the labeled train split.
  - Temperature-softened KL loss on the unlabeled images, against the
    teacher's pre-softmax logits (cached by train_teacher.py).
The two losses are combined as: ALPHA * KD_loss + (1 - ALPHA) * CE_loss.

Prerequisites
-------------
1. Run train_teacher.py first to produce teacher_soft_labels.npy and
   teacher_filenames.txt.
2. Make sure the SmallCNN definition in train_baby.py is the architecture
   you want distilled — distill.py imports SmallCNN and Preprocess from
   there, so any change you make in train_baby.py flows through.

Output: model.pt (TorchScript) — upload this to the leaderboard.

Hyperparameters worth sweeping
------------------------------
- T     : distillation temperature. Default 4. Try 2, 4, 6, 8.
- ALPHA : weight on the KD loss. Default 0.7. Lower if your teacher is
          weak; higher if your labeled set is too small to learn from.
- EPOCHS: 50 by default; with the cosine LR schedule, going much higher
          rarely helps once val plateaus.
=============================================================================
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset, random_split

from train_baby import ImageDataset, Preprocess, SmallCNN, count_params

TRAIN_ROOT = Path(__file__).parent / "train"
UNLABELED_ROOT = Path(__file__).parent / "unlabeled"
SOFT_LABELS = Path(__file__).parent / "teacher_soft_labels.npy"
FILENAMES = Path(__file__).parent / "teacher_filenames.txt"
NUM_CLASSES = 7
T = 4.0        # distillation temperature
ALPHA = 0.7    # weight on KD loss
EPOCHS = 50


class UnlabeledWithSoftLabels(Dataset):
    def __init__(self, root: Path, filenames_path: Path, soft_labels_path: Path):
        self.root = root
        self.filenames = filenames_path.read_text().strip().splitlines()
        self.logits = np.load(soft_labels_path)  # (N, 7)
        assert len(self.filenames) == self.logits.shape[0], (
            "filenames and soft-labels length mismatch — rerun train_teacher.py"
        )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, i):
        fn = self.filenames[i]
        img = torchvision.io.read_image(str(self.root / fn))
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        elif img.shape[0] == 4:
            img = img[:3]
        img = img.float() / 255.0  # (3, 256, 256) — server-shaped
        logits = torch.tensor(self.logits[i], dtype=torch.float32)
        return img, logits


def train_step(student, x_lab, y_lab, x_un, t_logits, opt):
    student.train()
    z_lab = student(x_lab)
    loss_ce = F.cross_entropy(z_lab, y_lab)

    z_un = student(x_un)
    loss_kd = F.kl_div(
        F.log_softmax(z_un / T, dim=1),
        F.softmax(t_logits / T, dim=1),
        reduction="batchmean",
    ) * (T * T)

    loss = ALPHA * loss_kd + (1 - ALPHA) * loss_ce
    opt.zero_grad(); loss.backward(); opt.step()
    return loss.item(), loss_ce.item(), loss_kd.item()


@torch.inference_mode()
def evaluate(model, loader, device):
    """Returns (accuracy, mean_cross_entropy)."""
    model.eval()
    correct, total, xent_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        correct += (logits.argmax(1) == y).sum().item()
        xent_sum += F.cross_entropy(logits, y, reduction="sum").item()
        total += x.size(0)
    return correct / total, xent_sum / total


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Same 80/20 split (seed=0) as train_baby.py — apples-to-apples val.
    labeled = ImageDataset(TRAIN_ROOT)
    n_val = max(1, len(labeled) // 5)
    n_train = len(labeled) - n_val
    train_ds, val_ds = random_split(
        labeled, [n_train, n_val],
        generator=torch.Generator().manual_seed(0),
    )
    unlabeled = UnlabeledWithSoftLabels(UNLABELED_ROOT, FILENAMES, SOFT_LABELS)

    lab_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    un_loader = DataLoader(unlabeled, batch_size=32, shuffle=True)

    student = Preprocess(SmallCNN(), size=64).to(device)
    n_params = count_params(student)
    print(f"Student parameters: {n_params:,}")
    assert n_params <= 500_000, f"Over cap: {n_params:,}"

    opt = torch.optim.Adam(student.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best_val = -1.0
    best_state = None
    for epoch in range(1, EPOCHS + 1):
        un_iter = iter(un_loader)
        total, ce_sum, kd_sum, count = 0.0, 0.0, 0.0, 0
        for x_lab, y_lab in lab_loader:
            try:
                x_un, t_logits = next(un_iter)
            except StopIteration:
                un_iter = iter(un_loader)
                x_un, t_logits = next(un_iter)
            x_lab = x_lab.to(device); y_lab = y_lab.to(device)
            x_un = x_un.to(device); t_logits = t_logits.to(device)
            loss, ce, kd = train_step(student, x_lab, y_lab, x_un, t_logits, opt)
            total += loss; ce_sum += ce; kd_sum += kd; count += 1
        scheduler.step()

        val_acc, val_loss = evaluate(student, val_loader, device)
        print(
            f"Epoch {epoch:3d}  loss={total / count:.3f}  "
            f"ce={ce_sum / count:.3f}  kd={kd_sum / count:.3f}  "
            f"lr={opt.param_groups[0]['lr']:.5f}  "
            f"val_acc={val_acc:.4f}  val_loss={val_loss:.4f}"
        )
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}

    # Restore the best-on-val checkpoint and serialize it for the server.
    if best_state is not None:
        student.load_state_dict(best_state)
    print(f"Best val_acc={best_val:.4f} — saving to model.pt")

    # TorchScript before saving so the server can load without train_baby
    # classes on its import path.
    torch.jit.save(torch.jit.script(student.cpu().eval()), "model.pt")
    print("Saved distilled model.pt — upload this to the leaderboard.")


if __name__ == "__main__":
    main()
