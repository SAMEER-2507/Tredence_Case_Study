"""
Self-Pruning Neural Network on CIFAR-10
Tredence AI Engineering Case Study

Architecture: Feed-forward net with custom PrunableLinear layers.
Mechanism: Each weight has a learnable gate (sigmoid-gated).
            L1 penalty on gates drives them toward 0 → sparsity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# ─────────────────────────────────────────────
# PART 1: PrunableLinear Layer
# ─────────────────────────────────────────────

class PrunableLinear(nn.Module):
    """
    Custom linear layer with per-weight learnable gates.

    For each weight w_ij, we have a gate score s_ij (same shape as weights).
    gate = sigmoid(s_ij)  ∈ (0, 1)
    pruned_weight = w_ij * gate_ij

    During training, L1 penalty on gates pushes gates → 0,
    effectively "pruning" low-importance weights.
    Gradients flow through both weight and gate_scores via autograd.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Standard weight + bias
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Learnable gate scores — same shape as weight
        # Initialized near 0 so gates start around sigmoid(0) = 0.5
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))

        # Kaiming init for weights (good for ReLU nets)
        nn.init.kaiming_uniform_(self.weight, a=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: compute gates ∈ (0, 1)
        gates = torch.sigmoid(self.gate_scores)

        # Step 2: element-wise multiply weights with gates
        pruned_weights = self.weight * gates

        # Step 3: standard linear operation (gradients flow to both weight & gate_scores)
        return F.linear(x, pruned_weights, self.bias)

    def get_gates(self) -> torch.Tensor:
        """Return current gate values (detached, for inspection)."""
        return torch.sigmoid(self.gate_scores).detach()

    def sparsity(self, threshold: float = 1e-2) -> float:
        """Fraction of weights with gate < threshold (i.e., effectively pruned)."""
        gates = self.get_gates()
        return (gates < threshold).float().mean().item()


# ─────────────────────────────────────────────
# Network Definition
# ─────────────────────────────────────────────

class SelfPruningNet(nn.Module):
    """
    Feed-forward classifier for CIFAR-10 (32x32x3 = 3072 input dims).
    All linear layers replaced with PrunableLinear.
    """

    def __init__(self):
        super().__init__()
        self.fc1 = PrunableLinear(3072, 1024)
        self.fc2 = PrunableLinear(1024, 512)
        self.fc3 = PrunableLinear(512, 256)
        self.fc4 = PrunableLinear(256, 10)   # 10 CIFAR-10 classes

        self.dropout = nn.Dropout(0.3)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)           # flatten: (B, 3072)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x

    def prunable_layers(self):
        """Yield all PrunableLinear layers."""
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                yield module

    def sparsity_loss(self) -> torch.Tensor:
        """
        L1 norm of ALL gate values across all PrunableLinear layers.
        Since gates = sigmoid(score) > 0 always, |gate| = gate.
        Minimizing this sum → pushes gates toward 0 → sparsity.
        """
        total = torch.tensor(0.0, requires_grad=True)
        for layer in self.prunable_layers():
            gates = torch.sigmoid(layer.gate_scores)
            total = total + gates.sum()
        return total

    def overall_sparsity(self, threshold: float = 1e-2) -> float:
        """Overall % of weights pruned across all layers."""
        all_gates = torch.cat([
            layer.get_gates().flatten()
            for layer in self.prunable_layers()
        ])
        return (all_gates < threshold).float().mean().item() * 100

    def all_gate_values(self) -> torch.Tensor:
        """Concatenate all gate values for plotting."""
        return torch.cat([
            layer.get_gates().flatten()
            for layer in self.prunable_layers()
        ])


# ─────────────────────────────────────────────
# PART 3: Data Loading
# ─────────────────────────────────────────────

def get_dataloaders(batch_size: int = 128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        ),
    ])

    train_ds = datasets.CIFAR10(root="./data", train=True,  download=True, transform=transform)
    test_ds  = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=256,        shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader


# ─────────────────────────────────────────────
# Training & Evaluation
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, device, lam):
    model.train()
    total_loss = total_correct = total_samples = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)

        # Total Loss = CrossEntropy + λ * L1_sparsity_on_gates
        clf_loss = F.cross_entropy(logits, labels)
        sparse_loss = model.sparsity_loss()
        loss = clf_loss + lam * sparse_loss

        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        total_correct  += (preds == labels).sum().item()
        total_samples  += labels.size(0)
        total_loss     += loss.item() * labels.size(0)

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_correct = total_samples = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
    return total_correct / total_samples


def run_experiment(lam: float, epochs: int, device, train_loader, test_loader):
    """Train one model with given lambda. Return (test_acc, sparsity%)."""
    print(f"\n{'='*55}")
    print(f"  λ = {lam}  |  epochs = {epochs}")
    print(f"{'='*55}")

    model = SelfPruningNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, lam)
        scheduler.step()
        sparsity = model.overall_sparsity()

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"Loss {train_loss:.4f} | "
                  f"Train Acc {train_acc*100:.1f}% | "
                  f"Sparsity {sparsity:.1f}%")

    test_acc = evaluate(model, test_loader, device)
    final_sparsity = model.overall_sparsity()

    print(f"\n  ✓ Test Accuracy : {test_acc*100:.2f}%")
    print(f"  ✓ Sparsity Level: {final_sparsity:.2f}%")

    return test_acc, final_sparsity, model


def plot_gate_distribution(model, lam, save_path="gate_distribution.png"):
    """Plot histogram of final gate values for given model."""
    gates = model.all_gate_values().cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(gates, bins=100, color="steelblue", edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Gate Value (sigmoid output)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Gate Value Distribution  |  λ = {lam}", fontsize=14)
    ax.axvline(x=0.01, color="red", linestyle="--", linewidth=1.5,
               label="Prune threshold (0.01)")
    ax.legend()

    # Annotation: % pruned
    pruned_pct = (gates < 0.01).mean() * 100
    ax.text(0.15, 0.85, f"Pruned: {pruned_pct:.1f}%",
            transform=ax.transAxes, fontsize=11,
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\n  Plot saved → {save_path}")
    plt.close()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    EPOCHS = 30           # Increase to 50+ for better accuracy
    BATCH_SIZE = 128

    train_loader, test_loader = get_dataloaders(BATCH_SIZE)

    # Three lambda values: low / medium / high
    lambdas = [1e-5, 1e-4, 1e-3]

    results = []
    best_model = None
    best_lam = None

    for lam in lambdas:
        acc, sparsity, model = run_experiment(lam, EPOCHS, device, train_loader, test_loader)
        results.append({"lambda": lam, "test_acc": acc * 100, "sparsity": sparsity})
        if best_model is None:
            best_model = model     # Save first (lowest λ) as "best balanced" model
            best_lam = lam

    # Print summary table
    print("\n" + "="*55)
    print(f"{'Lambda':<12} {'Test Accuracy':>15} {'Sparsity (%)':>14}")
    print("-"*55)
    for r in results:
        print(f"{r['lambda']:<12} {r['test_acc']:>14.2f}% {r['sparsity']:>13.2f}%")
    print("="*55)

    # Plot gate distribution for best model
    plot_gate_distribution(best_model, best_lam, save_path="gate_distribution.png")

    print("\nDone. Submit: self_pruning_network.py + report.md + gate_distribution.png")
