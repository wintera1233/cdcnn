"""Cross-validation on Batch 1, to ask whether a source-only stopping rule exists.

Every configuration this branch has trained saturates Batch 1 within about five
epochs while its target accuracy keeps moving for tens more, and the epoch at
which target accuracy peaks lands anywhere between 4 and 100 at Batch 1
accuracies from 0.58 to 0.975. No threshold on training accuracy can locate
that. Held-out accuracy is a different quantity, and this module measures it.

Nothing here opens a target batch.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src import normalize
from src.loss import cross_entropy
from src.model import VARIANTS, build, to_input
from src.protocol import ProtocolError
from src.train import seed_everything

# Folds are fixed by this constant so that every variant, learning rate and seed
# sees the identical partition; only the model initialisation varies with seed.
FOLD_SEED = 20260923
N_FOLDS = 5


def stratified_folds(y: np.ndarray, n_folds: int = N_FOLDS,
                     seed: int = FOLD_SEED) -> list[np.ndarray]:
    """Partition indices into `n_folds`, keeping each class's share even.

    Ethylene has 30 of the 445 source rows, so an unstratified split would leave
    folds with as few as two of them.
    """
    rng = np.random.default_rng(seed)
    assignment = np.empty(len(y), dtype=np.int64)
    for label in np.unique(y):
        where = np.flatnonzero(y == label)
        rng.shuffle(where)
        assignment[where] = np.arange(len(where)) % n_folds
    folds = [np.flatnonzero(assignment == f) for f in range(n_folds)]
    if sum(len(f) for f in folds) != len(y):
        raise ProtocolError("folds do not partition the source batch")
    return folds


def fold_curve(variant: str, seed: int, learning_rate: float, config: dict,
               x: np.ndarray, y: np.ndarray, train_index: np.ndarray,
               held_out_index: np.ndarray, device: str) -> dict[str, list[float]]:
    """Held-out accuracy **and cross-entropy** after every epoch, for one fold.

    Accuracy on 89 held-out rows moves in steps of 1/89 and saturates within a
    few epochs, so it is a blunt instrument. Cross-entropy is continuous and its
    classic overfitting signature - a minimum followed by a rise while accuracy
    stays flat - appears earlier. Both are recorded.

    The `Normal` block is fitted on the training part of the fold only, so the
    held-out part is never used to choose anything.
    """
    if variant not in VARIANTS:
        raise ProtocolError(f"unknown variant {variant!r}")
    training = config["training"]
    optimizer_config = config["optimizer"]
    scheduler_config = config["scheduler"]

    seed_everything(seed)
    params = normalize.fit(VARIANTS[variant]["normalizer"], x[train_index],
                           rows=train_index)
    train_x = to_input(normalize.apply(params, x[train_index])).to(device)
    train_y = (torch.as_tensor(y[train_index], dtype=torch.int64) - 1).to(device)
    held_x = to_input(normalize.apply(params, x[held_out_index])).to(device)
    held_y = (torch.as_tensor(y[held_out_index], dtype=torch.int64) - 1).to(device)

    model = build(variant).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate,
        momentum=optimizer_config["momentum"],
        weight_decay=optimizer_config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=scheduler_config["step_size"],
        gamma=scheduler_config["gamma"])
    loader = DataLoader(TensorDataset(train_x, train_y),
                        batch_size=training["batch_size"], shuffle=True,
                        num_workers=0, generator=torch.Generator().manual_seed(seed))

    accuracies: list[float] = []
    losses: list[float] = []
    for _ in range(training["epochs"]):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = cross_entropy(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            logits = model(held_x)
            accuracies.append(float((logits.argmax(dim=1) == held_y).float().mean()))
            losses.append(float(cross_entropy(logits, held_y)))
    return {"accuracy": accuracies, "loss": losses}


def summarise(curves: list[dict[str, list[float]]]) -> dict:
    """Mean held-out accuracy and loss per epoch, with the peak and the minimum."""
    accuracy = np.asarray([c["accuracy"] for c in curves], dtype=np.float64)
    loss = np.asarray([c["loss"] for c in curves], dtype=np.float64)
    accuracy_mean = accuracy.mean(axis=0)
    loss_mean = loss.mean(axis=0)
    peak = int(accuracy_mean.argmax())
    trough = int(loss_mean.argmin())
    return {"accuracy_per_epoch_mean": accuracy_mean.tolist(),
            "accuracy_per_epoch_sd": accuracy.std(axis=0, ddof=1).tolist(),
            "loss_per_epoch_mean": loss_mean.tolist(),
            "loss_per_epoch_sd": loss.std(axis=0, ddof=1).tolist(),
            "peak_epoch": peak + 1,
            "peak_accuracy": float(accuracy_mean[peak]),
            "final_accuracy": float(accuracy_mean[-1]),
            "min_loss_epoch": trough + 1,
            "min_loss": float(loss_mean[trough]),
            "final_loss": float(loss_mean[-1]),
            "folds": len(curves)}
