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
               held_out_index: np.ndarray, device: str) -> list[float]:
    """Held-out accuracy after every epoch, for one fold.

    The `Normal` block is fitted on the training part of the fold only, so the
    held-out part is never used to choose anything.
    """
    if variant not in VARIANTS:
        raise ProtocolError(f"unknown variant {variant!r}")
    training = config["training"]
    optimizer_config = config["optimizer"]
    scheduler_config = config["scheduler"]

    seed_everything(seed)
    params = normalize.fit(VARIANTS[variant]["normalizer"], x[train_index])
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
            predicted = model(held_x).argmax(dim=1)
        accuracies.append(float((predicted == held_y).float().mean()))
    return accuracies


def summarise(curves: list[list[float]]) -> dict:
    """Mean held-out accuracy per epoch, and where it peaks."""
    array = np.asarray(curves, dtype=np.float64)
    mean = array.mean(axis=0)
    peak = int(mean.argmax())
    return {"per_epoch_mean": mean.tolist(),
            "per_epoch_sd": array.std(axis=0, ddof=1).tolist(),
            "peak_epoch": peak + 1,
            "peak_accuracy": float(mean[peak]),
            "final_accuracy": float(mean[-1]),
            "folds": len(curves)}
