"""Training on Batch 1 only.

Nothing here can see a target batch: `src.data.load_target` demands a
`TargetAccessLog`, and this module never constructs one. The `Normal` block is
fitted on the source array passed in, which `scripts/run_baseline.py` takes from
`src.data.load_source`.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src import augment as augmentation, normalize
from src.data import N_CLASSES
from src.loss import EpochLoss, cross_entropy
from src.model import (AUGMENT_DISPLACEMENT, AUGMENT_LAMBDA, VARIANTS, build,
                       parameter_breakdown, to_input)
from src.protocol import ProtocolError, utc_now


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Requires CUBLAS_WORKSPACE_CONFIG=":4096:8", which docker/Dockerfile sets.
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def _loader(x: torch.Tensor, y: torch.Tensor, batch_size: int,
            generator: torch.Generator) -> DataLoader:
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True,
                      drop_last=False, num_workers=0, generator=generator)


def epoch_is_saved(epoch: int, dense_until: int = 20, then_every: int = 5) -> bool:
    """Every epoch early, then every fifth.

    The v7.0 curves put `R-txt`'s target peak at epoch 4 and its decay inside the
    first forty, so resolution matters early and not late. Saving all 100 epochs
    of every run would cost 14 GB against 28 GB free.
    """
    return epoch <= dense_until or epoch % then_every == 0


def train_one(variant: str, seed: int, config: dict, source_x: np.ndarray,
              source_y: np.ndarray, checkpoint_dir: Path, device: str,
              save_every_epoch: bool, learning_rate: float | None = None) -> dict:
    """Train one variant at one seed for `config['training']['epochs']` epochs.

    Returns the history; writes `epoch_XXX.pt` when `save_every_epoch` and always
    writes `final.pt`. Checkpoints are written, not frozen: freezing and hashing
    happen in `src.audit` once every run of the ladder is complete.
    """
    training = config["training"]
    optimizer_config = config["optimizer"]
    scheduler_config = config["scheduler"]

    if variant not in VARIANTS:
        raise ProtocolError(f"unknown variant {variant!r}")
    seed_everything(seed)
    # The Normal block is part of the variant's definition, not a free knob:
    # the 2x2 in proposal.md section 6 crosses it with the head.
    normalizer = normalize.fit(VARIANTS[variant]["normalizer"], source_x)
    normalised = normalize.apply(normalizer, source_x)
    labels = source_y

    # Fig. 2 places augmentation after the `Normal` block and concatenates the
    # augmented copies with the originals, keeping their labels (Table 1, step 1).
    # It is training-only: `src.evaluate` never calls this path.
    spec = VARIANTS[variant].get("augment")
    if spec is not None:
        generator = np.random.default_rng(seed)
        extra = augmentation.augment(
            normalised, source_y, generator,
            isotropic=spec["isotropic"], direction=spec["direction"],
            lam=AUGMENT_LAMBDA,
            displacement=spec.get("displacement", AUGMENT_DISPLACEMENT))
        normalised = np.concatenate([normalised, extra], axis=0)
        labels = np.concatenate([source_y, source_y], axis=0)

    x = to_input(normalised)
    # Labels on disk are 1..6; the model has 6 outputs indexed from zero.
    y = torch.as_tensor(labels, dtype=torch.int64) - 1
    if int(y.min()) < 0 or int(y.max()) >= N_CLASSES:
        raise ProtocolError("labels outside 1..6 after the zero-based shift")

    model = build(variant).to(device)
    lr = optimizer_config["lr"] if learning_rate is None else float(learning_rate)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr,
        momentum=optimizer_config["momentum"],
        weight_decay=optimizer_config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=scheduler_config["step_size"],
        gamma=scheduler_config["gamma"])

    generator = torch.Generator().manual_seed(seed)
    loader = _loader(x, y, training["batch_size"], generator)
    checkpoint_dir.mkdir(parents=True, exist_ok=False)

    history: list[dict] = []
    for epoch in range(1, training["epochs"] + 1):
        model.train()
        accumulator = EpochLoss()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = cross_entropy(logits, batch_y)
            loss.backward()
            optimizer.step()
            accumulator.update(loss, logits, batch_y)
        scheduler.step()
        record = {"epoch": epoch, "learning_rate": scheduler.get_last_lr()[0],
                  **accumulator.summary()}
        history.append(record)
        if save_every_epoch and epoch_is_saved(epoch):
            _save(checkpoint_dir / f"epoch_{epoch:03d}.pt", model, normalizer,
                  variant, seed, epoch, config, lr)

    _save(checkpoint_dir / "final.pt", model, normalizer, variant, seed,
          training["epochs"], config, lr)
    summary = {"variant": variant, "seed": seed, "device": device,
               "normalizer": normalizer["kind"], "learning_rate": lr,
               "finished_at": utc_now(),
               "parameters": parameter_breakdown(model),
               "source_rows": int(len(source_y)),
               "training_rows": int(len(labels)),
               "augment": VARIANTS[variant].get("augment"),
               "batches_per_epoch": int(history[-1]["batches"]),
               "final_train_accuracy": history[-1]["accuracy"],
               "final_train_loss_s2": history[-1]["loss_s2"],
               "history": history}
    (checkpoint_dir / "history.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _save(path: Path, model: nn.Module, normalizer: dict, variant: str,
          seed: int, epoch: int, config: dict, learning_rate: float) -> None:
    torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "normalizer": normalizer, "variant": variant, "seed": seed,
                "epoch": epoch, "learning_rate": learning_rate,
                "implementation_version": config["implementation_version"],
                "written_at": utc_now()}, path)
