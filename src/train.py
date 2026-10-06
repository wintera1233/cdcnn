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
from src.loss import (EpochLoss, cross_entropy, mse_consistency,
                      supervised_contrastive)
from src.model import (AUGMENT_DISPLACEMENT, AUGMENT_LAMBDA, LAMBDA_MSE,
                       VARIANTS, build, parameter_breakdown, to_input)
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
              save_every_epoch: bool, learning_rate: float | None = None,
              normalizer_params: dict | None = None) -> dict:
    """Train one variant at one seed for `config['training']['epochs']` epochs.

    Returns the history; writes `epoch_XXX.pt` when `save_every_epoch` and always
    writes `final.pt`. Checkpoints are written, not frozen: freezing and hashing
    happen in `src.audit` once every run of the ladder is complete.

    `normalizer_params` replaces the Normal block fitted on `source_x`. Only
    `scripts/leak_diagnostic_scaler.py` passes it, to fit the scaler on data a
    source-only run may not see; such a run is target-informed by construction
    and cannot pass the leakage audit.
    """
    training = config["training"]
    optimizer_config = config["optimizer"]
    scheduler_config = config["scheduler"]

    if variant not in VARIANTS:
        raise ProtocolError(f"unknown variant {variant!r}")
    seed_everything(seed)
    # The Normal block is part of the variant's definition, not a free knob:
    # the 2x2 in proposal.md section 6 crosses it with the head.
    if normalizer_params is None:
        normalizer = normalize.fit(VARIANTS[variant]["normalizer"], source_x)
    else:
        if normalizer_params["kind"] != VARIANTS[variant]["normalizer"]:
            raise ProtocolError("injected normalizer does not match the variant")
        normalizer = dict(normalizer_params)
    normalised = normalize.apply(normalizer, source_x)
    labels = source_y

    # Fig. 2 places augmentation after the `Normal` block and concatenates the
    # augmented copies with the originals, keeping their labels (Table 1, step 1).
    # It is training-only: `src.evaluate` never calls this path.
    # Kept before augmentation: the signed generation block (v10.0) measures its
    # direction on Batch 1's own acquisition blocks, which are row slices of the
    # original 445 rows in file order.
    source_input = to_input(normalised)
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

    # Eq. (4): min(L_ce + lambda_MSE L_MSE + lambda_con L_con). A variant with no
    # generation block has no second branch, so both auxiliary terms are absent
    # and this reduces to the S2 cross-entropy every v7 and v8 cell trained on.
    generates = model.generation is not None
    lambda_mse = float(VARIANTS[variant].get("lambda_mse", LAMBDA_MSE))
    ce_on_generated = bool(VARIANTS[variant].get("ce_on_generated", False))
    # v11.0: lambda_con is zero unless the variant declares it, so every earlier
    # cell trains exactly as before.
    lambda_con = float(VARIANTS[variant].get("lambda_con", 0.0))
    temperature = VARIANTS[variant].get("temperature")
    if lambda_con > 0 and not generates:
        raise ProtocolError("L_con needs the generated branch; declare a block")

    signed = generates and model.generation.sign is not None
    if signed:
        label = {"ethanol": 1}[model.generation.direction]
        (first_a, first_b), (second_a, second_b) = augmentation.SOURCE_BLOCKS[label]
        if not (source_y[first_a:first_b] == label).all() or not (
                source_y[second_a:second_b] == label).all():
            raise ProtocolError("block layout does not match the source rows")
        source_input = source_input.to(device)

    def refresh_direction() -> None:
        """Batch 1's two acquisition blocks through the current stem, no grad."""
        with torch.no_grad():
            mid = model.stem(source_input)
        model.generation.set_direction(mid[first_a:first_b], mid[second_a:second_b])

    history: list[dict] = []
    for epoch in range(1, training["epochs"] + 1):
        model.train()
        if signed:
            refresh_direction()
        accumulator = EpochLoss()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            components: dict[str, torch.Tensor] = {}
            if not generates:
                logits = model(batch_x)
                loss = cross_entropy(logits, batch_y)
            else:
                # The block's Gaussian draws come from the global RNG, which
                # `seed_everything` fixes, so the pair is reproducible per seed.
                branches = model.forward_pair(batch_x)
                logits = branches.logits
                # Eq. (S2) writes L_ce over Phi alone; `ce_on_generated` is the
                # algorithm box's reading, averaging the two branches so the
                # term keeps its scale and only its target set changes.
                entropy = cross_entropy(logits, batch_y)
                if ce_on_generated:
                    entropy = 0.5 * (entropy + cross_entropy(
                        branches.logits_generated, batch_y))
                mse = mse_consistency(logits, branches.logits_generated)
                loss = entropy + lambda_mse * mse
                components = {"ce": entropy, "mse": mse}
                if lambda_con > 0:
                    # Eq. (S5) over Z^f = z_f U z_bar_f, the generated feature
                    # carrying its original's label; f is the L2 normalisation
                    # inside `supervised_contrastive`.
                    con = supervised_contrastive(
                        torch.cat([branches.features.flatten(1),
                                   branches.features_generated.flatten(1)]),
                        torch.cat([batch_y, batch_y]), float(temperature))
                    loss = loss + lambda_con * con
                    components["con"] = con
            loss.backward()
            optimizer.step()
            accumulator.update(loss, logits, batch_y, components)
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
               "normalizer": normalizer["kind"],
               "normalizer_fitted_on": ("injected (see the run's leak_diagnostic.json)"
                                        if normalizer_params is not None else "Batch 1"),
               "learning_rate": lr,
               "finished_at": utc_now(),
               "parameters": parameter_breakdown(model),
               "source_rows": int(len(source_y)),
               "training_rows": int(len(labels)),
               "augment": VARIANTS[variant].get("augment"),
               "generate": VARIANTS[variant].get("generate"),
               "lambda_mse": lambda_mse if generates else None,
               "lambda_con": lambda_con if lambda_con > 0 else None,
               "contrastive_temperature": (
                   float(temperature) if lambda_con > 0 else None),
               "ce_on_generated": ce_on_generated if generates else None,
               "negative_scale_fraction": (
                   model.generation.negative_scale_fraction if generates else None),
               "signed_generation": (
                   {"sign": model.generation.sign,
                    "direction": model.generation.direction,
                    "displacement": model.generation.displacement,
                    "final_epoch": model.generation.direction_report}
                   if signed else None),
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
