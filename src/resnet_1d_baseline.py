"""Canonical source-only one-dimensional ResNet B0 for the gas drift dataset."""

from __future__ import annotations

import copy
import json
import math
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.cv_folds import load_folds
from src.pca_analysis import load_batch, save_json, sha256


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )
        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.main(x) + self.shortcut(x))


class GasResNet1D(nn.Module):
    def __init__(self):
        super().__init__()
        channels = (32, 64, 128, 256, 128)
        blocks, incoming = [], 1
        for outgoing in channels:
            blocks.append(ResidualBlock1D(incoming, outgoing))
            incoming = outgoing
        self.backbone = nn.Sequential(*blocks)
        self.flatten = nn.Flatten()
        self.fc128 = nn.Sequential(nn.Linear(128 * 128, 128), nn.BatchNorm1d(128))
        self.fc6 = nn.Linear(128, 6)

    def forward_features(self, x):
        return self.backbone(x)

    def forward(self, x):
        return self.fc6(self.fc128(self.flatten(self.forward_features(x))))


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def reshape(x):
    return x.astype(np.float32, copy=False).reshape(-1, 1, 128)


def parameter_counts(model):
    count = lambda module: sum(p.numel() for p in module.parameters())
    result = {
        "residual_backbone": count(model.backbone),
        "FC128_including_batch_norm": count(model.fc128),
        "FC128_linear_weights_only": model.fc128[0].weight.numel(),
        "FC6": count(model.fc6),
    }
    result["total"] = count(model)
    assert result["residual_backbone"] + result["FC128_including_batch_norm"] + result["FC6"] == result["total"]
    return result


def _loader(x, y, cfg, seed, shuffle):
    ds = TensorDataset(torch.from_numpy(reshape(x)), torch.from_numpy(y.astype(np.int64) - 1))
    generator = torch.Generator().manual_seed(seed)
    return ds, DataLoader(ds, batch_size=cfg["batch_size"], shuffle=shuffle, generator=generator)


def build_optimizer(model, cfg):
    """Build the configured optimizer without changing shared learning settings."""
    common = {"lr": cfg["learning_rate"], "weight_decay": cfg["weight_decay"]}
    if cfg["optimizer"] == "SGD":
        return torch.optim.SGD(model.parameters(), momentum=cfg["momentum"], **common)
    if cfg["optimizer"] == "Adam":
        return torch.optim.Adam(model.parameters(), **common)
    raise ValueError(f"Unsupported optimizer: {cfg['optimizer']!r}")


def build_scheduler(optimizer, scheduler_cfg):
    """Build one preregistered scheduler candidate."""
    scheduler_type = scheduler_cfg["type"]
    if scheduler_type == "none":
        return None
    if scheduler_type == "StepLR":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=scheduler_cfg["step_size"], gamma=scheduler_cfg["gamma"])
    raise ValueError(f"Unsupported scheduler: {scheduler_type!r}")


def validate_scheduler_candidates(cfg):
    candidates = cfg["scheduler_candidates"]
    names = [candidate["name"] for candidate in candidates]
    if len(candidates) != 2 or len(names) != len(set(names)):
        raise ValueError("Exactly two uniquely named scheduler candidates are required")
    controls = [candidate for candidate in candidates if candidate["type"] == "none"]
    step_lr = [candidate for candidate in candidates if candidate["type"] == "StepLR"]
    if len(controls) != 1 or controls[0].get("role") != "control" or len(step_lr) != 1:
        raise ValueError("Scheduler candidates must contain one no-scheduler control and one StepLR")
    step_cfg = step_lr[0]
    if step_cfg.get("step_size") != 10 or step_cfg.get("gamma") != 0.1:
        raise ValueError("The requested StepLR requires step_size=10 and gamma=0.1")
    expected = list(range(step_cfg["step_size"], cfg["training"]["max_epochs"], step_cfg["step_size"]))
    if step_cfg.get("decay_epochs") != expected or expected != [10, 20]:
        raise ValueError(f"StepLR decay epochs must resolve to [10, 20], got {expected}")
    return candidates


@torch.no_grad()
def predict_logits(model, x, batch_size):
    model.eval()
    result = []
    for start in range(0, len(x), batch_size):
        result.append(model(torch.from_numpy(reshape(x[start:start + batch_size]))).cpu())
    return torch.cat(result).numpy()


def train_cv_model(x_train, y_train, x_valid, y_valid, cfg, scheduler_cfg, seed):
    seed_everything(seed)
    model = GasResNet1D()
    ds, loader = _loader(x_train, y_train, cfg, seed, cfg["shuffle"])
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, scheduler_cfg)
    criterion = nn.CrossEntropyLoss()
    history, best_state, best_epoch, best_accuracy, best_loss, stale = [], None, 0, -1.0, math.inf, 0
    for epoch in range(1, cfg["max_epochs"] + 1):
        learning_rate = float(optimizer.param_groups[0]["lr"])
        model.train(); loss_sum = 0.0; correct = 0
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb); loss = criterion(logits, yb)
            loss.backward(); optimizer.step()
            loss_sum += loss.item() * len(yb); correct += (logits.argmax(1) == yb).sum().item()
        valid_logits = predict_logits(model, x_valid, cfg["batch_size"])
        valid_pred = valid_logits.argmax(1) + 1
        valid_accuracy = float(accuracy_score(y_valid, valid_pred))
        valid_loss = float(criterion(torch.from_numpy(valid_logits), torch.from_numpy(y_valid.astype(np.int64) - 1)).item())
        improved = valid_accuracy > best_accuracy or (valid_accuracy == best_accuracy and valid_loss < best_loss)
        if improved:
            best_accuracy, best_loss, best_epoch = valid_accuracy, valid_loss, epoch
            best_state, stale = copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if scheduler is not None and epoch < cfg["max_epochs"]:
            scheduler.step()
        learning_rate_after_step = float(optimizer.param_groups[0]["lr"])
        history.append({"epoch": epoch, "training_loss": loss_sum / len(ds), "training_accuracy": correct / len(ds),
                        "validation_loss": valid_loss, "validation_accuracy": valid_accuracy,
                        "learning_rate": learning_rate, "learning_rate_after_scheduler_step": learning_rate_after_step,
                        "scheduler_step_applied": learning_rate_after_step != learning_rate,
                        "is_selected_epoch": epoch == best_epoch})
        if stale >= cfg["patience"]:
            break
    model.load_state_dict(best_state)
    for row in history:
        row["is_selected_epoch"] = row["epoch"] == best_epoch
    return model, history, best_epoch, best_accuracy, best_loss


def train_fixed_epochs(x, y, cfg, scheduler_cfg, seed, epochs):
    seed_everything(seed)
    model = GasResNet1D()
    ds, loader = _loader(x, y, cfg, seed, cfg["shuffle"])
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, scheduler_cfg)
    criterion = nn.CrossEntropyLoss(); history = []
    for epoch in range(1, epochs + 1):
        learning_rate = float(optimizer.param_groups[0]["lr"])
        model.train(); loss_sum = 0.0; correct = 0
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb); loss = criterion(logits, yb)
            loss.backward(); optimizer.step()
            loss_sum += loss.item() * len(yb); correct += (logits.argmax(1) == yb).sum().item()
        if scheduler is not None and epoch < epochs:
            scheduler.step()
        learning_rate_after_step = float(optimizer.param_groups[0]["lr"])
        history.append({"epoch": epoch, "training_loss": loss_sum / len(ds), "training_accuracy": correct / len(ds),
                        "learning_rate": learning_rate, "learning_rate_after_scheduler_step": learning_rate_after_step,
                        "scheduler_step_applied": learning_rate_after_step != learning_rate})
    return model, history


def prediction_proportions(frame):
    counts = frame.groupby(["batch", "predicted_gas_label"]).size().unstack(fill_value=0).reindex(columns=range(1, 7), fill_value=0)
    rows = []
    for batch, values in counts.iterrows():
        total = int(values.sum())
        for label, count in values.items():
            rows.append({"batch": int(batch), "predicted_gas_label": int(label), "count": int(count), "proportion": float(count / total)})
    return pd.DataFrame(rows)


def load_adam_comparison(root, cfg):
    comparison_dir = (root / cfg["adam_comparison"]["run"]).resolve()
    paths = {name: comparison_dir / name for name in ("configuration.json", "summary.json", "batch_metrics.csv")}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Adam comparison artifacts: {missing}")
    adam_cfg = json.loads(paths["configuration.json"].read_text(encoding="utf-8"))
    adam_summary = json.loads(paths["summary.json"].read_text(encoding="utf-8"))
    adam_metrics = pd.read_csv(paths["batch_metrics.csv"])
    if adam_cfg["training"]["optimizer"] != "Adam":
        raise ValueError("Configured comparison run is not an Adam run")
    for key in ("learning_rate", "weight_decay"):
        if adam_cfg["training"][key] != cfg["training"][key]:
            raise ValueError(f"Adam comparison has a different training.{key}")
    for key in ("architecture", "source_batch", "saved_cv_folds"):
        if adam_cfg[key] != cfg[key]:
            raise ValueError(f"Adam comparison has a different {key}")
    return comparison_dir, paths, adam_summary, adam_metrics


def comparison_table(summary, metrics, adam_summary, adam_metrics):
    adam_by_batch = adam_metrics.set_index("batch")
    rows = [{"scope": "source_cv_mean", "batch": pd.NA,
             "sgd_accuracy": summary["source_cv_mean_accuracy"],
             "adam_accuracy": adam_summary["source_cv_mean_accuracy"]}]
    for row in metrics.itertuples():
        rows.append({"scope": "target_batch", "batch": int(row.batch), "sgd_accuracy": float(row.accuracy),
                     "adam_accuracy": float(adam_by_batch.loc[row.batch, "accuracy"])})
    rows.extend([
        {"scope": "target_unweighted_mean", "batch": pd.NA,
         "sgd_accuracy": summary["target_unweighted_mean_accuracy"],
         "adam_accuracy": adam_summary["target_unweighted_mean_accuracy"]},
        {"scope": "target_pooled", "batch": pd.NA,
         "sgd_accuracy": summary["target_pooled_accuracy"],
         "adam_accuracy": adam_summary["target_pooled_accuracy"]},
    ])
    result = pd.DataFrame(rows)
    result["sgd_minus_adam"] = result["sgd_accuracy"] - result["adam_accuracy"]
    return result


def main(config_path="configs/resnet_1d.json"):
    root = Path.cwd().resolve(); config_file = (root / config_path).resolve()
    cfg = json.loads(config_file.read_text()); data_file = (root / cfg["pca_config"]).resolve()
    data_cfg = json.loads(data_file.read_text()); fold_file = (root / cfg["saved_cv_folds"]).resolve()
    if cfg["training"]["optimizer"] != "SGD" or cfg["training"].get("momentum") != 0.9:
        raise ValueError("Canonical SGD B0 requires optimizer='SGD' and momentum=0.9")
    scheduler_candidates = validate_scheduler_candidates(cfg)
    adam_dir, adam_paths, adam_summary, adam_metrics = load_adam_comparison(root, cfg)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_canonical_1d_resnet_b0_sgd"
    out = root / data_cfg["output_root"] / run_id; out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(cfg["training"]["num_threads"])

    arrays, labels, lines, inputs = {}, {}, {}, []
    ds_cfg = data_cfg["dataset"]
    for batch in ds_cfg["expected_batch_ids"]:
        path = root / ds_cfg["path"] / ds_cfg["batch_file_pattern"].format(batch_id=batch)
        arrays[batch], labels[batch], lines[batch] = load_batch(path, batch, 128)
        inputs.append({"role": "dataset_batch", "batch": batch, "path": str(path.relative_to(root)), "bytes": path.stat().st_size,
                       "sha256": sha256(path), "records": len(arrays[batch])})
    observed = (sum(map(len, arrays.values())), len(arrays), len(np.unique(np.concatenate(list(labels.values())))), arrays[1].shape[1])
    if observed != (13910, 10, 6, 128) or not np.isfinite(np.vstack(list(arrays.values()))).all():
        raise ValueError(f"Dataset validation failed: {observed}")

    source = cfg["source_batch"]
    fold_frame, fold_ids = load_folds(fold_file, arrays[source], lines[source], labels[source], source)
    fold_frame.to_csv(out / "cv_fold_assignments.csv", index=False)
    inputs.append({"role": "saved_cv_folds", "path": str(fold_file.relative_to(root)), "bytes": fold_file.stat().st_size,
                   "sha256": sha256(fold_file), "records": len(fold_frame)})
    for name, path in adam_paths.items():
        inputs.append({"role": f"adam_comparison_{name.removesuffix('.json').removesuffix('.csv')}",
                       "path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha256(path)})

    shape_model = GasResNet1D(); dummy = torch.zeros(2, 1, 128)
    with torch.no_grad():
        feature_shape = list(shape_model.forward_features(dummy).shape); output_shape = list(shape_model(dummy).shape)
    shape_check = {"input": list(dummy.shape), "backbone_output": feature_shape,
                   "flattened_features_per_sample": int(np.prod(feature_shape[1:])), "output": output_shape}
    expected_shape = {"input": [2, 1, 128], "backbone_output": [2, 128, 128],
                      "flattened_features_per_sample": 16384, "output": [2, 6]}
    if shape_check != expected_shape: raise AssertionError(shape_check)

    cv_rows, histories, candidate_rows = [], [], []
    for candidate_order, scheduler_cfg in enumerate(scheduler_candidates):
        candidate_fold_rows = []
        for fold in range(1, cfg["validation"]["folds"] + 1):
            train_idx, valid_idx = np.flatnonzero(fold_ids != fold), np.flatnonzero(fold_ids == fold)
            scaler = StandardScaler().fit(arrays[source][train_idx])
            model, history, best_epoch, best_accuracy, best_loss = train_cv_model(
                scaler.transform(arrays[source][train_idx]), labels[source][train_idx],
                scaler.transform(arrays[source][valid_idx]), labels[source][valid_idx],
                cfg["training"], scheduler_cfg, cfg["random_seed"] + fold)
            pred = predict_logits(model, scaler.transform(arrays[source][valid_idx]), cfg["training"]["batch_size"]).argmax(1) + 1
            measured_accuracy = float(accuracy_score(labels[source][valid_idx], pred))
            if measured_accuracy != best_accuracy:
                raise AssertionError("Restored CV model does not reproduce selected accuracy")
            row = {"scheduler": scheduler_cfg["name"], "scheduler_type": scheduler_cfg["type"], "fold": fold,
                   "train_samples": len(train_idx), "validation_samples": len(valid_idx), "overlap_samples": 0,
                   "scaler_fit_samples": len(train_idx), "epochs_run": len(history), "selected_epoch": best_epoch,
                   "stopped_early": len(history) < cfg["training"]["max_epochs"], "accuracy": measured_accuracy,
                   "selected_validation_loss": best_loss}
            candidate_fold_rows.append(row); cv_rows.append(row)
            histories.extend({"scheduler": scheduler_cfg["name"], "scheduler_type": scheduler_cfg["type"],
                              "fold": fold, **history_row} for history_row in history)
        candidate_fold_df = pd.DataFrame(candidate_fold_rows)
        candidate_rows.append({"scheduler": scheduler_cfg["name"], "scheduler_type": scheduler_cfg["type"],
                               "candidate_order": candidate_order,
                               "mean_cv_accuracy": float(candidate_fold_df.accuracy.mean()),
                               "std_cv_accuracy": float(candidate_fold_df.accuracy.std(ddof=1)),
                               "mean_selected_validation_loss": float(candidate_fold_df.selected_validation_loss.mean()),
                               "selected_epochs": int(math.floor(float(candidate_fold_df.selected_epoch.median()) + 0.5))})
    candidate_df = pd.DataFrame(candidate_rows).sort_values(
        ["mean_cv_accuracy", "mean_selected_validation_loss", "candidate_order"],
        ascending=[False, True, True], kind="stable").reset_index(drop=True)
    selected_scheduler_name = str(candidate_df.iloc[0].scheduler)
    selected_scheduler = next(candidate for candidate in scheduler_candidates if candidate["name"] == selected_scheduler_name)
    candidate_df["selected"] = candidate_df.scheduler == selected_scheduler_name
    all_cv_df = pd.DataFrame(cv_rows)
    all_history_df = pd.DataFrame(histories)
    all_cv_df["selected_scheduler"] = all_cv_df.scheduler == selected_scheduler_name
    all_history_df["selected_scheduler"] = all_history_df.scheduler == selected_scheduler_name
    all_cv_df.to_csv(out / "scheduler_candidate_cv_fold_results.csv", index=False)
    all_history_df.to_csv(out / "scheduler_candidate_cv_training_history.csv", index=False)
    candidate_df.to_csv(out / "scheduler_candidate_cv_summary.csv", index=False)
    cv_df = all_cv_df[all_cv_df.selected_scheduler].copy()
    cv_df.to_csv(out / "cv_fold_results.csv", index=False)
    all_history_df[all_history_df.selected_scheduler].to_csv(out / "cv_training_history.csv", index=False)
    selected_epochs = int(math.floor(float(cv_df.selected_epoch.median()) + 0.5))

    final_scaler = StandardScaler().fit(arrays[source])
    final_model, final_history = train_fixed_epochs(final_scaler.transform(arrays[source]), labels[source], cfg["training"],
                                                    selected_scheduler, cfg["random_seed"], selected_epochs)
    torch.save({"model_state_dict": final_model.state_dict(), "scaler_mean": final_scaler.mean_, "scaler_scale": final_scaler.scale_,
                "architecture": cfg["architecture"], "selected_epochs": selected_epochs,
                "canonical_baseline": "1D ResNet B0", "optimizer": cfg["training"]["optimizer"],
                "optimizer_parameters": {"learning_rate": cfg["training"]["learning_rate"],
                                         "weight_decay": cfg["training"]["weight_decay"],
                                         "momentum": cfg["training"]["momentum"]},
                "scheduler": selected_scheduler}, out / "model.pt")
    pd.DataFrame(final_history).to_csv(out / "final_training_history.csv", index=False)

    predictions, metrics, confusion_rows = [], [], []
    for batch in range(2, 11):
        pred = predict_logits(final_model, final_scaler.transform(arrays[batch]), cfg["training"]["batch_size"]).argmax(1) + 1
        correct = pred == labels[batch]
        metrics.append({"batch": batch, "samples": len(pred), "correct": int(correct.sum()), "accuracy": float(correct.mean())})
        predictions.append(pd.DataFrame({"sample_id": [f"batch{batch}:line{v}" for v in lines[batch]], "batch": batch,
            "line_number": lines[batch], "true_gas_label": labels[batch], "predicted_gas_label": pred, "correct": correct}))
        cm = confusion_matrix(labels[batch], pred, labels=range(1, 7))
        confusion_rows.extend({"batch": batch, "true_gas_label": true, "predicted_gas_label": predicted, "count": int(cm[true-1, predicted-1])}
                              for true in range(1, 7) for predicted in range(1, 7))
    metric_df = pd.DataFrame(metrics); metric_df.to_csv(out / "batch_metrics.csv", index=False)
    prediction_df = pd.concat(predictions, ignore_index=True)
    prediction_df.to_csv(out / "predictions.csv.gz", index=False, compression="gzip")
    prediction_proportions(prediction_df).to_csv(out / "class_prediction_proportions.csv", index=False)
    confusion_df = pd.DataFrame(confusion_rows)
    confusion_df.to_csv(out / "confusion_matrices.csv", index=False)
    pooled_accuracy = float(prediction_df.correct.mean())
    summary = {"experiment_id": "B0", "canonical_baseline": "1D ResNet B0", "legacy_2d_framework_banned": True,
               "parameter_counts": parameter_counts(shape_model), "tensor_shape_check": shape_check,
               "source_cv_mean_accuracy": float(cv_df.accuracy.mean()), "source_cv_std_accuracy": float(cv_df.accuracy.std(ddof=1)),
               "source_selected_epochs": selected_epochs, "target_unweighted_mean_accuracy": float(metric_df.accuracy.mean()),
               "target_pooled_accuracy": pooled_accuracy, "target_total_samples": int(len(prediction_df)),
               "selected_scheduler": selected_scheduler,
               "scheduler_selection_scope": "Batch 1 cross-validation only",
               "scheduler_candidate_cv": candidate_df.to_dict(orient="records")}
    comparison_df = comparison_table(summary, metric_df, adam_summary, adam_metrics)
    comparison_df.to_csv(out / "adam_comparison.csv", index=False)
    summary["optimizer"] = {"name": "SGD", "learning_rate": cfg["training"]["learning_rate"],
                            "weight_decay": cfg["training"]["weight_decay"], "momentum": cfg["training"]["momentum"]}
    summary["adam_comparison_run"] = str(adam_dir.relative_to(root))
    save_json(out / "summary.json", summary)
    resolved = json.loads(json.dumps(cfg)); resolved["resolved_dataset_configuration"] = data_cfg
    save_json(out / "configuration.json", resolved); save_json(out / "input_manifest.json", inputs)
    save_json(out / "software_versions.json", {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__, "torch": torch.__version__, "platform": platform.platform()})
    try: revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception: revision = None
    save_json(out / "run_manifest.json", {"run_id": run_id, "experiment_id": "B0", "canonical_baseline": "1D ResNet B0",
        "created_utc": datetime.now(timezone.utc).isoformat(), "command": f"{sys.executable} scripts/run_resnet_1d.py --config {config_path}",
        "git_revision": revision, "input_config_sha256": sha256(config_file), "data_config_sha256": sha256(data_file),
        "preprocessing_fit_scope": "Each CV scaler fit on that Batch 1 training fold only; final scaler fit on all Batch 1; targets transform only.",
        "selection_scope": "Scheduler, early stopping, and epoch selection use Batch 1 folds only. Batches 2-10 are evaluated once after scheduler selection and final fitting.",
        "sample_policy": "All valid samples retained; no deduplication, clipping, imputation, augmentation, or resampling.",
        "exclusions": [], "legacy_note": "The Conv2d 16x8 framework is banned; obsolete runs are quarantined and are not canonical.", "summary": summary})
    write_report(out, cfg, summary, candidate_df, cv_df, metric_df, prediction_proportions(prediction_df), confusion_df, comparison_df)
    save_json(out / "output_inventory.json", sorted(p.name for p in out.iterdir() if p.name != "output_inventory.json"))
    return out


def write_report(out, cfg, summary, scheduler_candidates, cv, metrics, proportions, confusions, comparison):
    scheduler_rows = "\n".join(
        f"| {r.scheduler} | {r.scheduler_type} | {r.mean_cv_accuracy:.6f} | {r.mean_selected_validation_loss:.6f} | "
        f"{int(r.selected_epochs)} | {'yes' if r.selected else 'no'} |" for r in scheduler_candidates.itertuples())
    cv_rows = "\n".join(f"| {int(r.fold)} | {int(r.train_samples)} | {int(r.validation_samples)} | {int(r.epochs_run)} | {int(r.selected_epoch)} | {r.accuracy:.6f} |" for r in cv.itertuples())
    target_rows = "\n".join(f"| {int(r.batch)} | {int(r.samples)} | {int(r.correct)} | {r.accuracy:.6f} |" for r in metrics.itertuples())
    prop_rows = []
    for batch in range(2, 11):
        values = proportions[proportions.batch == batch].sort_values("predicted_gas_label")
        prop_rows.append("| " + str(batch) + " | " + " | ".join(f"{v:.4f}" for v in values.proportion) + " |")
    c = summary["parameter_counts"]
    comparison_rows = []
    for r in comparison.itertuples():
        scope = f"Batch {int(r.batch)}" if pd.notna(r.batch) else r.scope.replace("_", " ").title()
        comparison_rows.append(f"| {scope} | {r.sgd_accuracy:.6f} | {r.adam_accuracy:.6f} | {r.sgd_minus_adam:+.6f} |")
    confusion_sections = []
    for batch in range(2, 11):
        matrix = confusions[confusions.batch == batch].pivot(
            index="true_gas_label", columns="predicted_gas_label", values="count").reindex(index=range(1, 7), columns=range(1, 7))
        rows = "\n".join("| " + str(true) + " | " + " | ".join(str(int(v)) for v in matrix.loc[true]) + " |" for true in range(1, 7))
        confusion_sections.append(f"### Batch {batch}\n\nRows are true labels and columns are predicted labels.\n\n"
                                  f"| True \\ Predicted | 1 | 2 | 3 | 4 | 5 | 6 |\n"
                                  f"|---:|---:|---:|---:|---:|---:|---:|\n{rows}")
    changed = cfg["adam_comparison"]["changed_parameters"]
    text = f"""# Canonical 1D ResNet B0 — SGD

This run uses SGD and compares a StepLR candidate with a no-scheduler control. Scheduler selection, early stopping, and epoch selection use Batch 1 only; target batches are evaluated only after the selected configuration is refit. Earlier run artifacts remain unchanged.

## Changed parameters

| Parameter | Adam B0 | This run |
|---|---:|---:|
| `training.optimizer` | {changed['training.optimizer']['adam_b0']} | {changed['training.optimizer']['this_run']} |
| `training.momentum` | {changed['training.momentum']['adam_b0']} | {changed['training.momentum']['this_run']} |

The initial learning rate is **{cfg['training']['learning_rate']}** and weight decay is **{cfg['training']['weight_decay']}**. Momentum **0.9** is an implementation assumption requested for this SGD variant; the Adam run has no corresponding momentum parameter.

## Scheduler selection

The candidates were the no-scheduler control and `StepLR(step_size=10, gamma=0.1)`. The paper does not specify `step_size`, so **10 is recorded as an implementation assumption**. With 30 maximum epochs, `scheduler.step()` is called after a completed epoch when another epoch remains, reducing the learning rate after epochs 10 and 20 (for use beginning in epochs 11 and 21). Candidate selection uses mean Batch 1 CV accuracy only; ties use lower mean selected validation cross-entropy and then declared candidate order.

| Candidate | Type | Mean CV accuracy | Mean selected validation CE | Median selected epoch | Selected |
|---|---|---:|---:|---:|---|
{scheduler_rows}

Selected scheduler: **{summary['selected_scheduler']['name']}**.

## Architecture

The verified path is `N×1×128 → N×128×128 → N×16,384 → FC128 → BatchNorm1d → FC6`. Five residual blocks use main-path Conv1d layers with kernel 3, stride 1, padding 1; shortcuts use kernel 1; channels are 1→32→64→128→256→128; there is no pooling.

Total trainable parameters: **{c['total']:,}** (backbone {c['residual_backbone']:,}, FC128/BatchNorm {c['FC128_including_batch_norm']:,}, FC6 {c['FC6']:,}).

## Batch 1 source cross-validation

Each StandardScaler was fit only on its training fold. Both scheduler candidates used identical folds and seeds. Early stopping used validation accuracy (validation loss breaks accuracy ties), maximum {cfg['training']['max_epochs']} epochs, and patience {cfg['training']['patience']}.

| Fold | Train | Validation | Epochs run | Selected epoch | Accuracy |
|---:|---:|---:|---:|---:|---:|
{cv_rows}

Mean source CV accuracy: **{summary['source_cv_mean_accuracy']:.6f}** (sample standard deviation {summary['source_cv_std_accuracy']:.6f}). The median fold-best epoch selected for final fitting was **{summary['source_selected_epochs']}**.

## Frozen target evaluation

After source-only selection, one scaler and model were fit on all Batch 1 for {summary['source_selected_epochs']} epochs. Batches 2–10 were then transformed and evaluated once.

| Batch | Samples | Correct | Accuracy |
|---:|---:|---:|---:|
{target_rows}

Unweighted mean batch accuracy: **{summary['target_unweighted_mean_accuracy']:.6f}**. Pooled target accuracy: **{summary['target_pooled_accuracy']:.6f}** across {summary['target_total_samples']:,} samples.

## Direct comparison with Adam B0

Adam reference: `{summary['adam_comparison_run']}`. Positive deltas favor SGD.

| Scope | SGD accuracy | Adam accuracy | SGD − Adam |
|---|---:|---:|---:|
{chr(10).join(comparison_rows)}

## Confusion matrices

{chr(10).join(confusion_sections)}

## Predicted-class proportions

| Batch | Class 1 | Class 2 | Class 3 | Class 4 | Class 5 | Class 6 |
|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(prop_rows)}

No samples were excluded, modified, augmented, deduplicated, clipped, or imputed. Scheduler choice and all other fitting and epoch selection used Batch 1 only; Batches 2–10 were not used to tune any setting. A1 was not run.
"""
    (out / "report.md").write_text(text, encoding="utf-8")
