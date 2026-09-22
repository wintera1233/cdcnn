#!/usr/bin/env python
"""The single entry point for the v7 baseline ladder.

    python scripts/run_baseline.py smoke      --config configs/baseline_ladder.json
    python scripts/run_baseline.py gpu-smoke  --config configs/baseline_ladder.json
    python scripts/run_baseline.py launch     --config configs/baseline_ladder.json \
        --max-workers 1 --gpu-smoke-run runs/<passing_gpu_smoke>
    python scripts/run_baseline.py evaluate   --run runs/<launched_with_no_evaluate>

`launch` trains every variant at every seed on Batch 1, freezes and hashes all
checkpoints, and only then opens Batches 2-10 — once. `--no-evaluate` stops after
the freeze, leaving the target pass to `evaluate`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src import audit, evaluate as evaluation  # noqa: E402
from src.config import load as load_config  # noqa: E402
from src.data import load_source  # noqa: E402
from src.protocol import ProtocolError, TargetAccessLog, utc_now  # noqa: E402
from src.train import train_one  # noqa: E402


def _device(require_cuda: bool) -> str:
    if torch.cuda.is_available():
        return "cuda"
    if require_cuda:
        raise ProtocolError("CUDA is not available; run inside the pinned container")
    return "cpu"


def _write(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def _train_all(run_dir: Path, config: dict, variants, seeds, device: str,
               epochs: int | None, per_epoch_seed) -> list[Path]:
    source_x, source_y = load_source()
    training = dict(config["training"])
    if epochs is not None:
        training["epochs"] = epochs
    effective = {**config, "training": training}

    checkpoints: list[Path] = []
    summaries = []
    total = len(variants) * len(seeds)
    index = 0
    for variant in variants:
        for seed in seeds:
            index += 1
            target = run_dir / "checkpoints" / f"{variant}_seed{seed}"
            print(f"[{index}/{total}] {variant} seed {seed} -> {target.name}", flush=True)
            summary = train_one(
                variant, seed, effective, source_x, source_y, target, device,
                save_every_epoch=(seed == per_epoch_seed))
            summaries.append(summary)
            checkpoints.extend(sorted(target.glob("*.pt")))
            print(f"    train acc {summary['final_train_accuracy']:.4f}  "
                  f"loss {summary['final_train_loss_s2']:.4f}  "
                  f"params {summary['parameters']['total']:,}", flush=True)
    _write(run_dir / "training_summaries.json", summaries)
    return checkpoints


def command_smoke(args) -> int:
    """Batch 1 only, a few epochs, CPU or GPU. Never opens a target batch."""
    config = load_config(Path(args.config))
    device = _device(require_cuda=args.require_cuda)
    run_dir = audit.new_run_dir(args.name)
    audit.write_manifest(run_dir, config)
    variants = args.variants or list(config["variants"])
    checkpoints = _train_all(run_dir, config, variants, [config["seeds"][0]], device,
                             args.epochs, per_epoch_seed=None)
    access_log = TargetAccessLog()
    digests = audit.freeze(checkpoints, access_log)
    report = audit.leakage_audit(run_dir, access_log)
    _write(run_dir / "smoke.json",
           {"status": "passed", "device": device, "epochs": args.epochs,
            "variants": variants, "checkpoints": len(digests),
            "leakage_audit": report["status"], "finished_at": utc_now()})
    print(f"\nsmoke passed on {device}: {len(digests)} checkpoints, "
          f"leakage audit {report['status']}\n{run_dir}")
    return 0


def command_gpu_smoke(args) -> int:
    args.require_cuda = True
    args.name = "baseline_gpu_smoke"
    return command_smoke(args)


def _gate(gpu_smoke_run: str | None) -> Path:
    """`CLAUDE.md` section 6: no full run without a passing GPU smoke artifact."""
    if not gpu_smoke_run:
        raise ProtocolError(
            "--gpu-smoke-run is required: a full run needs a passing GPU smoke "
            "artifact from the current code")
    path = Path(gpu_smoke_run)
    artifact = path / "smoke.json"
    if not artifact.exists():
        raise ProtocolError(f"no smoke artifact at {artifact}")
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    if payload.get("status") != "passed" or payload.get("device") != "cuda":
        raise ProtocolError(f"{artifact} is not a passing GPU smoke: {payload}")
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    current = audit.code_hashes()
    if manifest["code_sha256"] != current:
        changed = sorted(name for name in current
                         if manifest["code_sha256"].get(name) != current[name])
        raise ProtocolError(
            f"code changed since the GPU smoke: {changed}. Re-run gpu-smoke.")
    return path


def command_launch(args) -> int:
    if args.max_workers != 1:
        raise ProtocolError("CUDA runs are single-worker; pass --max-workers 1")
    config = load_config(Path(args.config))
    gpu_smoke = _gate(args.gpu_smoke_run)
    device = _device(require_cuda=True)

    run_dir = audit.new_run_dir(args.name)
    audit.write_manifest(run_dir, config)
    _write(run_dir / "gate.json", {"gpu_smoke_run": str(gpu_smoke),
                                   "checked_at": utc_now()})

    checkpoints = _train_all(run_dir, config, list(config["variants"]),
                             list(config["seeds"]), device, args.epochs,
                             config.get("per_epoch_checkpoint_seed"))

    access_log = TargetAccessLog()
    digests = audit.freeze(checkpoints, access_log)
    _write(run_dir / "checkpoint_digests.json", digests)
    print(f"\nfrozen: {len(digests)} checkpoints", flush=True)

    if args.no_evaluate:
        audit.leakage_audit(run_dir, access_log)
        print(f"stopped before the target pass\n{run_dir}")
        return 0
    return _evaluate(run_dir, access_log, device)


def _evaluate(run_dir: Path, access_log: TargetAccessLog, device: str) -> int:
    finals = sorted((run_dir / "checkpoints").glob("*/final.pt"))
    print(f"opening Batches 2-10 for {len(finals)} final checkpoints", flush=True)
    results = [evaluation.evaluate_checkpoint(path, access_log, device)
               for path in finals]
    summary = evaluation.aggregate(results)
    _write(run_dir / "target_results.json", results)
    _write(run_dir / "target_summary.json", summary)
    report = audit.leakage_audit(run_dir, access_log)

    print(f"\nleakage audit: {report['status']}\n")
    print(f"{'variant':12s} {'source':>7s} {'target mean':>12s} {'sd':>8s}")
    for variant, entry in summary.items():
        print(f"{variant:12s} {entry['source_accuracy']:7.4f} "
              f"{entry['target_mean']:12.4f} {entry['target_mean_sd']:8.4f}")
    print(f"\npaper ResNet target mean: 0.6344\n{run_dir}")
    return 0


def command_evaluate(args) -> int:
    run_dir = Path(args.run)
    digests_path = run_dir / "checkpoint_digests.json"
    if not digests_path.exists():
        raise ProtocolError(f"{run_dir} has no frozen checkpoints")
    recorded = json.loads(digests_path.read_text(encoding="utf-8"))
    access_log = TargetAccessLog()
    for path_text, digest in sorted(recorded.items()):
        path = Path(path_text)
        current = audit.sha256_path(path)
        if current != digest:
            raise ProtocolError(f"{path} changed since it was frozen")
        access_log.record_freeze(path, current)
    return _evaluate(run_dir, access_log, _device(require_cuda=True))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    smoke = sub.add_parser("smoke", help="Batch 1 only, a few epochs")
    smoke.add_argument("--config", required=True)
    smoke.add_argument("--epochs", type=int, default=3)
    smoke.add_argument("--variants", nargs="*")
    smoke.add_argument("--name", default="baseline_smoke")
    smoke.add_argument("--require-cuda", action="store_true")
    smoke.set_defaults(handler=command_smoke)

    gpu = sub.add_parser("gpu-smoke", help="the pre-flight gate artifact")
    gpu.add_argument("--config", required=True)
    gpu.add_argument("--epochs", type=int, default=3)
    gpu.add_argument("--variants", nargs="*")
    gpu.set_defaults(handler=command_gpu_smoke)

    launch = sub.add_parser("launch", help="the full ladder")
    launch.add_argument("--config", required=True)
    launch.add_argument("--gpu-smoke-run")
    launch.add_argument("--max-workers", type=int, default=1)
    launch.add_argument("--epochs", type=int)
    launch.add_argument("--name", default="baseline_ladder_full")
    launch.add_argument("--no-evaluate", action="store_true")
    launch.set_defaults(handler=command_launch)

    evaluate_parser = sub.add_parser("evaluate", help="the target pass for a frozen run")
    evaluate_parser.add_argument("--run", required=True)
    evaluate_parser.set_defaults(handler=command_evaluate)

    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except ProtocolError as error:
        print(f"ProtocolError: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
