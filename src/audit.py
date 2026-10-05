"""Run manifests, checkpoint freezing, and the leakage audit.

`CLAUDE.md` section 2: Batches 2-10 stay closed until every checkpoint for the
comparison is written to disk and hashed, and every run records when each target
file was first opened and when each checkpoint was frozen.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

from src.data import ROOT, dataset_hashes
from src.protocol import ProtocolError, TargetAccessLog, utc_now

CODE_FILES = ("src/protocol.py", "src/data.py", "src/normalize.py", "src/model.py",
              "src/augment.py", "src/generate.py", "src/project.py",
              "src/loss.py", "src/train.py", "src/evaluate.py", "src/audit.py",
              "src/config.py",
              "scripts/run_baseline.py")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def code_hashes() -> dict[str, str]:
    return {name: sha256_path(ROOT / name) for name in CODE_FILES
            if (ROOT / name).exists()}


def _git_revision() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, timeout=15,
                              check=True).stdout.strip()
    except Exception:
        return "unavailable"


def _driver_version() -> str:
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15, check=True).stdout.strip()
    except Exception:
        return "unavailable"


def environment() -> dict:
    import numpy
    import sklearn
    import torch
    return {"python": platform.python_version(), "executable": sys.executable,
            "platform": platform.platform(), "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "device_name": (torch.cuda.get_device_name(0)
                            if torch.cuda.is_available() else "cpu"),
            "driver": _driver_version(), "numpy": numpy.__version__,
            "scikit_learn": sklearn.__version__, "git_revision": _git_revision()}


def write_manifest(run_dir: Path, config: dict) -> dict:
    manifest = {"created_at": utc_now(), "run_dir": str(run_dir),
                "config": config,
                "config_sha256": sha256_bytes(
                    json.dumps(config, sort_keys=True).encode("utf-8")),
                "code_sha256": code_hashes(), "dataset_sha256": dataset_hashes(),
                "environment": environment()}
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def freeze(checkpoints: list[Path], access_log: TargetAccessLog) -> dict[str, str]:
    """Hash every checkpoint and record the freeze.

    `TargetAccessLog.record_freeze` raises if any target file has already been
    opened, so a run that reads a target early cannot complete this step.
    """
    if not checkpoints:
        raise ProtocolError("nothing to freeze")
    digests: dict[str, str] = {}
    for path in sorted(checkpoints):
        if not path.exists():
            raise ProtocolError(f"checkpoint missing at freeze time: {path}")
        digest = sha256_path(path)
        access_log.record_freeze(path, digest)
        digests[str(path)] = digest
    return digests


def leakage_audit(run_dir: Path, access_log: TargetAccessLog) -> dict:
    """Fail if any target access precedes the last checkpoint freeze."""
    freezes = access_log.freezes
    accesses = access_log.accesses
    last_freeze = max((event["frozen_at"] for event in freezes), default=None)
    first_access = min(accesses.values(), default=None)
    violations = []
    # A run that never opened a target file cannot have leaked, so it needs no
    # freeze: source-only runs such as the cross-validation sweep pass on that
    # ground alone. A run that did open one must have frozen first.
    if accesses and not freezes:
        violations.append("a target file was opened but no checkpoint was frozen")
    if first_access is not None and last_freeze is not None and first_access < last_freeze:
        violations.append(
            f"a target file was opened at {first_access}, before the last "
            f"checkpoint freeze at {last_freeze}")
    report = {"status": "passed" if not violations else "failed",
              "checked_at": utc_now(), "violations": violations,
              "checkpoints_frozen": len(freezes), "last_freeze": last_freeze,
              "target_files_opened": len(accesses), "first_target_access": first_access,
              "target_first_access": accesses, "checkpoint_freezes": freezes}
    (run_dir / "leakage_target_access_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if violations:
        raise ProtocolError("; ".join(violations))
    return report


def new_run_dir(name: str) -> Path:
    """A unique `runs/<timestamp>_<name>/`. Never reuses an existing directory."""
    stamp = utc_now().replace("-", "").replace(":", "").replace(".", "")
    run_dir = ROOT / "runs" / f"{stamp}_{name}"
    if run_dir.exists():
        raise ProtocolError(f"run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir
