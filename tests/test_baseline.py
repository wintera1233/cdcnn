"""Tests for the v7 baseline ladder.

No test opens a target batch. The audit machinery is exercised through
`TargetAccessLog` directly and through `load_target`'s refusal to run without
one, so running the suite can never contaminate a run.
"""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src import audit, normalize
from src.config import load as load_config, validate
from src.data import (BATCH_ROWS, GAS_LABELS, batch_path, class_counts,
                      load_source, load_target)
from src.loss import EpochLoss, cross_entropy, loss_module
from src.model import VARIANTS, build, parameter_breakdown, to_input
from src.protocol import ProtocolError, TargetAccessLog
from src.train import train_one

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "baseline_ladder.json"

# proposal.md section 6.
EXPECTED_PARAMETERS = {
    "R-txt": {"backbone": 336_416, "head": 2_098_310, "total": 2_434_726},
    "R-txt-ps": {"backbone": 336_416, "head": 2_098_310, "total": 2_434_726},
    "R-lite": {"backbone": 336_416, "head": 17_542, "total": 353_958},
    "R-lite-ps": {"backbone": 336_416, "head": 17_542, "total": 353_958},
}


def _smoke_config(**overrides):
    config = {"implementation_version": "test", "normalizer": "standard_scaler",
              "training": {"epochs": 2, "batch_size": 64},
              "optimizer": {"lr": 0.001, "momentum": 0.9, "weight_decay": 1e-4},
              "scheduler": {"step_size": 25, "gamma": 0.5}}
    config.update(overrides)
    return config


class DataTests(unittest.TestCase):
    def test_source_rows_and_class_histogram(self):
        x, y = load_source()
        self.assertEqual(x.shape, (445, 128))
        self.assertEqual(x.dtype, np.float64)
        counts = {GAS_LABELS[label]: count
                  for label, count in sorted(class_counts(y).items())}
        self.assertEqual(counts, {"Acetone": 90, "Acetaldehyde": 98, "Ethanol": 83,
                                  "Ethylene": 30, "Ammonia": 70, "Toluene": 74})

    def test_every_batch_file_is_present_and_unmodified_in_length(self):
        for index, rows in BATCH_ROWS.items():
            path = batch_path(index)
            self.assertTrue(path.exists(), path)
            with path.open("r", encoding="utf-8") as handle:
                self.assertEqual(sum(1 for line in handle if line.strip()), rows,
                                 f"batch{index}.dat row count changed")

    def test_target_batches_cannot_be_read_without_an_access_log(self):
        with self.assertRaisesRegex(ProtocolError, "TargetAccessLog"):
            load_target(2, None)
        with self.assertRaisesRegex(ProtocolError, "not a target batch"):
            load_target(1, TargetAccessLog())


class NormalizerTests(unittest.TestCase):
    def test_per_sample_fits_nothing_and_standardises_each_row(self):
        x = np.random.default_rng(0).normal(size=(7, 128)) * 5 + 3
        params = normalize.fit("per_sample", x)
        self.assertEqual(params, {"kind": "per_sample"})
        out = normalize.apply(params, x)
        np.testing.assert_allclose(out.mean(axis=1), 0, atol=1e-12)
        np.testing.assert_allclose(out.std(axis=1), 1, atol=1e-12)

    def test_standard_scaler_uses_only_the_array_it_was_fitted_on(self):
        source = np.random.default_rng(1).normal(size=(50, 128))
        params = normalize.fit("standard_scaler", source)
        np.testing.assert_allclose(
            normalize.apply(params, source).mean(axis=0), 0, atol=1e-12)
        shifted = source + 10.0
        # A drifted array is not re-centred: the source statistics still apply.
        self.assertGreater(abs(normalize.apply(params, shifted).mean()), 1.0)


class ModelTests(unittest.TestCase):
    def test_parameter_counts_match_the_proposal(self):
        for variant, expected in EXPECTED_PARAMETERS.items():
            self.assertEqual(parameter_breakdown(build(variant)), expected, variant)

    def test_backbone_keeps_length_128_and_contains_no_pooling(self):
        for variant in VARIANTS:
            model = build(variant)
            features = model.features(to_input(torch.zeros(4, 128)))
            self.assertEqual(tuple(features.shape), (4, 128, 128), variant)
            for module in model.blocks.modules():
                self.assertNotIsInstance(
                    module, (torch.nn.MaxPool1d, torch.nn.AvgPool1d,
                             torch.nn.AdaptiveAvgPool1d, torch.nn.AdaptiveMaxPool1d),
                    f"{variant}: the backbone must not pool")

    def test_the_variants_form_a_complete_two_by_two(self):
        from src.model import VARIANTS as spec
        grid = {(entry["head"], entry["normalizer"]) for entry in spec.values()}
        self.assertEqual(grid, {("flatten", "standard_scaler"),
                                ("flatten", "per_sample"),
                                ("gap", "standard_scaler"),
                                ("gap", "per_sample")})
        channels = {entry["channels"] for entry in spec.values()}
        self.assertEqual(len(channels), 1, "all four share one backbone")

    def test_heads_differ_only_in_the_reduction(self):
        self.assertEqual(parameter_breakdown(build("R-txt"))["backbone"],
                         parameter_breakdown(build("R-lite"))["backbone"])
        gap = build("R-lite").head
        self.assertIsInstance(gap.reduce[0], torch.nn.AdaptiveAvgPool1d)
        self.assertIsInstance(build("R-txt").head.reduce, torch.nn.Flatten)

    def test_input_is_one_channel_of_length_128(self):
        self.assertEqual(tuple(to_input(torch.zeros(5, 128)).shape), (5, 1, 128))
        with self.assertRaisesRegex(ProtocolError, r"\[N, 128\]"):
            to_input(torch.zeros(5, 16, 8))

    def test_unknown_variant_is_refused(self):
        with self.assertRaisesRegex(ProtocolError, "unknown variant"):
            build("R-fig")


class LossTests(unittest.TestCase):
    def test_matches_cross_entropy_loss(self):
        generator = torch.Generator().manual_seed(0)
        logits = torch.randn(64, 6, generator=generator)
        targets = torch.randint(0, 6, (64,), generator=generator)
        torch.testing.assert_close(cross_entropy(logits, targets),
                                   loss_module()(logits, targets))

    def test_s2_equals_a_global_mean_only_when_the_batch_size_divides(self):
        generator = torch.Generator().manual_seed(1)

        def accumulate(sizes):
            accumulator = EpochLoss()
            for size in sizes:
                logits = torch.randn(size, 6, generator=generator)
                targets = torch.randint(0, 6, (size,), generator=generator)
                accumulator.update(cross_entropy(logits, targets), logits, targets)
            return accumulator

        # 448 rows at batch 64: seven equal batches, so the two agree.
        equal = accumulate([64] * 7)
        self.assertAlmostEqual(equal.s2, equal.sample_weighted, places=12)

        # 445 rows at batch 64: six of 64 and one of 61, so they do not.
        ragged = accumulate([64] * 6 + [61])
        self.assertEqual(int(ragged.summary()["samples"]), 445)
        self.assertNotAlmostEqual(ragged.s2, ragged.sample_weighted, places=6)


class AuditTests(unittest.TestCase):
    def test_freezing_after_a_target_access_is_refused(self):
        log = TargetAccessLog()
        log.record_access(batch_path(2))
        with self.assertRaisesRegex(ProtocolError, "after target files were opened"):
            log.record_freeze(Path("checkpoints/final.pt"), "deadbeef")

    def test_leakage_audit_passes_when_freezes_precede_accesses(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            log = TargetAccessLog()
            log.record_freeze(Path("a.pt"), "aa")
            log.record_access(batch_path(2))
            report = audit.leakage_audit(run_dir, log)
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["checkpoints_frozen"], 1)
            self.assertTrue((run_dir / "leakage_target_access_audit.json").exists())

    def test_leakage_audit_fails_when_nothing_was_frozen(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            log = TargetAccessLog()
            log.record_access(batch_path(2))
            with self.assertRaisesRegex(ProtocolError, "no checkpoint was frozen"):
                audit.leakage_audit(run_dir, log)
            report = json.loads(
                (run_dir / "leakage_target_access_audit.json").read_text())
            self.assertEqual(report["status"], "failed")

    def test_manifest_records_code_data_and_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = audit.write_manifest(Path(directory), {"a": 1})
            self.assertIn("src/model.py", manifest["code_sha256"])
            self.assertIn("batch1.dat", manifest["dataset_sha256"])
            self.assertIn("torch", manifest["environment"])
            self.assertEqual(len(manifest["config_sha256"]), 64)


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(CONFIG)

    def test_the_ladder_config_is_valid(self):
        self.assertEqual(self.config["variants"],
                         ["R-txt", "R-txt-ps", "R-lite", "R-lite-ps"])
        self.assertEqual(self.config["seeds"], [1042, 2024, 3407])
        self.assertEqual(self.config["training"], {"epochs": 100, "batch_size": 64})

    def test_undeclared_variant_is_rejected(self):
        changed = copy.deepcopy(self.config)
        changed["variants"].append("R-fig")
        with self.assertRaisesRegex(ProtocolError, "undeclared variant"):
            validate(changed)

    def test_the_pinned_version_refuses_altered_protocol_constants(self):
        for section, key, value in (("training", "epochs", 50),
                                    ("optimizer", "lr", 0.01),
                                    ("scheduler", "step_size", 10)):
            changed = copy.deepcopy(self.config)
            changed[section][key] = value
            with self.assertRaisesRegex(ProtocolError, f"fixes {section}"):
                validate(changed)

    def test_duplicate_seeds_are_rejected(self):
        changed = copy.deepcopy(self.config)
        changed["seeds"] = [1042, 1042, 2024]
        with self.assertRaisesRegex(ProtocolError, "listed twice"):
            validate(changed)


class TrainingTests(unittest.TestCase):
    def test_the_same_seed_reproduces_identical_losses(self):
        x, y = load_source()
        config = _smoke_config()
        histories = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as directory:
                summary = train_one("R-lite", 1042, config, x, y,
                                    Path(directory) / "ck", "cpu",
                                    save_every_epoch=False)
                histories.append(summary["history"])
        self.assertEqual(histories[0], histories[1])

    def test_a_different_seed_gives_a_different_trajectory(self):
        x, y = load_source()
        config = _smoke_config()
        results = []
        for seed in (1042, 2024):
            with tempfile.TemporaryDirectory() as directory:
                results.append(train_one("R-lite", seed, config, x, y,
                                         Path(directory) / "ck", "cpu",
                                         save_every_epoch=False)["history"])
        self.assertNotEqual(results[0], results[1])

    def test_training_refuses_to_write_into_an_existing_directory(self):
        x, y = load_source()
        with tempfile.TemporaryDirectory() as directory:
            existing = Path(directory) / "ck"
            existing.mkdir()
            with self.assertRaises(FileExistsError):
                train_one("R-lite", 1042, _smoke_config(), x, y, existing, "cpu",
                          save_every_epoch=False)

    def test_batches_per_epoch_follows_445_rows(self):
        x, y = load_source()
        with tempfile.TemporaryDirectory() as directory:
            summary = train_one("R-lite", 1042, _smoke_config(), x, y,
                                Path(directory) / "ck", "cpu", save_every_epoch=False)
        self.assertEqual(summary["batches_per_epoch"], 7)
        self.assertEqual(summary["source_rows"], 445)

    def test_checkpoints_carry_the_normalizer_and_are_written_per_epoch(self):
        x, y = load_source()
        with tempfile.TemporaryDirectory() as directory:
            checkpoints = Path(directory) / "ck"
            train_one("R-lite-ps", 1042, _smoke_config(normalizer="per_sample"),
                      x, y, checkpoints, "cpu", save_every_epoch=True)
            names = sorted(path.name for path in checkpoints.glob("*.pt"))
            self.assertEqual(names, ["epoch_001.pt", "epoch_002.pt", "final.pt"])
            payload = torch.load(checkpoints / "final.pt", weights_only=False)
            self.assertEqual(payload["normalizer"], {"kind": "per_sample"})
            self.assertEqual(payload["variant"], "R-lite-ps")


if __name__ == "__main__":
    unittest.main()
