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
from src.loss import EpochLoss, cross_entropy, loss_module, mse_consistency
from src.model import (GENERATION_SPLIT, LAMBDA_MSE, VARIANTS, build,
                       parameter_breakdown, to_input)
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
    "R-fig": {"backbone": 1_058_080, "head": 2_098_310, "total": 3_156_390},
    "R-fig-ps": {"backbone": 1_058_080, "head": 2_098_310, "total": 3_156_390},
}


def _smoke_config(**overrides):
    config = {"implementation_version": "test",
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
        self.assertEqual(counts, {"Ethanol": 90, "Ammonia": 98, "Ethylene": 83,
                                  "Acetaldehyde": 30, "Acetone": 70,
                                  "Toluene": 74})

    def test_every_batch_file_is_present_and_unmodified_in_length(self):
        for index, rows in BATCH_ROWS.items():
            path = batch_path(index)
            self.assertTrue(path.exists(), path)
            with path.open("r", encoding="utf-8") as handle:
                self.assertEqual(sum(1 for line in handle if line.strip()), rows,
                                 f"batch{index}.dat row count changed")

    def test_the_per_label_counts_are_unchanged(self):
        """The counts are the immutable fact; the gas names are an interpretation.

        `docs/label-mapping.md` records why the paper's Table 2 cannot be used to
        derive the mapping: its per-gas counts match this file only under a
        column order that contradicts the dataset's own stated encoding.
        """
        expected = {
            1: [90, 98, 83, 30, 70, 74], 2: [164, 334, 100, 109, 532, 5],
            3: [365, 490, 216, 240, 275, 0], 4: [64, 43, 12, 30, 12, 0],
            5: [28, 40, 20, 46, 63, 0], 6: [514, 574, 110, 29, 606, 467],
            7: [649, 662, 360, 744, 630, 568], 8: [30, 30, 40, 33, 143, 18],
            9: [61, 55, 100, 75, 78, 101], 10: [600] * 6}
        for index, counts in expected.items():
            seen = {label: 0 for label in range(1, 7)}
            with batch_path(index).open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        seen[int(line.split(None, 1)[0])] += 1
            self.assertEqual([seen[label] for label in range(1, 7)], counts,
                             f"batch {index} per-label counts changed")

    def test_the_adopted_label_mapping(self):
        self.assertEqual(GAS_LABELS, {1: "Ethanol", 2: "Ammonia", 3: "Ethylene",
                                      4: "Acetaldehyde", 5: "Acetone",
                                      6: "Toluene"})
        # The class the models cannot transfer is the smallest in Batch 1.
        _, y = load_source()
        self.assertEqual(int((y == 4).sum()), 30)
        self.assertEqual(GAS_LABELS[4], "Acetaldehyde")

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

    def test_the_v7_0_ladder_is_a_complete_two_by_two_on_one_backbone(self):
        from src.model import TEXT_CAPPED_128, VARIANTS as spec
        cells = {(entry["head"], entry["normalizer"]) for name, entry in spec.items()
                 if entry["channels"] == TEXT_CAPPED_128}
        self.assertEqual(cells, {("flatten", "standard_scaler"),
                                 ("flatten", "per_sample"),
                                 ("gap", "standard_scaler"),
                                 ("gap", "per_sample")})

    def test_the_figure_widths_differ_only_where_the_sources_conflict(self):
        from src.model import FIGURE_WIDTHS, TEXT_CAPPED_128
        self.assertEqual(FIGURE_WIDTHS[:3], TEXT_CAPPED_128[:3])
        self.assertEqual(FIGURE_WIDTHS[3], (128, 256, 256))
        self.assertEqual(FIGURE_WIDTHS[4], (256, 512, 128))
        # Resnet1's main path is never narrower than its shortcut: `3 Conv 2` is
        # read as 32; see proposal.md section 2.2.
        self.assertEqual(FIGURE_WIDTHS[0], (1, 32, 32))

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
            build("R-nonexistent")


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

    def test_retired_variants_are_refused_in_a_new_configuration(self):
        changed = copy.deepcopy(self.config)
        changed["implementation_version"] = "CDCNN_v7.1_something"
        with self.assertRaisesRegex(ProtocolError, "retired variant"):
            validate(changed)

    def test_the_completed_ladder_still_validates(self):
        # The v7.0 config is the record of a finished run; it keeps validating so
        # its checkpoints stay loadable.
        validate(self.config)

    def test_undeclared_variant_is_rejected(self):
        changed = copy.deepcopy(self.config)
        changed["variants"].append("R-nonexistent")
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


class CrossValidationTests(unittest.TestCase):
    def test_folds_partition_the_source_and_keep_every_class_even(self):
        from src.cv import N_FOLDS, stratified_folds
        _, y = load_source()
        folds = stratified_folds(y)
        self.assertEqual(len(folds), N_FOLDS)
        self.assertEqual(sorted(np.concatenate(folds).tolist()),
                         list(range(len(y))))
        for label in range(1, 7):
            sizes = [int((y[f] == label).sum()) for f in folds]
            self.assertLessEqual(max(sizes) - min(sizes), 1,
                                 f"class {label} is unevenly split: {sizes}")
        # Ethylene's 30 rows must not concentrate in one fold.
        self.assertEqual([int((y[f] == 4).sum()) for f in folds], [6] * N_FOLDS)

    def test_folds_are_fixed_independently_of_the_model_seed(self):
        from src.cv import stratified_folds
        _, y = load_source()
        first = [f.tolist() for f in stratified_folds(y)]
        second = [f.tolist() for f in stratified_folds(y)]
        self.assertEqual(first, second)
        different = [f.tolist() for f in stratified_folds(y, seed=1)]
        self.assertNotEqual(first, different)

    def test_a_fold_curve_has_one_entry_per_epoch_and_never_sees_the_held_out_part(self):
        from src.cv import fold_curve, stratified_folds
        x, y = load_source()
        folds = stratified_folds(y)
        held_out = folds[0]
        train_index = np.concatenate(folds[1:])
        curve = fold_curve("R-fig-ps", 1042, 0.001, _smoke_config(), x, y,
                           train_index, held_out, "cpu")
        self.assertEqual(sorted(curve), ["accuracy", "loss"])
        self.assertEqual(len(curve["accuracy"]), 2)
        self.assertEqual(len(curve["loss"]), 2)
        self.assertTrue(all(0.0 <= v <= 1.0 for v in curve["accuracy"]))
        self.assertTrue(all(v > 0.0 for v in curve["loss"]))
        self.assertEqual(len(set(train_index) & set(held_out.tolist())), 0)

    def test_summarise_reports_both_the_accuracy_peak_and_the_loss_minimum(self):
        from src.cv import summarise
        entry = summarise([{"accuracy": [0.1, 0.5, 0.4], "loss": [2.0, 1.0, 1.5]},
                           {"accuracy": [0.1, 0.5, 0.4], "loss": [2.0, 1.0, 1.5]}])
        self.assertEqual(entry["peak_epoch"], 2)
        self.assertEqual(entry["min_loss_epoch"], 2)
        self.assertAlmostEqual(entry["min_loss"], 1.0)
        self.assertAlmostEqual(entry["final_loss"], 1.5)

    def test_a_run_that_opens_no_target_needs_no_freeze(self):
        with tempfile.TemporaryDirectory() as directory:
            report = audit.leakage_audit(Path(directory), TargetAccessLog())
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["target_files_opened"], 0)

    def test_a_run_that_opens_a_target_without_freezing_still_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            log = TargetAccessLog()
            log.record_access(batch_path(2))
            with self.assertRaisesRegex(ProtocolError, "no checkpoint was frozen"):
                audit.leakage_audit(Path(directory), log)


class FeatureGenerationTests(unittest.TestCase):
    def setUp(self):
        from src.generate import FeatureGeneration, STYLE_AXES
        self.FeatureGeneration = FeatureGeneration
        self.styles = STYLE_AXES
        self.z = torch.randn(64, 128, 128,
                             generator=torch.Generator().manual_seed(0))

    def test_the_decomposition_is_exact(self):
        """Eqs. (8)-(9): the pooled part and the residual must sum to z."""
        block = self.FeatureGeneration()
        pooled, residual = block.decompose(self.z)
        torch.testing.assert_close(pooled + residual, self.z)

    def test_the_pooled_part_is_piecewise_constant(self):
        """MaxPool then nearest upsample repeats each pooled value, so the
        pooled part carries half as many distinct values along the length."""
        block = self.FeatureGeneration(pool=2)
        pooled, _ = block.decompose(self.z)
        torch.testing.assert_close(pooled[..., 0::2], pooled[..., 1::2])

    def test_each_style_axis_reduces_the_declared_dimensions(self):
        from src.generate import reduce_dims
        expected = {"scalar": (64, 1, 1), "channel": (64, 128, 1),
                    "position": (64, 1, 128)}
        for style in self.styles:
            shape = tuple(self.z.mean(dim=reduce_dims(style), keepdim=True).shape)
            self.assertEqual(shape, expected[style], style)

    def test_the_restyle_changes_the_features_and_is_reproducible(self):
        for style in self.styles:
            block = self.FeatureGeneration(style=style)
            first = block(self.z, torch.Generator().manual_seed(1))
            again = block(self.z, torch.Generator().manual_seed(1))
            torch.testing.assert_close(first, again)
            self.assertGreater(float((first - self.z).norm()), 0.0, style)

    def test_a_batch_of_one_is_returned_unchanged(self):
        """Eqs. (12)-(13) estimate the style distribution from the batch, so a
        single sample has no distribution to draw from."""
        block = self.FeatureGeneration()
        single = self.z[:1]
        torch.testing.assert_close(block(single, torch.Generator().manual_seed(1)),
                                   single)

    def test_an_unknown_style_axis_is_refused(self):
        with self.assertRaisesRegex(ProtocolError, "unknown style axis"):
            self.FeatureGeneration(style="sensor")

    def _two_sessions(self):
        g = torch.Generator().manual_seed(7)
        first = torch.randn(40, 128, 128, generator=g)
        # The second session is the first with a fixed per-position style shift,
        # so the block offset in style space is known and non-zero.
        shift = torch.linspace(-1.0, 1.0, 128).view(1, 1, 128)
        second = torch.randn(8, 128, 128, generator=g) + shift
        return first, second

    def test_a_signed_block_refuses_to_run_without_a_direction(self):
        block = self.FeatureGeneration(sign="fold")
        with self.assertRaisesRegex(ProtocolError, "needs a direction"):
            block(self.z, torch.Generator().manual_seed(1))

    def test_an_unknown_sign_mode_is_refused(self):
        with self.assertRaisesRegex(ProtocolError, "unknown sign mode"):
            self.FeatureGeneration(sign="up")

    def test_fold_moves_the_style_mean_along_the_offset_sign(self):
        """The paper's draw is symmetric; the folded one must lean the way the
        block offset points, coordinate by coordinate."""
        first, second = self._two_sessions()
        block = self.FeatureGeneration(sign="fold")
        block.set_direction(first, second)
        out = block(self.z, torch.Generator().manual_seed(1))
        mean_before, _ = block.style_moments(self.z)
        mean_after, _ = block.style_moments(out)
        moved = (mean_after - mean_before).mean(dim=0).flatten()
        sign = torch.sign(block._direction_mean).flatten()
        agree = float((torch.sign(moved) == sign).float().mean())
        self.assertGreater(agree, 0.9)

    def test_shift_displaces_the_style_mean_by_the_declared_offsets(self):
        first, second = self._two_sessions()
        block = self.FeatureGeneration(sign="shift", displacement=2.0)
        block.set_direction(first, second)
        out = block(self.z, torch.Generator().manual_seed(1))
        mean_before, _ = block.style_moments(self.z)
        mean_after, _ = block.style_moments(out)
        moved = (mean_after - mean_before).mean(dim=0, keepdim=True)
        # The batch-level restyle recentres every sample on the drawn mean, so
        # the batch's average move equals the directed part, 2 x the offset,
        # up to the paper's noise term.
        torch.testing.assert_close(moved, 2.0 * block._direction_mean,
                                   atol=0.05, rtol=0.2)

    def test_the_unsigned_block_is_unchanged_by_a_direction(self):
        first, second = self._two_sessions()
        plain = self.FeatureGeneration()
        told = self.FeatureGeneration()
        told.set_direction(first, second)
        torch.testing.assert_close(plain(self.z, torch.Generator().manual_seed(1)),
                                   told(self.z, torch.Generator().manual_seed(1)))

    def test_supervised_contrastive_matches_the_formula_on_a_small_case(self):
        """Eq. (S5) by hand on four anchors, two per label, against the code."""
        from src.loss import supervised_contrastive
        z = torch.tensor([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0], [-0.6, 0.8]])
        y = torch.tensor([0, 0, 1, 1])
        tau = 0.5
        unit = torch.nn.functional.normalize(z, dim=1)
        sim = unit @ unit.T / tau
        expected = []
        for i in range(4):
            others = [a for a in range(4) if a != i]
            denominator = torch.logsumexp(sim[i, others], dim=0)
            positives = [p for p in others if y[p] == y[i]]
            expected.append(-sum(sim[i, p] - denominator for p in positives) / len(positives))
        torch.testing.assert_close(supervised_contrastive(z, y, tau),
                                   torch.stack(expected).mean())

    def test_supervised_contrastive_rises_when_a_collapsed_class_is_perturbed(self):
        """SupCon is minimised when every same-label sample sits on one point.
        Starting there, a perturbed generated branch must score higher than an
        identical one. (An identical twin alone does not minimise the loss for
        arbitrary features: the twin's exp(1/tau) also inflates the denominator
        of every other positive, so the test starts from collapsed classes.)"""
        from src.loss import supervised_contrastive
        g = torch.Generator().manual_seed(3)
        prototypes = torch.eye(6, 16)
        y = torch.arange(32) % 6
        z = prototypes[y]
        same = supervised_contrastive(torch.cat([z, z]), torch.cat([y, y]), 0.07)
        moved = supervised_contrastive(
            torch.cat([z, z + 0.3 * torch.randn(32, 16, generator=g)]),
            torch.cat([y, y]), 0.07)
        self.assertLess(float(same), float(moved))

    def test_supervised_contrastive_refuses_an_anchor_without_a_positive(self):
        from src.loss import supervised_contrastive
        z = torch.randn(3, 4)
        with self.assertRaisesRegex(ProtocolError, "at least one positive"):
            supervised_contrastive(z, torch.tensor([0, 1, 2]), 0.07)

    def test_the_v11_variants_carry_a_block_and_a_positive_lambda_con(self):
        from src.model import VARIANTS
        for name in ("R-con", "R-con-shift", "R-con-t5"):
            spec = VARIANTS[name]
            self.assertGreater(spec["lambda_con"], 0.0)
            self.assertIn("generate", spec)
            self.assertEqual(spec["augment"], VARIANTS["R-gen"]["augment"])
        self.assertEqual(VARIANTS["R-con-shift"]["generate"]["sign"], "shift")
        self.assertEqual(VARIANTS["R-con-t5"]["temperature"], 0.5)
        self.assertNotIn("lambda_con", VARIANTS["R-gen"])

    def test_the_v10_variants_declare_a_sign_and_keep_r_gen_otherwise(self):
        from src.model import VARIANTS
        for name, sign in (("R-gen-sign", "fold"), ("R-gen-shift", "shift")):
            spec = VARIANTS[name]
            self.assertEqual(spec["generate"]["sign"], sign)
            self.assertEqual(spec["augment"], VARIANTS["R-gen"]["augment"])
            self.assertEqual(spec["lambda_mse"], VARIANTS["R-gen"]["lambda_mse"])

    def test_evaluation_never_generates_features(self):
        """Fig. 2: 'during forecasting, the program separates the data
        manipulation block from the prediction process'."""
        import inspect
        from src import evaluate
        self.assertNotIn("FeatureGeneration", inspect.getsource(evaluate))


class GenerationWiringTests(unittest.TestCase):
    """v9.0: the block inside the network, and L_MSE on the pair."""

    GENERATION_CELLS = ("R-gen", "R-gen-ce2", "R-gen-m10")

    def setUp(self):
        self.x = to_input(np.random.default_rng(0).normal(size=(8, 128)))

    def test_the_block_sits_after_the_first_three_resnet_blocks(self):
        """Section 2.4: 'Rs(.) means the first three convolutional layers'."""
        self.assertEqual(GENERATION_SPLIT, 3)
        model = build("R-gen")
        mid = model.stem(self.x)
        torch.testing.assert_close(model.trunk(mid), model.features(self.x))

    def test_the_block_adds_no_parameters(self):
        """So a v9 checkpoint has the same state_dict as a v8 one, and the
        comparison against R-aug-t2 changes capacity by nothing."""
        plain, generating = build("R-aug-t2"), build("R-gen")
        self.assertEqual(parameter_breakdown(plain), parameter_breakdown(generating))
        self.assertEqual(set(plain.state_dict()), set(generating.state_dict()))

    def test_forward_is_the_plain_backbone_even_when_a_block_is_declared(self):
        """Fig. 2 separates the data manipulation block from forecasting, so
        `forward` - and with it every evaluation - must not call it."""
        model = build("R-gen").eval()
        with torch.no_grad():
            torch.testing.assert_close(model(self.x), model.head(model.features(self.x)))
            torch.testing.assert_close(model(self.x), model.forward_pair(self.x).logits)

    def test_forward_pair_returns_a_second_branch_only_when_declared(self):
        model = build("R-gen").eval()
        with torch.no_grad():
            branches = model.forward_pair(self.x)
        self.assertIsNotNone(branches.logits_generated)
        self.assertEqual(branches.logits_generated.shape, branches.logits.shape)
        self.assertGreater(
            float((branches.logits_generated - branches.logits).abs().sum()), 0.0)

        plain = build("R-aug-t2").eval()
        with torch.no_grad():
            empty = plain.forward_pair(self.x)
        self.assertIsNone(empty.logits_generated)
        self.assertIsNone(empty.features_generated)

    def test_the_generation_cells_keep_v8_1_augmentation(self):
        """The factor under test is the block, so `augment` must be identical."""
        reference = VARIANTS["R-aug-t2"]["augment"]
        for name in self.GENERATION_CELLS:
            self.assertEqual(VARIANTS[name]["augment"], reference, name)
            self.assertEqual(VARIANTS[name]["generate"],
                             {"style": "position", "pool": 2}, name)

    def test_mse_is_zero_on_identical_branches_and_sums_over_classes(self):
        logits = torch.randn(4, 6, generator=torch.Generator().manual_seed(2))
        self.assertAlmostEqual(float(mse_consistency(logits, logits)), 0.0, places=6)
        other = torch.randn(4, 6, generator=torch.Generator().manual_seed(3))
        # Eq. (S4) prints no 1/C, so the class axis is summed, not averaged.
        expected = float((logits.softmax(1) - other.softmax(1)).pow(2).sum(1).mean())
        self.assertAlmostEqual(float(mse_consistency(logits, other)), expected, places=6)

    def test_mse_is_bounded_by_two(self):
        """It compares probabilities, so it cannot outgrow L_ce by scale."""
        one = torch.tensor([[50.0, -50.0, 0.0, 0.0, 0.0, 0.0]])
        two = torch.tensor([[-50.0, 50.0, 0.0, 0.0, 0.0, 0.0]])
        self.assertLessEqual(float(mse_consistency(one, two)), 2.0 + 1e-6)

    def test_the_declared_lambda_is_the_one_the_grid_uses(self):
        self.assertEqual(LAMBDA_MSE, 0.5)
        self.assertEqual(VARIANTS["R-gen"]["lambda_mse"], 0.5)
        self.assertEqual(VARIANTS["R-gen-m10"]["lambda_mse"], 1.0)
        self.assertFalse(VARIANTS["R-gen"]["ce_on_generated"])
        self.assertTrue(VARIANTS["R-gen-ce2"]["ce_on_generated"])

    def test_training_records_both_loss_terms(self):
        config = load_config(Path("configs/feature_generation.json"))
        config = copy.deepcopy(config)
        config["training"] = {"epochs": 1, "batch_size": 32}
        x, y = load_source()
        with tempfile.TemporaryDirectory() as directory:
            summary = train_one("R-gen", 1042, config, x, y,
                                Path(directory) / "cell", "cpu", False,
                                learning_rate=0.0003)
        record = summary["history"][-1]
        self.assertIn("loss_ce", record)
        self.assertIn("loss_mse", record)
        # loss_s2 is the weighted total of Eq. (4).
        self.assertAlmostEqual(record["loss_s2"],
                               record["loss_ce"] + 0.5 * record["loss_mse"], places=5)
        self.assertEqual(summary["lambda_mse"], 0.5)
        self.assertIsNotNone(summary["negative_scale_fraction"])

    def test_a_cell_without_a_block_records_no_components(self):
        config = copy.deepcopy(load_config(Path("configs/feature_generation.json")))
        config["training"] = {"epochs": 1, "batch_size": 32}
        x, y = load_source()
        with tempfile.TemporaryDirectory() as directory:
            summary = train_one("R-aug-t2", 1042, config, x, y,
                                Path(directory) / "cell", "cpu", False,
                                learning_rate=0.0003)
        self.assertNotIn("loss_mse", summary["history"][-1])
        self.assertIsNone(summary["lambda_mse"])


class AugmentationTests(unittest.TestCase):
    def setUp(self):
        from src import augment
        self.augment = augment
        x, self.y = load_source()
        self.z = normalize.apply(
            normalize.fit("signed_log_then_per_sample", x), x)

    def test_the_block_layout_matches_the_labels(self):
        offsets = self.augment.block_offsets(self.z, self.y)
        self.assertEqual(sorted(offsets), [1, 2, 3])
        for label, ((a, b), (c, d)) in self.augment.SOURCE_BLOCKS.items():
            self.assertTrue((self.y[a:b] == label).all(), label)
            self.assertTrue((self.y[c:d] == label).all(), label)
        # Acetaldehyde, Acetone and Toluene have one block each, so no offset.
        self.assertNotIn(4, offsets)

    def test_lambda_is_inert_under_per_sample_normalisation(self):
        """Eq. (7) mixes two samples' mean and variance, but per-sample
        normalisation sets every sample's to 0 and 1, so the weight does
        nothing. Recorded rather than assumed; see baseline.md."""
        for lam in (0.0, 0.5, 1.0):
            drawn = self.augment.paper_noise(
                self.z, self.y, lam, np.random.default_rng(7))
            reference = self.augment.paper_noise(
                self.z, self.y, 0.5, np.random.default_rng(7))
            np.testing.assert_allclose(drawn, reference, atol=1e-12)

    def test_the_direction_is_a_unit_vector_and_reproducible(self):
        for source in self.augment.DIRECTION_SOURCES:
            unit, scale = self.augment.drift_direction(self.z, self.y, source)
            self.assertAlmostEqual(float(np.linalg.norm(unit)), 1.0, places=10)
            self.assertGreater(scale, 0.0)
            again, _ = self.augment.drift_direction(self.z, self.y, source)
            np.testing.assert_allclose(unit, again, atol=0)

    def test_the_average_direction_weighs_the_three_classes_equally(self):
        offsets = self.augment.block_offsets(self.z, self.y)
        expected = np.mean([v / np.linalg.norm(v) for v in offsets.values()], axis=0)
        expected = expected / np.linalg.norm(expected)
        unit, _ = self.augment.drift_direction(self.z, self.y, "average")
        np.testing.assert_allclose(unit, expected, atol=1e-12)

    def test_a_directed_only_augmentation_moves_along_one_line(self):
        unit, _ = self.augment.drift_direction(self.z, self.y, "average")
        out = self.augment.augment(self.z, self.y, np.random.default_rng(0),
                                   isotropic=False, direction="average")
        delta = out - self.z
        # Every displacement is a non-negative multiple of the same unit vector.
        along = delta @ unit
        np.testing.assert_allclose(delta, along[:, None] * unit[None, :], atol=1e-10)
        self.assertTrue((along >= -1e-10).all())

    def test_an_augmentation_must_do_something(self):
        with self.assertRaisesRegex(ProtocolError, "isotropic, directed, or both"):
            self.augment.augment(self.z, self.y, np.random.default_rng(0),
                                 isotropic=False, direction=None)

    def test_training_doubles_the_rows_and_keeps_the_labels(self):
        x, y = load_source()
        with tempfile.TemporaryDirectory() as directory:
            plain = train_one("R-fig-logps", 1042, _smoke_config(), x, y,
                              Path(directory) / "a", "cpu", save_every_epoch=False)
        with tempfile.TemporaryDirectory() as directory:
            augmented = train_one("R-aug-t3", 1042, _smoke_config(), x, y,
                                  Path(directory) / "b", "cpu", save_every_epoch=False)
        self.assertEqual(plain["training_rows"], 445)
        self.assertEqual(augmented["training_rows"], 890)
        self.assertIsNone(plain["augment"])
        self.assertEqual(augmented["augment"],
                         {"isotropic": False, "direction": "ethanol",
                          "displacement": 3.0})

    def test_evaluation_never_augments(self):
        """Fig. 2: 'During forecasting, the program separates the data
        manipulation block from the prediction process.'"""
        import inspect
        from src import evaluate
        self.assertNotIn("augment", inspect.getsource(evaluate))

    def test_the_displacement_lives_on_the_variant(self):
        from src.model import VARIANTS as spec
        self.assertEqual(spec["R-aug-t2"]["augment"]["displacement"], 2.0)
        self.assertEqual(spec["R-aug-t3"]["augment"]["displacement"], 3.0)
        self.assertEqual(spec["R-aug-t4"]["augment"]["displacement"], 4.0)
        # v8.0's cells keep the 18 they were run with.
        self.assertEqual(spec["R-aug-ethd"]["augment"]["displacement"], 18.0)

    def test_the_displacement_scales_the_offset_not_the_class_radius(self):
        """T multiplies ||block offset||, the distance between Batch 1's two
        acquisition sessions - not the class radius, which is a different
        quantity that happens to be numerically close."""
        _, scale = self.augment.drift_direction(self.z, self.y, "ethanol")
        for multiplier in (2.0, 4.0):
            out = self.augment.augment(
                self.z, self.y, np.random.default_rng(0), isotropic=False,
                direction="ethanol", displacement=multiplier)
            along = np.linalg.norm(out - self.z, axis=1)
            self.assertLessEqual(along.max(), multiplier * scale + 1e-9)
            self.assertGreater(along.max(), 0.9 * multiplier * scale)

    def test_the_v8_1_config_is_valid_and_pins_its_grid(self):
        config = load_config(ROOT / "configs" / "displacement.json")
        self.assertEqual(config["variants"],
                         ["R-fig-logps", "R-aug-t2", "R-aug-t3", "R-aug-t4"])
        self.assertEqual(config["augmentation"]["displacement_multiplier"],
                         [2.0, 3.0, 4.0])
        self.assertFalse(config["augmentation"]["isotropic"])
        self.assertTrue(config["prediction"]["recorded_before_the_run"])

    def test_the_v8_config_is_valid_and_pins_its_grid(self):
        config = load_config(ROOT / "configs" / "augmentation.json")
        self.assertEqual(config["variants"],
                         ["R-fig-logps", "R-aug-paper", "R-aug-eth", "R-aug-ethd"])
        self.assertEqual(config["augmentation"]["displacement_multiplier"], 18.0)
        # The prediction is recorded before the run so it cannot be revised after.
        self.assertTrue(config["prediction"]["recorded_before_the_run"])
        self.assertEqual(config["prediction"]["coverage"]["R-aug-ethd"], 0.813)
        changed = copy.deepcopy(config)
        changed["variants"] = config["variants"][:3]
        with self.assertRaisesRegex(ProtocolError, "fixes variants"):
            validate(changed)


class InputNormalisationTests(unittest.TestCase):
    def test_every_normalizer_is_finite_and_standardised(self):
        x, _ = load_source()
        for kind in normalize.NORMALIZERS:
            z = normalize.apply(normalize.fit(kind, x), x)
            self.assertEqual(z.shape, x.shape, kind)
            self.assertTrue(np.isfinite(z).all(), kind)
            self.assertAlmostEqual(float(z.std()), 1.0, places=6, msg=kind)

    def test_composed_normalizers_equal_their_stages(self):
        x, _ = load_source()
        scaler = normalize.fit("standard_scaler", x)
        expected = normalize.apply({"kind": "per_sample"},
                                   normalize.apply(scaler, x))
        actual = normalize.apply(normalize.fit("standard_then_per_sample", x), x)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)

    def test_per_statistic_group_standardises_each_statistic_across_sensors(self):
        x, _ = load_source()
        z = normalize.apply(normalize.fit("per_statistic_group", x), x)
        grouped = z.reshape(-1, normalize.N_SENSORS, normalize.N_STATISTICS)
        np.testing.assert_allclose(grouped.mean(axis=1), 0, atol=1e-10)
        np.testing.assert_allclose(grouped.std(axis=1), 1, atol=1e-10)
        # Sensor-major layout: statistic k of sensor s sits at 8*(s-1)+k.
        self.assertEqual(normalize.N_SENSORS * normalize.N_STATISTICS, 128)

    def test_only_the_scaler_stage_carries_fitted_parameters(self):
        x, _ = load_source()
        for kind in ("per_sample", "signed_log_then_per_sample", "per_statistic_group"):
            self.assertEqual(normalize.fit(kind, x), {"kind": kind}, kind)
        fitted = normalize.fit("standard_then_per_sample", x)
        self.assertEqual(sorted(fitted), ["kind", "mean", "std"])

    def test_the_v7_5_escalation_config_adds_only_the_reserve_seeds(self):
        four = load_config(ROOT / "configs" / "input_normalisation.json")
        five = load_config(ROOT / "configs" / "input_normalisation_5seed.json")
        self.assertEqual(five["variants"], four["variants"])
        self.assertEqual(five["learning_rates"], four["learning_rates"])
        self.assertEqual(five["training"], four["training"])
        self.assertEqual(five["optimizer"], four["optimizer"])
        self.assertEqual(five["seeds"], four["seeds"] + four["reserve_seeds"])

    def test_the_v7_4_config_is_valid_and_pins_its_grid(self):
        config = load_config(ROOT / "configs" / "input_normalisation.json")
        self.assertEqual(config["variants"],
                         ["R-fig-ps", "R-fig-ssps", "R-fig-logps", "R-fig-grp"])
        changed = copy.deepcopy(config)
        changed["variants"] = ["R-fig-ps"]
        with self.assertRaisesRegex(ProtocolError, "fixes variants"):
            validate(changed)


class HeadNormalisationTests(unittest.TestCase):
    def test_the_four_cells_cross_both_normalisations_at_equal_cost(self):
        from src.model import VARIANTS as spec
        cells = {(spec[v]["normalizer"], spec[v].get("head_norm", "batchnorm")): v
                 for v in ("R-fig", "R-fig-ln", "R-fig-ps", "R-fig-ps-ln")}
        self.assertEqual(set(cells), {("standard_scaler", "batchnorm"),
                                      ("standard_scaler", "layernorm"),
                                      ("per_sample", "batchnorm"),
                                      ("per_sample", "layernorm")})
        counts = {parameter_breakdown(build(v))["total"] for v in cells.values()}
        self.assertEqual(counts, {3_156_390}, "the four cells must cost the same")

    def test_the_head_carries_the_declared_normalisation(self):
        for variant, expected in (("R-fig", torch.nn.BatchNorm1d),
                                  ("R-fig-ps", torch.nn.BatchNorm1d),
                                  ("R-fig-ln", torch.nn.LayerNorm),
                                  ("R-fig-ps-ln", torch.nn.LayerNorm)):
            self.assertIsInstance(build(variant).head.norm, expected, variant)

    def test_no_batchnorm_adapts_to_the_evaluation_batch(self):
        """track_running_stats=False would normalise a target batch by its own
        statistics at inference, which is test-time adaptation on target data and
        makes a prediction depend on which other samples share its batch. It is
        prohibited; see baseline.md section 4.3."""
        for variant in VARIANTS:
            for module in build(variant).modules():
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    self.assertTrue(module.track_running_stats, variant)
                    self.assertIsNotNone(module.running_mean, variant)

    def test_an_unknown_head_normalisation_is_refused(self):
        from src.model import _head_norm
        with self.assertRaisesRegex(ProtocolError, "unknown head normalisation"):
            _head_norm("groupnorm")

    def test_the_v7_2_config_is_valid_and_pins_its_grid(self):
        config = load_config(ROOT / "configs" / "head_normalisation.json")
        self.assertEqual(config["variants"],
                         ["R-fig", "R-fig-ln", "R-fig-ps", "R-fig-ps-ln"])
        self.assertEqual(config["learning_rates"], [0.0003])
        changed = copy.deepcopy(config)
        changed["learning_rates"] = [0.001]
        with self.assertRaisesRegex(ProtocolError, "fixes learning_rates"):
            validate(changed)


class LearningRateSweepTests(unittest.TestCase):
    def test_the_sweep_config_is_valid_and_declares_six_cells(self):
        config = load_config(ROOT / "configs" / "channel_restore.json")
        self.assertEqual(config["variants"], ["R-txt-ps", "R-fig-ps"])
        self.assertEqual(config["learning_rates"], [0.001, 0.0003, 0.0001])
        self.assertEqual(len(config["variants"]) * len(config["learning_rates"]), 6)

    def test_the_sweep_version_pins_its_learning_rates(self):
        config = load_config(ROOT / "configs" / "channel_restore.json")
        changed = copy.deepcopy(config)
        changed["learning_rates"] = [0.001, 0.0005]
        with self.assertRaisesRegex(ProtocolError, "fixes learning_rates"):
            validate(changed)

    def test_the_learning_rate_reaches_the_optimizer_and_the_checkpoint(self):
        x, y = load_source()
        with tempfile.TemporaryDirectory() as directory:
            checkpoints = Path(directory) / "ck"
            summary = train_one("R-txt-ps", 1042, _smoke_config(), x, y, checkpoints,
                                "cpu", save_every_epoch=False, learning_rate=0.0003)
            self.assertEqual(summary["learning_rate"], 0.0003)
            payload = torch.load(checkpoints / "final.pt", weights_only=False)
            self.assertEqual(payload["learning_rate"], 0.0003)
            # StepLR(25, 0.5) has not stepped after two epochs.
            self.assertAlmostEqual(summary["history"][-1]["learning_rate"], 0.0003)

    def test_a_smaller_rate_gives_a_different_trajectory(self):
        x, y = load_source()
        histories = []
        for rate in (0.001, 0.0001):
            with tempfile.TemporaryDirectory() as directory:
                histories.append(train_one(
                    "R-txt-ps", 1042, _smoke_config(), x, y, Path(directory) / "ck",
                    "cpu", save_every_epoch=False, learning_rate=rate)["history"])
        self.assertLess(histories[1][-1]["accuracy"], histories[0][-1]["accuracy"])

    def test_the_checkpoint_schedule_is_dense_early_and_sparse_late(self):
        from src.train import epoch_is_saved
        saved = [e for e in range(1, 101) if epoch_is_saved(e)]
        self.assertEqual(saved[:20], list(range(1, 21)))
        self.assertEqual(saved[20:], list(range(25, 101, 5)))
        self.assertEqual(len(saved), 36)


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

    def test_the_normalizer_comes_from_the_variant_not_the_config(self):
        x, y = load_source()
        expected = {"R-txt": "standard_scaler", "R-txt-ps": "per_sample",
                    "R-lite": "standard_scaler", "R-lite-ps": "per_sample"}
        for variant, kind in expected.items():
            with tempfile.TemporaryDirectory() as directory:
                summary = train_one(variant, 1042, _smoke_config(), x, y,
                                    Path(directory) / "ck", "cpu",
                                    save_every_epoch=False)
            self.assertEqual(summary["normalizer"], kind, variant)

    def test_checkpoints_carry_the_normalizer_and_are_written_per_epoch(self):
        x, y = load_source()
        with tempfile.TemporaryDirectory() as directory:
            checkpoints = Path(directory) / "ck"
            train_one("R-lite-ps", 1042, _smoke_config(), x, y, checkpoints,
                      "cpu", save_every_epoch=True)
            names = sorted(path.name for path in checkpoints.glob("*.pt"))
            self.assertEqual(names, ["epoch_001.pt", "epoch_002.pt", "final.pt"])
            payload = torch.load(checkpoints / "final.pt", weights_only=False)
            self.assertEqual(payload["normalizer"], {"kind": "per_sample"})
            self.assertEqual(payload["variant"], "R-lite-ps")


if __name__ == "__main__":
    unittest.main()


class DriftProjectionTests(unittest.TestCase):
    def setUp(self):
        from src import normalize, project
        from src.data import load_source
        self.normalize, self.project = normalize, project
        self.x, self.y = load_source()
        self.z = normalize.apply(normalize.fit("signed_log_then_per_sample", self.x),
                                 self.x)

    def test_each_subspace_is_orthonormal_and_projection_is_idempotent(self):
        rows = np.arange(len(self.z))
        for name, k in (("offset_axis", 1), ("sub3", 3), ("eth", 1)):
            basis = self.project.drift_basis(self.z, rows, name)
            self.assertEqual(basis.shape, (128, k), name)
            np.testing.assert_allclose(basis.T @ basis, np.eye(k), atol=1e-10)
            once = self.project.project(self.z, basis)
            twice = self.project.project(once, basis)
            np.testing.assert_allclose(once, twice, atol=1e-10)
            np.testing.assert_allclose(once @ basis, 0.0, atol=1e-9)

    def test_the_session_offsets_match_the_augmentation_module(self):
        """Same rows, same offsets: project.py must agree with augment.py."""
        from src.augment import block_offsets
        theirs = block_offsets(self.z, self.y)
        mine = self.project.session_offsets(self.z, np.arange(len(self.z)))
        for label in theirs:
            np.testing.assert_allclose(mine[label], theirs[label])

    def test_the_basis_depends_only_on_the_session_rows(self):
        """Rows outside the six acquisition blocks may change freely."""
        from src.augment import SOURCE_BLOCKS
        inside = np.zeros(len(self.z), dtype=bool)
        for (a, b), (c, d) in SOURCE_BLOCKS.values():
            inside[a:b] = True
            inside[c:d] = True
        altered = self.z.copy()
        altered[~inside] += 10.0
        rows = np.arange(len(self.z))
        for name in ("offset_axis", "sub3", "eth"):
            np.testing.assert_allclose(
                self.project.drift_basis(self.z, rows, name),
                self.project.drift_basis(altered, rows, name))

    def test_a_fold_fits_its_basis_from_its_own_rows_only(self):
        from src.cv import stratified_folds
        folds = stratified_folds(self.y)
        held = folds[0]
        train = np.setdiff1d(np.arange(len(self.y)), held)
        params = self.normalize.fit("logps_proj_offset_axis", self.x[train], rows=train)
        basis = np.asarray(params["basis"])
        self.assertEqual(basis.shape, (128, 1))
        # Perturbing the held-out rows cannot change a basis fitted without them.
        altered = self.x.copy()
        altered[held] *= 3.0
        again = self.normalize.fit("logps_proj_offset_axis", altered[train], rows=train)
        np.testing.assert_allclose(basis, np.asarray(again["basis"]))

    def test_too_few_session_rows_is_refused(self):
        with self.assertRaisesRegex(ProtocolError, "too few rows"):
            self.project.session_offsets(self.z[:100], np.arange(100))

    def test_the_projected_normaliser_removes_the_axis_and_evaluation_applies_it(self):
        """What evaluate.predict sees is normalize.apply with the stored params,
        so the projection reaches every target sample through that path."""
        params = self.normalize.fit("logps_proj_offset_axis", self.x)
        basis = np.asarray(params["basis"])
        out = self.normalize.apply(params, self.x)
        np.testing.assert_allclose(out @ basis, 0.0, atol=1e-9)
        plain = self.normalize.apply({"kind": "signed_log_then_per_sample"}, self.x)
        self.assertGreater(float(np.abs(plain @ basis).mean()), 0.1)

    def test_the_v12_variants_carry_no_augmentation_or_block(self):
        from src.model import VARIANTS, build
        for name, kind in (("R-proj-axis", "logps_proj_offset_axis"),
                           ("R-proj-sub3", "logps_proj_sub3"),
                           ("R-proj-eth", "logps_proj_eth")):
            spec = VARIANTS[name]
            self.assertEqual(spec["normalizer"], kind)
            self.assertNotIn("augment", spec)
            self.assertNotIn("generate", spec)
            self.assertEqual(spec["channels"], VARIANTS["R-fig-logps"]["channels"])
            self.assertIsNone(build(name).generation)

