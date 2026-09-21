import ast
import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.a3_stage import CONTRASTIVE_ON_ZF_STAGES, CONTRASTIVE_STAGES
from src.cdcnn_ablation import (
    ALL_STAGES,
    CDCNNModel,
    DataAccessGuard,
    ProtocolError,
    build_optimizer_and_scheduler,
    clip_a3_gradients,
    compute_loss,
    generate_a1_views,
    probability_consistency_mse,
    resolve_training_device,
    supervised_contrastive_mean,
    validate_config,
    seed_everything,
)
from scripts.run_cdcnn_v6_full import enforce_cuda_worker_strategy


ROOT = Path(__file__).resolve().parents[1]


class ConfigAndProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = json.loads((ROOT / "configs/cdcnn_v6.json").read_text(encoding="utf-8"))

    def test_canonical_config_passes(self):
        validate_config(self.cfg)

    def test_training_drift_is_rejected(self):
        changed = copy.deepcopy(self.cfg)
        changed["training"]["epochs"] = 99
        with self.assertRaisesRegex(ProtocolError, "training.epochs"):
            validate_config(changed)

    def test_a3_stability_drift_is_rejected(self):
        changed = copy.deepcopy(self.cfg)
        changed["a3_numerical_stability"]["gradient_max_norm"] = 5.0
        with self.assertRaisesRegex(ProtocolError, "gradient_max_norm"):
            validate_config(changed)

    def test_legacy_2d_framework_is_disabled_and_has_no_settings(self):
        disabled = json.loads((ROOT / "configs/resnet.json").read_text(encoding="utf-8"))
        self.assertEqual(disabled["status"], "disabled")
        self.assertNotIn("training", disabled)
        self.assertNotIn("architecture", disabled)
        from src.resnet_baseline import Legacy2DFrameworkDisabled, main
        with self.assertRaisesRegex(Legacy2DFrameworkDisabled, "banned"):
            main()

    def test_active_source_does_not_construct_2d_convolutions(self):
        violations = []
        forbidden_name = "Conv" + "2d"
        for directory in (ROOT / "src", ROOT / "scripts"):
            for path in directory.glob("*.py"):
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                for node in ast.walk(tree):
                    if not isinstance(node, ast.Call):
                        continue
                    name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
                    if name == forbidden_name:
                        violations.append(str(path.relative_to(ROOT)))
        self.assertEqual(violations, [])

    def test_cpu_device_is_rejected_by_cuda_protocol(self):
        changed = copy.deepcopy(self.cfg)
        changed["training"]["device"] = "cpu"
        with self.assertRaisesRegex(ProtocolError, "training.device"):
            validate_config(changed)
        with self.assertRaisesRegex(ProtocolError, "requires CUDA"):
            resolve_training_device(changed["training"])

    def test_cuda_full_launcher_rejects_parallel_workers(self):
        with self.assertRaisesRegex(ProtocolError, "max-workers 1"):
            enforce_cuda_worker_strategy(self.cfg, 4)
        enforce_cuda_worker_strategy(self.cfg, 1)

    def test_all_modes_share_backbone_geometry(self):
        for stage in ALL_STAGES:
            model = CDCNNModel(stage)
            x = torch.zeros(2, 1, 128)
            with torch.no_grad():
                z = model.forward_to_block3(x)
                logits = model(x)
            self.assertEqual(tuple(z.shape), (2, 128, 128))
            self.assertEqual(tuple(logits.shape), (2, 6))
            self.assertEqual(sum(isinstance(m, torch.nn.Conv2d) for m in model.modules()), 0)
            # A learned head exists only for this project's post-FC128 placement;
            # the paper's pre-FC128 placement normalizes the latent directly.
            self.assertEqual(
                model.projection is not None,
                stage in CONTRASTIVE_STAGES and stage not in CONTRASTIVE_ON_ZF_STAGES)

    def test_same_seed_gives_identical_shared_initialization(self):
        states = {}
        for stage in ALL_STAGES:
            seed_everything(1042)
            states[stage] = {
                key: value.detach().clone() for key, value in CDCNNModel(stage).named_parameters()
                if not key.startswith("projection.")
            }
        reference = states["B0"]
        for stage, state in states.items():
            self.assertEqual(reference.keys(), state.keys(), stage)
            for key in reference:
                torch.testing.assert_close(reference[key], state[key], msg=lambda msg: f"{stage} {key}: {msg}")

    def test_target_access_is_blocked_until_freeze(self):
        guard = DataAccessGuard(ROOT, self.cfg["dataset"])
        source = guard.load(1, "unit test")
        self.assertEqual(source.x.shape, (445, 128))
        with self.assertRaisesRegex(ProtocolError, "before checkpoint freeze"):
            guard.load(2, "forbidden unit-test access")
        self.assertEqual([e["batch"] for e in guard.events if e["event"] == "raw_file_load"], [1])


class A1Tests(unittest.TestCase):
    def test_augmentation_mixes_variances_and_uses_sqrt_as_numpy_scale(self):
        x = np.vstack([
            np.arange(128, dtype=np.float64),
            np.arange(128, dtype=np.float64) * 2.0,
            np.arange(128, dtype=np.float64) + 10.0,
            np.arange(128, dtype=np.float64) * 3.0,
        ])
        y = np.array([1, 1, 2, 2])
        augmented, provenance = generate_a1_views(
            x, y, np.arange(4), np.arange(1, 5), 123, "unit")
        self.assertEqual(augmented.shape, x.shape)
        self.assertTrue(provenance.same_class.all())
        self.assertTrue(provenance.non_self.all())
        expected_variance = (
            provenance["lambda"] * provenance["anchor_variance"]
            + (1.0 - provenance["lambda"]) * provenance["partner_variance"]
        )
        np.testing.assert_allclose(provenance["mixed_variance"], expected_variance, rtol=0, atol=0)
        np.testing.assert_allclose(
            provenance["normal_api_scale"], np.sqrt(provenance["mixed_variance"]), rtol=1e-15)
        self.assertTrue(np.isfinite(augmented).all())


class A2SemanticsTests(unittest.TestCase):
    def test_channelwise_statistics_are_population_variance_plus_epsilon(self):
        model = CDCNNModel("A2-semantic", epsilon=1e-5)
        z = torch.arange(2 * 128 * 128, dtype=torch.float32).reshape(2, 128, 128) / 1000
        _, details = model.generate_features(z, torch.Generator().manual_seed(7))
        component = details["z_low"]
        expected_mean = component.mean(dim=-1, keepdim=True)
        expected_sigma = torch.sqrt(component.var(dim=-1, unbiased=False, keepdim=True) + 1e-5)
        self.assertEqual(tuple(details["component_mean"].shape), (2, 128, 1))
        self.assertEqual(tuple(details["component_sigma"].shape), (2, 128, 1))
        torch.testing.assert_close(details["component_mean"], expected_mean)
        torch.testing.assert_close(details["component_sigma"], expected_sigma)

    def test_semantic_and_paper_literal_restyle_different_components(self):
        z = torch.randn(4, 128, 128, generator=torch.Generator().manual_seed(8))
        semantic = CDCNNModel("A2-semantic")
        paper = CDCNNModel("A2-paper-literal")
        semantic_generated, semantic_details = semantic.generate_features(
            z, torch.Generator().manual_seed(9))
        paper_generated, paper_details = paper.generate_features(
            z, torch.Generator().manual_seed(9))
        self.assertEqual(semantic_details["restyled_component_name"], "low_frequency_like_pooled_upsampled")
        self.assertEqual(paper_details["restyled_component_name"], "paper_L_residual")
        torch.testing.assert_close(semantic_details["restyled_component"], semantic_details["z_low"])
        torch.testing.assert_close(paper_details["restyled_component"], paper_details["z_high"])
        self.assertFalse(torch.allclose(semantic_generated, paper_generated))

    def test_a3_inherits_semantic_low_branch(self):
        z = torch.randn(2, 128, 128, generator=torch.Generator().manual_seed(81))
        model = CDCNNModel("A3")
        _, details = model.generate_features(z, torch.Generator().manual_seed(82))
        self.assertEqual(details["restyled_component_name"], "low_frequency_like_pooled_upsampled")
        torch.testing.assert_close(details["restyled_component"], details["z_low"])
        self.assertIsNotNone(model.projection)

    def test_canonical_log_normal_rule_and_positive_sigma(self):
        z = torch.randn(3, 128, 128, generator=torch.Generator().manual_seed(10))
        model = CDCNNModel("A2-semantic", epsilon=1e-5)
        seed = 11
        _, details = model.generate_features(z, torch.Generator().manual_seed(seed))
        replay = torch.Generator().manual_seed(seed)
        torch.randn(details["component_mean"].shape, generator=replay)  # sampled-mean noise
        sigma_noise = torch.randn(details["component_sigma"].shape, generator=replay)
        sigma_base = details["sigma_center"].clamp_min(1e-5)
        expected = torch.exp(torch.log(sigma_base) + details["sigma_spread"] / sigma_base * sigma_noise)
        torch.testing.assert_close(details["sampled_sigma"], expected)
        self.assertTrue(bool((details["sampled_sigma"] > 0).all()))

    def test_probability_mse_is_mean_of_per_sample_squared_l2(self):
        original = torch.tensor([[2.0, 0.0], [0.0, 1.0]])
        generated = torch.tensor([[1.0, 1.0], [2.0, 0.0]])
        expected = ((original.softmax(1) - generated.softmax(1)) ** 2).sum(1).mean()
        torch.testing.assert_close(probability_consistency_mse(original, generated), expected)

    def test_a2_loss_contains_ce_and_weighted_probability_mse(self):
        cfg = json.loads((ROOT / "configs/cdcnn_v6.json").read_text(encoding="utf-8"))
        model = CDCNNModel("A2-semantic")
        outputs = {
            "original_logits": torch.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True),
            "generated_logits": torch.tensor([[1.0, 1.0], [1.0, 1.0]], requires_grad=True),
        }
        y = torch.tensor([0, 1])
        parts = compute_loss(model, outputs, y, cfg)
        expected_ce = 0.5 * (F.cross_entropy(outputs["original_logits"], y)
                             + F.cross_entropy(outputs["generated_logits"], y))
        torch.testing.assert_close(parts["total"], expected_ce + 0.5 * parts["mse"])
        self.assertGreater(float(parts["mse"].detach()), 0.0)


class A3AndScheduleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = json.loads((ROOT / "configs/cdcnn_v6.json").read_text(encoding="utf-8"))

    def test_a3_uses_layernorm_without_batchnorm_running_state(self):
        a3 = CDCNNModel("A3", a3_stability=self.cfg["a3_numerical_stability"])
        self.assertTrue(any(isinstance(module, torch.nn.LayerNorm) for module in a3.modules()))
        self.assertFalse(any(
            isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
            and module.track_running_stats for module in a3.modules()))
        for stage in ("B0", "A1", "A2-semantic", "A2-paper-literal"):
            self.assertTrue(any(
                isinstance(module, torch.nn.BatchNorm1d) for module in CDCNNModel(stage).modules()))

    def test_synthetic_companion_does_not_change_original_a3_features(self):
        model = CDCNNModel("A3", a3_stability=self.cfg["a3_numerical_stability"])
        model.train()
        x = torch.randn(4, 1, 128, generator=torch.Generator().manual_seed(120))
        with torch.no_grad():
            original_z = model.forward_to_block3(x)
            original_alone = model.forward_tail_features(original_z)
            outputs = model.training_outputs(x, torch.Generator().manual_seed(121))
        torch.testing.assert_close(outputs["original_features"], original_alone)

    def test_a3_hard_bounds_sigma_residuals_and_contrastive_vectors(self):
        stability = self.cfg["a3_numerical_stability"]
        model = CDCNNModel("A3", a3_stability=stability)
        z = torch.randn(3, 128, 128, generator=torch.Generator().manual_seed(130)) * 1e6
        generated, details = model.generate_features(z, torch.Generator().manual_seed(131))
        self.assertGreaterEqual(float(details["sampled_sigma"].min()), stability["sigma_min"])
        self.assertLessEqual(float(details["sampled_sigma"].max()), stability["sigma_max"])
        self.assertGreaterEqual(float(generated.min()), stability["residual_output_min"])
        self.assertLessEqual(float(generated.max()), stability["residual_output_max"])
        self.assertGreaterEqual(float(details["z_high"].min()), stability["residual_output_min"])
        self.assertLessEqual(float(details["z_high"].max()), stability["residual_output_max"])
        projected = model.project_contrastive(torch.full((4, 128), 1e6))
        self.assertGreaterEqual(float(projected.min()), stability["contrastive_feature_min"])
        self.assertLessEqual(float(projected.max()), stability["contrastive_feature_max"])

    def test_supervised_contrastive_uses_unit_sphere_and_mean_reduction(self):
        projections = torch.tensor([[1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [0.0, 3.0]])
        labels = torch.tensor([0, 0, 1, 1])
        temperature = 0.5
        actual = supervised_contrastive_mean(projections, labels, temperature)
        normalized = F.normalize(projections, dim=1)
        logits = normalized @ normalized.T / temperature
        expected_terms = []
        for i in range(4):
            positive = next(j for j in range(4) if j != i and labels[j] == labels[i])
            denominator = torch.logsumexp(torch.stack([logits[i, j] for j in range(4) if j != i]), dim=0)
            expected_terms.append(-(logits[i, positive] - denominator))
        torch.testing.assert_close(actual, torch.stack(expected_terms).mean())

    def test_a3_gradient_norm_is_clipped_before_optimizer_step(self):
        model = CDCNNModel("A3", a3_stability=self.cfg["a3_numerical_stability"])
        parameter = next(model.parameters())
        parameter.grad = torch.full_like(parameter, 100.0)
        norms = clip_a3_gradients(model, self.cfg)
        self.assertIsNotNone(norms)
        before, after = norms
        self.assertGreater(before, self.cfg["a3_numerical_stability"]["gradient_max_norm"])
        self.assertLessEqual(
            after, self.cfg["a3_numerical_stability"]["gradient_max_norm"] + 1e-5)
        baseline = CDCNNModel("B0")
        baseline_parameter = next(baseline.parameters())
        baseline_parameter.grad = torch.full_like(baseline_parameter, 100.0)
        unchanged = baseline_parameter.grad.detach().clone()
        self.assertIsNone(clip_a3_gradients(baseline, self.cfg))
        torch.testing.assert_close(baseline_parameter.grad, unchanged)

    def test_step_lr_is_shared_and_decays_after_epochs_25_50_75_100(self):
        cfg = json.loads((ROOT / "configs/cdcnn_v6.json").read_text(encoding="utf-8"))
        for stage in ALL_STAGES:
            model = CDCNNModel(stage)
            optimizer, scheduler = build_optimizer_and_scheduler(model, cfg["training"])
            lrs = []
            for _ in range(100):
                optimizer.step()
                scheduler.step()
                lrs.append(optimizer.param_groups[0]["lr"])
            self.assertAlmostEqual(lrs[23], 0.001)
            self.assertAlmostEqual(lrs[24], 0.0005)
            self.assertAlmostEqual(lrs[49], 0.00025)
            self.assertAlmostEqual(lrs[74], 0.000125)
            self.assertAlmostEqual(lrs[99], 0.0000625)


if __name__ == "__main__":
    unittest.main()
