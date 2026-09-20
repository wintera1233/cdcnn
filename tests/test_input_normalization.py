import json
import unittest
from pathlib import Path

import numpy as np
import torch

from src.cdcnn_ablation import (
    CDCNNModel,
    INPUT_EXPERIMENT_STAGES,
    INPUT_IMPLEMENTATION_VERSION,
    ProtocolError,
    validate_config,
)
from src.input_transform import (
    CLIP_LIMIT,
    apply_input_transform,
    input_transform_for,
    prepare_inputs,
    prepare_inputs_from_params,
)


ROOT = Path(__file__).resolve().parents[1]


class _Scaler:
    """Minimal stand-in with the sklearn transform interface."""

    def __init__(self, mean, scale):
        self.mean_, self.scale_ = mean, scale

    def transform(self, x):
        return (x - self.mean_) / self.scale_


class InputTransformTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = json.loads(
            (ROOT / "configs/cdcnn_v6_4_input_norm.json").read_text(encoding="utf-8"))
        cls.canonical = json.loads((ROOT / "configs/cdcnn_v6.json").read_text(encoding="utf-8"))

    def test_config_validates_and_shares_canonical_protocol(self):
        validate_config(self.cfg)
        self.assertEqual(self.cfg["implementation_version"], INPUT_IMPLEMENTATION_VERSION)
        for section in ("dataset", "seeds", "training", "architecture", "loss",
                        "a3_numerical_stability", "saved_cv_folds"):
            self.assertEqual(self.cfg[section], self.canonical[section], section)

    def test_stage_mapping_matches_config(self):
        self.assertEqual(
            self.cfg["input_normalization"],
            {stage: input_transform_for(stage) for stage in INPUT_EXPERIMENT_STAGES})
        for stage in ("B0", "B0-LN", "A3", "A2-stab"):
            self.assertEqual(input_transform_for(stage), "identity", stage)

    def test_identity_leaves_canonical_stages_untouched(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(7, 128))
        scaler = _Scaler(rng.normal(size=128), rng.uniform(0.5, 2.0, size=128))
        for stage in ("B0", "A1", "A2-semantic", "A3"):
            np.testing.assert_array_equal(
                prepare_inputs(stage, scaler, x), scaler.transform(x), err_msg=stage)

    def test_per_sample_is_invariant_to_row_offset_and_gain(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(5, 128))
        shifted = 3.5 * x + 12.0
        np.testing.assert_allclose(
            apply_input_transform("per_sample", x),
            apply_input_transform("per_sample", shifted), rtol=1e-6, atol=1e-6)
        result = apply_input_transform("per_sample", x)
        np.testing.assert_allclose(result.mean(axis=1), 0.0, atol=1e-10)
        np.testing.assert_allclose(result.std(axis=1), 1.0, rtol=1e-6)

    def test_signed_log_and_clip_bound_extreme_values(self):
        x = np.array([[-14757.0, -1.0, 0.0, 1.0, 14757.0]])
        signed = apply_input_transform("signed_log", x)
        self.assertLess(float(np.abs(signed).max()), 10.0)
        np.testing.assert_array_equal(np.sign(signed), np.sign(x))
        clipped = apply_input_transform("clip", x)
        self.assertLessEqual(float(np.abs(clipped).max()), CLIP_LIMIT)
        np.testing.assert_array_equal(clipped[0, 1:4], x[0, 1:4])

    def test_unknown_transform_is_rejected(self):
        with self.assertRaisesRegex(ProtocolError, "Unknown input transform"):
            apply_input_transform("whitening", np.zeros((2, 4)))

    def test_frozen_checkpoint_path_matches_training_path(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=(6, 128)) * 100.0
        mean, scale = rng.normal(size=128), rng.uniform(0.5, 2.0, size=128)
        scaler = _Scaler(mean, scale)
        for stage in INPUT_EXPERIMENT_STAGES:
            np.testing.assert_allclose(
                prepare_inputs(stage, scaler, x),
                prepare_inputs_from_params(input_transform_for(stage), mean, scale, x),
                rtol=0, atol=0, err_msg=stage)

    def test_layernorm_membership_of_new_stages(self):
        expected = {"B0-PS": False, "B0-LN-PS": True, "B0-LN-LOG": True, "B0-LN-CLIP": True}
        for stage, layernorm in expected.items():
            modules = list(CDCNNModel(stage).modules())
            self.assertEqual(any(isinstance(m, torch.nn.LayerNorm) for m in modules), layernorm, stage)
            self.assertEqual(any(isinstance(m, torch.nn.BatchNorm1d) for m in modules), not layernorm, stage)
            self.assertIsNone(CDCNNModel(stage).a3_stabilizer, stage)


if __name__ == "__main__":
    unittest.main()
