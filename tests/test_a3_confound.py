import copy
import json
import unittest
from pathlib import Path

import numpy as np
import torch

from src.cdcnn_ablation import (
    CONFOUND_IMPLEMENTATION_VERSION,
    CDCNNModel,
    ProtocolError,
    _training_arrays,
    clip_a3_gradients,
    compute_loss,
    validate_config,
)


ROOT = Path(__file__).resolve().parents[1]


def _has(model, kind):
    return any(isinstance(module, kind) for module in model.modules())


class ConfoundStageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.canonical = json.loads((ROOT / "configs/cdcnn_v6.json").read_text(encoding="utf-8"))
        cls.cfg = json.loads(
            (ROOT / "configs/cdcnn_v6_3_a3_confound.json").read_text(encoding="utf-8"))

    def test_confound_config_passes_and_shares_canonical_protocol(self):
        validate_config(self.cfg)
        self.assertEqual(self.cfg["implementation_version"], CONFOUND_IMPLEMENTATION_VERSION)
        for section in ("dataset", "seeds", "training", "architecture", "loss",
                        "a3_numerical_stability", "saved_cv_folds"):
            self.assertEqual(self.cfg[section], self.canonical[section], section)

    def test_canonical_version_rejects_confound_stages(self):
        changed = copy.deepcopy(self.canonical)
        changed["stages"] = self.cfg["stages"]
        with self.assertRaisesRegex(ProtocolError, "stages"):
            validate_config(changed)

    def test_normalization_stabilizer_and_projection_assignment(self):
        expected = {  # stage: (LayerNorm, stabilizer, projection head)
            "B0": (False, False, False), "B0-LN": (True, False, False),
            "B0-stab": (True, True, False), "A2-semantic": (False, False, False),
            "A2-stab": (True, True, False), "A3": (True, True, True),
        }
        for stage, (layernorm, stabilized, projection) in expected.items():
            model = CDCNNModel(stage)
            self.assertEqual(_has(model, torch.nn.LayerNorm), layernorm, stage)
            self.assertEqual(_has(model, torch.nn.BatchNorm1d), not layernorm, stage)
            self.assertEqual(model.a3_stabilizer is not None, stabilized, stage)
            self.assertEqual(model.projection is not None, projection, stage)

    def test_a2_stab_is_a3_without_contrastive_term(self):
        torch.manual_seed(5)
        a3 = CDCNNModel("A3")
        a2_stab = CDCNNModel("A2-stab")
        a2_stab.load_state_dict({
            key: value for key, value in a3.state_dict().items()
            if not key.startswith("projection.")})
        x = torch.randn(8, 1, 128, generator=torch.Generator().manual_seed(6))
        y = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        a3_outputs = a3.training_outputs(x, torch.Generator().manual_seed(7))
        stab_outputs = a2_stab.training_outputs(x, torch.Generator().manual_seed(7))
        self.assertEqual(stab_outputs["generation"]["restyled_component_name"],
                         "low_frequency_like_pooled_upsampled")
        torch.testing.assert_close(stab_outputs["generated_logits"], a3_outputs["generated_logits"])
        a3_loss = compute_loss(a3, a3_outputs, y, self.cfg)
        stab_loss = compute_loss(a2_stab, stab_outputs, y, self.cfg)
        self.assertEqual(float(stab_loss["contrastive"]), 0.0)
        self.assertGreater(float(a3_loss["contrastive"]), 0.0)
        torch.testing.assert_close(
            stab_loss["total"], a3_loss["ce"] + self.cfg["loss"]["lambda_mse"] * a3_loss["mse"])

    def test_gradient_clipping_follows_stabilizer(self):
        for stage, clipped in (("B0-LN", False), ("B0-stab", True), ("A2-stab", True)):
            model = CDCNNModel(stage)
            parameter = next(model.parameters())
            parameter.grad = torch.full_like(parameter, 100.0)
            self.assertEqual(clip_a3_gradients(model, self.cfg) is not None, clipped, stage)

    def test_only_augmented_stages_receive_a1_views(self):
        x = np.random.default_rng(0).normal(size=(6, 128))
        y = np.array([1, 1, 2, 2, 3, 3])
        for stage, rows in (("B0-LN", 6), ("B0-stab", 6), ("A2-stab", 12)):
            train_x, _, _ = _training_arrays(stage, x, y, np.arange(6), np.arange(1, 7), 1, "unit")
            self.assertEqual(len(train_x), rows, stage)


if __name__ == "__main__":
    unittest.main()
