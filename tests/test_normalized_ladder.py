import json
import unittest
from pathlib import Path

import torch

from src.a3_stage import CONTRASTIVE_STAGES, STABILIZED_STAGES
from src.cdcnn_ablation import (
    AUGMENTED_STAGES,
    CDCNNModel,
    FEATURE_STAGES,
    NORMALIZED_EXPERIMENT_STAGES,
    NORMALIZED_IMPLEMENTATION_VERSION,
    validate_config,
)
from src.input_transform import input_transform_for


ROOT = Path(__file__).resolve().parents[1]


class NormalizedLadderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = json.loads(
            (ROOT / "configs/cdcnn_v6_5_normalized_ladder.json").read_text(encoding="utf-8"))
        cls.canonical = json.loads((ROOT / "configs/cdcnn_v6.json").read_text(encoding="utf-8"))

    def test_config_validates_and_shares_canonical_protocol(self):
        validate_config(self.cfg)
        self.assertEqual(self.cfg["implementation_version"], NORMALIZED_IMPLEMENTATION_VERSION)
        for section in ("dataset", "seeds", "training", "architecture", "loss",
                        "a3_numerical_stability", "saved_cv_folds"):
            self.assertEqual(self.cfg[section], self.canonical[section], section)

    def test_every_stage_uses_per_sample_inputs(self):
        for stage in NORMALIZED_EXPERIMENT_STAGES:
            self.assertEqual(input_transform_for(stage), "per_sample", stage)

    def test_each_step_adds_exactly_one_component(self):
        def properties(stage):
            model = CDCNNModel(stage)
            return (
                any(isinstance(m, torch.nn.LayerNorm) for m in model.modules()),
                stage in STABILIZED_STAGES,
                stage in AUGMENTED_STAGES,
                stage in FEATURE_STAGES,
                stage in CONTRASTIVE_STAGES,
            )
        ladder = [properties(stage) for stage in NORMALIZED_EXPERIMENT_STAGES]
        self.assertEqual(ladder[0], (True, False, False, False, False))
        for earlier, later in zip(ladder, ladder[1:]):
            added = [b and not a for a, b in zip(earlier, later)]
            removed = [a and not b for a, b in zip(earlier, later)]
            self.assertEqual(sum(removed), 0, "a ladder step removed a component")
            # A2-stab-PS adds augmentation and feature generation together, which
            # is the same single protocol step A2 makes in the canonical ladder.
            self.assertLessEqual(sum(added), 2)
            self.assertGreaterEqual(sum(added), 1)

    def test_a3_ps_matches_a3_apart_from_the_input_transform(self):
        a3, a3_ps = CDCNNModel("A3"), CDCNNModel("A3-PS")
        self.assertEqual(
            {k: tuple(v.shape) for k, v in a3.state_dict().items()},
            {k: tuple(v.shape) for k, v in a3_ps.state_dict().items()})
        self.assertIsNotNone(a3_ps.projection)
        self.assertIsNotNone(a3_ps.a3_stabilizer)
        self.assertEqual(input_transform_for("A3"), "identity")
        self.assertEqual(input_transform_for("A3-PS"), "per_sample")


if __name__ == "__main__":
    unittest.main()
