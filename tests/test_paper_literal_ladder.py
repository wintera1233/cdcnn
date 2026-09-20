import json
import unittest
from pathlib import Path

import torch

from src.a3_stage import CONTRASTIVE_STAGES, PAPER_LITERAL_STAGES
from src.cdcnn_ablation import (
    AUGMENTED_STAGES,
    CDCNNModel,
    FEATURE_STAGES,
    LITERAL_EXPERIMENT_STAGES,
    LITERAL_IMPLEMENTATION_VERSION,
    validate_config,
)
from src.input_transform import input_transform_for


ROOT = Path(__file__).resolve().parents[1]


class PaperLiteralLadderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = json.loads(
            (ROOT / "configs/cdcnn_v6_6_paper_literal.json").read_text(encoding="utf-8"))
        cls.canonical = json.loads((ROOT / "configs/cdcnn_v6.json").read_text(encoding="utf-8"))

    def test_config_validates_and_shares_canonical_protocol(self):
        validate_config(self.cfg)
        self.assertEqual(self.cfg["implementation_version"], LITERAL_IMPLEMENTATION_VERSION)
        for section in ("dataset", "seeds", "training", "architecture", "loss",
                        "a3_numerical_stability", "saved_cv_folds"):
            self.assertEqual(self.cfg[section], self.canonical[section], section)

    def test_ladder_adds_one_component_per_step(self):
        expected = [
            ("B0-stab-PS", False, False, False),
            ("A1-stab-PS", True, False, False),
            ("A2-lit-PS", True, True, False),
            ("A3-lit-PS", True, True, True),
        ]
        self.assertEqual([row[0] for row in expected], list(LITERAL_EXPERIMENT_STAGES))
        for stage, augmented, feature, contrastive in expected:
            self.assertEqual(stage in AUGMENTED_STAGES, augmented, stage)
            self.assertEqual(stage in FEATURE_STAGES, feature, stage)
            self.assertEqual(stage in CONTRASTIVE_STAGES, contrastive, stage)
            self.assertEqual(input_transform_for(stage), "per_sample", stage)

    def test_literal_stages_restyle_the_residual_branch(self):
        z = torch.randn(4, 128, 128, generator=torch.Generator().manual_seed(11))
        for stage in ("A2-lit-PS", "A3-lit-PS"):
            self.assertIn(stage, PAPER_LITERAL_STAGES)
            _, details = CDCNNModel(stage).generate_features(z, torch.Generator().manual_seed(12))
            self.assertEqual(details["restyled_component_name"], "paper_L_residual")
            torch.testing.assert_close(details["restyled_component"], details["z_high"])

    def test_semantic_stages_still_restyle_the_pooled_branch(self):
        z = torch.randn(4, 128, 128, generator=torch.Generator().manual_seed(13))
        for stage in ("A2-semantic", "A3", "A2-stab-PS", "A3-PS"):
            self.assertNotIn(stage, PAPER_LITERAL_STAGES)
            _, details = CDCNNModel(stage).generate_features(z, torch.Generator().manual_seed(14))
            self.assertEqual(details["restyled_component_name"], "low_frequency_like_pooled_upsampled")
            torch.testing.assert_close(details["restyled_component"], details["z_low"])

    def test_literal_and_semantic_generate_different_features(self):
        z = torch.randn(4, 128, 128, generator=torch.Generator().manual_seed(15))
        literal, _ = CDCNNModel("A2-lit-PS").generate_features(z, torch.Generator().manual_seed(16))
        semantic, _ = CDCNNModel("A2-stab-PS").generate_features(z, torch.Generator().manual_seed(16))
        self.assertFalse(torch.allclose(literal, semantic))


if __name__ == "__main__":
    unittest.main()
