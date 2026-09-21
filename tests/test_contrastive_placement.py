import json
import unittest
from pathlib import Path

import torch

from src.a3_stage import CONTRASTIVE_ON_ZF_STAGES, CONTRASTIVE_STAGES, contrastive_term
from src.cdcnn_ablation import (
    CDCNNModel,
    ZF_EXPERIMENT_STAGES,
    ZF_IMPLEMENTATION_VERSION,
    compute_loss,
    seed_everything,
    validate_config,
)


ROOT = Path(__file__).resolve().parents[1]


def _outputs(model, batch=8):
    x = torch.randn(batch, 1, 128, generator=torch.Generator().manual_seed(21))
    return x, model.training_outputs(x, torch.Generator().manual_seed(22))


class ContrastivePlacementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = json.loads(
            (ROOT / "configs/cdcnn_v6_10_contrastive_placement.json").read_text(encoding="utf-8"))

    def test_config_validates_and_pins_the_placement(self):
        validate_config(self.cfg)
        self.assertEqual(self.cfg["implementation_version"], ZF_IMPLEMENTATION_VERSION)
        self.assertEqual(self.cfg["contrastive_placement"]["A3-zf-aln-PS"], "pre_fc128_unit_sphere")
        self.assertEqual(self.cfg["contrastive_placement"]["A3-lit-aln-PS"],
                         "post_fc128_projection_head")

    def test_paper_placement_has_no_learned_head(self):
        zf = CDCNNModel("A3-zf-aln-PS")
        self.assertIn("A3-zf-aln-PS", CONTRASTIVE_STAGES)
        self.assertIsNone(zf.projection)
        self.assertFalse(any(k.startswith("projection") for k in zf.state_dict()))

    def test_backbone_initialisation_is_unaffected_by_dropping_the_head(self):
        def build(stage):
            seed_everything(1042)
            return CDCNNModel(stage).state_dict()
        zf, head = build("A3-zf-aln-PS"), build("A3-lit-aln-PS")
        shared = [k for k in zf if not k.startswith("projection")]
        self.assertEqual(len(shared), 36)
        for key in shared:
            torch.testing.assert_close(zf[key], head[key], msg=lambda m, k=key: f"{k}: {m}")

    def test_paper_placement_uses_the_prefc_latent(self):
        model = CDCNNModel("A3-zf-aln-PS")
        _, outputs = _outputs(model)
        self.assertEqual(tuple(outputs["original_latent"].shape), (8, 128 * 128))
        self.assertEqual(tuple(outputs["original_features"].shape), (8, 128))
        # fc128 applied to the latent must reproduce the classifier feature exactly
        torch.testing.assert_close(model.fc128(outputs["original_latent"]),
                                   outputs["original_features"])

    def test_project_placement_does_not_expose_the_latent(self):
        _, outputs = _outputs(CDCNNModel("A3-lit-aln-PS"))
        self.assertNotIn("original_latent", outputs)

    def test_the_two_placements_give_different_contrastive_values(self):
        y = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        seed_everything(7)
        zf = CDCNNModel("A3-zf-aln-PS")
        seed_everything(7)
        head = CDCNNModel("A3-lit-aln-PS")
        _, zf_out = _outputs(zf)
        _, head_out = _outputs(head)
        zf_term = float(contrastive_term(zf, zf_out, y, self.cfg))
        head_term = float(contrastive_term(head, head_out, y, self.cfg))
        self.assertNotAlmostEqual(zf_term, head_term, places=4)
        for value in (zf_term, head_term):
            self.assertTrue(0.0 < value < 100.0)

    def test_loss_assembly_includes_the_paper_placement(self):
        model = CDCNNModel("A3-zf-aln-PS")
        _, outputs = _outputs(model)
        y = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        parts = compute_loss(model, outputs, y, self.cfg)
        self.assertGreater(float(parts["contrastive"]), 0.0)
        expected = (parts["ce"] + self.cfg["loss"]["lambda_mse"] * parts["mse"]
                    + self.cfg["loss"]["lambda_contrastive"] * parts["contrastive"])
        torch.testing.assert_close(parts["total"], expected)


if __name__ == "__main__":
    unittest.main()
