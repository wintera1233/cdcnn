import json
import unittest
from pathlib import Path

import numpy as np

from src.cdcnn_ablation import (
    AUGMENTED_STAGES,
    SCALE_EXPERIMENT_STAGES,
    SCALE_IMPLEMENTATION_VERSION,
    _training_arrays,
    generate_a1_views,
    perturbation_scale_for,
    validate_config,
)


ROOT = Path(__file__).resolve().parents[1]


class AugmentationScaleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = json.loads(
            (ROOT / "configs/cdcnn_v6_7_augmentation_scale.json").read_text(encoding="utf-8"))
        cls.x = np.random.default_rng(3).normal(size=(8, 128))
        cls.y = np.array([1, 1, 2, 2, 3, 3, 4, 4])

    def _views(self, scale):
        return generate_a1_views(
            self.x, self.y, np.arange(8), np.arange(1, 9), 11, "unit", scale)

    def test_config_validates_and_pins_scales(self):
        validate_config(self.cfg)
        self.assertEqual(self.cfg["implementation_version"], SCALE_IMPLEMENTATION_VERSION)
        self.assertEqual(self.cfg["a1_perturbation_scales"],
                         {s: perturbation_scale_for(s) for s in SCALE_EXPERIMENT_STAGES})
        self.assertEqual(self.cfg["augmentation"]["perturbation_scale"], 1.0)

    def test_canonical_stages_keep_scale_one(self):
        for stage in ("A1", "A1-stab-PS", "A2-semantic", "A3", "A2-lit-PS"):
            self.assertEqual(perturbation_scale_for(stage), 1.0, stage)

    def test_default_call_is_identical_to_explicit_scale_one(self):
        default, _ = generate_a1_views(
            self.x, self.y, np.arange(8), np.arange(1, 9), 11, "unit")
        explicit, _ = self._views(1.0)
        np.testing.assert_array_equal(default, explicit)

    def test_scale_multiplies_the_noise_exactly(self):
        base, _ = self._views(1.0)
        for scale in (0.5, 0.2, 0.05):
            scaled, provenance = self._views(scale)
            np.testing.assert_allclose(
                scaled - self.x, scale * (base - self.x), rtol=1e-12, atol=1e-12)
            self.assertTrue((provenance.perturbation_scale == scale).all())

    def test_scale_stages_are_augmented_and_use_their_scale(self):
        for stage in SCALE_EXPERIMENT_STAGES:
            self.assertIn(stage, AUGMENTED_STAGES)
            train_x, train_y, provenance = _training_arrays(
                stage, self.x, self.y, np.arange(8), np.arange(1, 9), 11, "unit")
            self.assertEqual(len(train_x), 16)
            self.assertEqual(len(train_y), 16)
            self.assertTrue(
                (provenance.perturbation_scale == perturbation_scale_for(stage)).all(), stage)


if __name__ == "__main__":
    unittest.main()
