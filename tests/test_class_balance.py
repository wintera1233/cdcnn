import collections
import json
import unittest
from pathlib import Path

import numpy as np
import torch

from src.cdcnn_ablation import (
    BALANCED_SAMPLING_STAGES,
    BALANCE_EXPERIMENT_STAGES,
    BALANCE_IMPLEMENTATION_VERSION,
    CLASS_WEIGHTED_STAGES,
    ProtocolError,
    _make_loader,
    class_weights_for,
    validate_config,
)


ROOT = Path(__file__).resolve().parents[1]
BATCH1 = np.repeat(np.arange(1, 7), [90, 98, 83, 30, 70, 74])  # the real Batch 1 balance
CPU = torch.device("cpu")


class ClassBalanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = json.loads(
            (ROOT / "configs/cdcnn_v6_11_class_balance.json").read_text(encoding="utf-8"))

    def test_config_validates_and_pins_the_methods(self):
        validate_config(self.cfg)
        self.assertEqual(self.cfg["implementation_version"], BALANCE_IMPLEMENTATION_VERSION)
        self.assertEqual(self.cfg["class_balance"],
                         {"B0-stab-PS": "none", "B0-wce-PS": "weighted_cross_entropy",
                          "B0-bal-PS": "balanced_sampling"})
        self.assertEqual(self.cfg["training"],
                         json.loads((ROOT / "configs/cdcnn_v6.json").read_text())["training"])

    def test_weights_are_inverse_frequency_with_mean_one(self):
        w = class_weights_for("B0-wce-PS", BATCH1, CPU)
        counts = np.array([90, 98, 83, 30, 70, 74], dtype=float)
        expected = counts.sum() / (6 * counts)
        expected = expected / expected.mean()
        np.testing.assert_allclose(w.numpy(), expected, rtol=1e-6)
        self.assertAlmostEqual(float(w.mean()), 1.0, places=6)
        self.assertEqual(int(w.argmax()), 3, "Ethylene, the smallest class, must weigh most")

    def test_other_stages_keep_unweighted_cross_entropy(self):
        for stage in ("B0-stab-PS", "B0-bal-PS", "A3", "A3-lit-aln-PS"):
            self.assertIsNone(class_weights_for(stage, BATCH1, CPU), stage)

    def test_a_missing_gas_is_rejected(self):
        with self.assertRaisesRegex(ProtocolError, "every gas"):
            class_weights_for("B0-wce-PS", BATCH1[BATCH1 != 4], CPU)

    def test_balanced_sampling_equalises_classes_and_keeps_epoch_size(self):
        x = np.random.default_rng(0).normal(size=(len(BATCH1), 128))
        loader = _make_loader(x, BATCH1, 64, 42, True, balanced=True)
        seen = collections.Counter(int(v) + 1 for _, labels in loader for v in labels)
        self.assertEqual(sum(seen.values()), len(BATCH1))
        share = np.array([seen[g] for g in range(1, 7)]) / len(BATCH1)
        # every gas within a few points of 1/6, against 0.07-0.22 unbalanced
        self.assertLess(float(np.abs(share - 1 / 6).max()), 0.06)

    def test_unbalanced_loader_is_untouched(self):
        x = np.random.default_rng(1).normal(size=(len(BATCH1), 128))
        plain = _make_loader(x, BATCH1, 64, 42, True)
        seen = collections.Counter(int(v) + 1 for _, labels in plain for v in labels)
        self.assertEqual([seen[g] for g in range(1, 7)], [90, 98, 83, 30, 70, 74])

    def test_balanced_stages_are_declared_consistently(self):
        self.assertEqual(set(BALANCE_EXPERIMENT_STAGES),
                         {"B0-stab-PS"} | set(CLASS_WEIGHTED_STAGES) | set(BALANCED_SAMPLING_STAGES))


if __name__ == "__main__":
    unittest.main()
