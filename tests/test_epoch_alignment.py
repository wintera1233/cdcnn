import json
import unittest
from pathlib import Path

import numpy as np

from src.cdcnn_ablation import (
    ALIGNED_EXPERIMENT_STAGES,
    ALIGNED_IMPLEMENTATION_VERSION,
    AUGMENTED_STAGES,
    EPOCH_ALIGNED_STAGES,
    _make_loader,
    validate_config,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROWS = 445


def _pool():
    x = np.random.default_rng(4).normal(size=(2 * SOURCE_ROWS, 128))
    y = np.tile(np.repeat(np.arange(1, 7), 75)[:SOURCE_ROWS], 2)
    return x, y


class EpochAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = json.loads(
            (ROOT / "configs/cdcnn_v6_9_epoch_aligned.json").read_text(encoding="utf-8"))
        cls.canonical = json.loads((ROOT / "configs/cdcnn_v6.json").read_text(encoding="utf-8"))

    def test_config_validates_and_keeps_the_training_protocol(self):
        validate_config(self.cfg)
        self.assertEqual(self.cfg["implementation_version"], ALIGNED_IMPLEMENTATION_VERSION)
        # The schedule itself is untouched: only the rows drawn per epoch change.
        self.assertEqual(self.cfg["training"], self.canonical["training"])
        self.assertEqual(self.cfg["epoch_alignment"],
                         {s: s in EPOCH_ALIGNED_STAGES for s in ALIGNED_EXPERIMENT_STAGES})

    def test_aligned_stages_are_augmented(self):
        for stage in EPOCH_ALIGNED_STAGES:
            self.assertIn(stage, AUGMENTED_STAGES, stage)

    def test_aligned_epoch_matches_an_unaugmented_epoch(self):
        x, y = _pool()
        aligned = _make_loader(x, y, 64, 42, True, samples_per_epoch=SOURCE_ROWS)
        unaugmented = _make_loader(x[:SOURCE_ROWS], y[:SOURCE_ROWS], 64, 42, True)
        self.assertEqual(len(aligned), len(unaugmented))
        self.assertEqual(sum(len(b[1]) for b in aligned), SOURCE_ROWS)

    def test_unaligned_loader_is_unchanged(self):
        x, y = _pool()
        loader = _make_loader(x, y, 64, 42, True)
        self.assertEqual(sum(len(b[1]) for b in loader), 2 * SOURCE_ROWS)
        self.assertEqual(len(loader), 14)

    def test_epoch_subsets_differ_but_are_reproducible(self):
        x, y = _pool()
        def epochs(seed, count=3):
            loader = _make_loader(x, y, 64, seed, True, samples_per_epoch=SOURCE_ROWS)
            return [np.concatenate([b[0].numpy().reshape(len(b[0]), -1)[:, 0] for b in loader])
                    for _ in range(count)]
        first, again, other = epochs(42), epochs(42), epochs(7)
        for a, b in zip(first, again):
            np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(first[0], first[1]))
        self.assertFalse(np.array_equal(first[0], other[0]))
        self.assertEqual(len(set(first[0].tolist())), SOURCE_ROWS)

    def test_pool_is_covered_across_the_full_schedule(self):
        x, y = _pool()
        loader = _make_loader(x, y, 64, 42, True, samples_per_epoch=SOURCE_ROWS)
        seen = set()
        for _ in range(self.cfg["training"]["epochs"]):
            for batch, _labels in loader:
                seen.update(batch.numpy().reshape(len(batch), -1)[:, 0].tolist())
        self.assertEqual(len(seen), 2 * SOURCE_ROWS)


if __name__ == "__main__":
    unittest.main()
