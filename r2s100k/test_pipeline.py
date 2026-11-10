import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from .pipeline import PALETTE, decode, evaluate


class PipelineTests(unittest.TestCase):
    def test_palette_round_trip_and_unknown_color(self):
        ids = np.array([[0, 1, 2], [3, 8, 14]], dtype=np.uint8)
        rgb = PALETTE[ids]
        np.testing.assert_array_equal(decode(rgb), ids)
        rgb[0, 0] = [1, 2, 3]
        with self.assertRaisesRegex(ValueError, "Unknown GT colors"):
            decode(rgb)

    def test_perfect_prediction_and_boundary_metric(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ids = np.zeros((32, 40), dtype=np.uint8)
            ids[8:24, 9:30] = 2
            image = np.zeros((*ids.shape, 3), dtype=np.uint8)
            gt = PALETTE[ids]
            Image.fromarray(image).save(root / "sample.jpg")
            Image.fromarray(gt).save(root / "sample.jpg___fuse.png")
            pred_dir = root / "pred"
            pred_dir.mkdir()
            Image.fromarray(ids).save(pred_dir / "sample.jpg.png")
            image_manifest = root / "images.json"
            label_manifest = root / "labels.json"
            image_manifest.write_text(json.dumps([{"image_id": "sample.jpg", "image_path": str(root / "sample.jpg")}]))
            label_manifest.write_text(json.dumps([{"image_id": "sample.jpg", "gt_path": str(root / "sample.jpg___fuse.png")}]))
            evaluate(type("Args", (), {"image_manifest": image_manifest, "label_manifest": label_manifest,
                                        "predictions": pred_dir, "output": root / "metrics"})())
            report = json.loads((root / "metrics" / "metrics.json").read_text())
            self.assertEqual(report["miou_15"], 1.0)
            self.assertEqual(report["miou_foreground14"], 1.0)
            self.assertEqual(report["boundary_f1_foreground14"], 1.0)

    def test_missing_prediction_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image_manifest = root / "images.json"
            label_manifest = root / "labels.json"
            image_manifest.write_text(json.dumps([{"image_id": "missing.jpg", "image_path": "missing.jpg"}]))
            label_manifest.write_text(json.dumps([{"image_id": "missing.jpg", "gt_path": "missing.png"}]))
            with self.assertRaisesRegex(FileNotFoundError, "Missing prediction or GT"):
                evaluate(type("Args", (), {"image_manifest": image_manifest, "label_manifest": label_manifest,
                                            "predictions": root / "pred", "output": root / "metrics"})())


if __name__ == "__main__":
    unittest.main()
