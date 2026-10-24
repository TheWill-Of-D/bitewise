"""Optional model checks with synthetic images, not a food accuracy evaluation."""

import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
HAS_MODEL_DEPS = all(importlib.util.find_spec(name) for name in ("torch", "torchvision"))


@unittest.skipUnless(HAS_MODEL_DEPS, "Install optional model dependencies to check training and inference")
class ModelChecks(unittest.TestCase):
    def test_stratified_split_is_reproducible_and_disjoint(self):
        from train_model import split_indices

        targets = [0] * 20 + [1] * 20
        splits = split_indices(targets, 42)
        self.assertEqual(splits, split_indices(targets, 42))
        self.assertEqual([len(split) for split in splits], [32, 4, 4])
        self.assertEqual(set().union(*map(set, splits)), set(range(40)))
        for i, split in enumerate(splits):
            self.assertEqual(sum(targets[index] == 0 for index in split), [16, 2, 2][i])
            for other in splits[i + 1:]:
                self.assertFalse(set(split) & set(other))

    def test_macro_f1_handles_external_subset_of_classes(self):
        from train_model import accuracy_and_macro_f1

        accuracy, f1 = accuracy_and_macro_f1([0, 1, 1, 1], [0, 0, 1, 1], 3)
        self.assertAlmostEqual(accuracy, .75)
        self.assertAlmostEqual(f1, ((2 / 3) + .8) / 2)

    def test_training_checkpoint_and_inference_round_trip(self):
        import recognition

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            data = folder / "images"
            for label, color in (("Bak kut teh", "red"), ("Laksa", "green")):
                (data / label).mkdir(parents=True)
                for index in range(3):
                    Image.new("RGB", (80 + index, 60), color).save(data / label / f"{index}.png")
            output = folder / "model"
            result = subprocess.run(
                [sys.executable, str(ROOT / "train_model.py"), "--data", str(data), "--from-scratch", "--finetune-epochs", "1", "--batch-size", "2", "--output", str(output)],
                cwd=ROOT, env={**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"},
                capture_output=True, text=True, timeout=120,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(json.loads((output / "class_names.json").read_text()), ["Bak kut teh", "Laksa"])
            manifest = json.loads((output / "split_manifest.json").read_text())
            self.assertEqual([len(manifest["splits"][key]) for key in ("train", "validation", "test")], [2, 2, 2])
            self.assertTrue((output / "test_metrics.json").exists())
            image = io.BytesIO()
            Image.new("RGB", (80, 60), "green").save(image, format="PNG")
            recognition._load.cache_clear()
            try:
                with patch.object(recognition, "WEIGHTS", output / "food_classifier.pt"), patch.object(recognition, "LABELS", output / "class_names.json"):
                    predictions = recognition.predict_image(image.getvalue())
                self.assertEqual({name for name, _ in predictions}, {"Bak kut teh", "Laksa"})
                self.assertAlmostEqual(sum(score for _, score in predictions), 1.0, places=5)
                self.assertGreaterEqual(predictions[0][1], predictions[1][1])
            finally:
                recognition._load.cache_clear()

    def test_installed_pretrained_model_runs_without_downloading(self):
        import recognition

        if not (recognition.PRETRAINED_DIR / "model.safetensors").is_file():
            self.skipTest("Run python setup_model.py to test the real pretrained checkpoint")
        image = io.BytesIO()
        Image.new("RGB", (100, 80), "green").save(image, format="PNG")
        recognition._load.cache_clear()
        try:
            with patch.dict(os.environ, {"BITEWISE_MODEL": "pretrained"}), patch("urllib.request.urlopen", side_effect=AssertionError("Inference must not download")), patch("torch.hub.download_url_to_file", side_effect=AssertionError("Inference must not download")):
                predictions = recognition.predict_image(image.getvalue(), top_k=101)
            self.assertEqual(len(predictions), 101)
            self.assertIn("Pizza", [name for name, _ in predictions])
            self.assertAlmostEqual(sum(score for _, score in predictions), 1.0, places=5)
            self.assertTrue(all(0 <= score <= 1 for _, score in predictions))
        finally:
            recognition._load.cache_clear()


if __name__ == "__main__":
    unittest.main()
