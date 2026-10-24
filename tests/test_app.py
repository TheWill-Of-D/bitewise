"""Workflow regression checks. Each test uses a disposable local database."""

import io
from contextlib import closing
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


def photo_bytes(color="green"):
    buffer = io.BytesIO()
    Image.new("RGB", (80, 60), color).save(buffer, format="PNG")
    return buffer.getvalue()


class MealWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "test.sqlite3"
        self.env = patch.dict(os.environ, {"BITEWISE_DB_PATH": str(self.db)})
        self.env.start()
        self.app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=15).run()
        self.assertHealthy()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def assertHealthy(self):
        self.assertEqual(len(self.app.exception), 0, str(list(self.app.exception)))

    def button(self, label):
        return next(button for button in self.app.button if button.label == label)

    def number(self, label):
        return next(number for number in self.app.number_input if number.label == label)

    def metrics(self):
        return {metric.label: metric.value for metric in self.app.metric}

    def test_dish_requires_explicit_selection(self):
        self.assertIsNone(self.app.selectbox[0].value)
        self.assertTrue(self.button("Add to today's meals").disabled)
        self.assertEqual(self.metrics()["Meals logged"], "0")

    def test_portion_math_logging_persistence_and_removal(self):
        self.app.selectbox[0].select("Chicken rice (steamed)").run()
        self.number("Serving size (g)").set_value(200).run()
        self.button("Add to today's meals").click().run()
        self.assertHealthy()
        self.assertEqual(self.metrics()["Protein today"], "24 g")
        self.assertEqual(self.metrics()["Calories"], "370 kcal")
        self.assertEqual(self.metrics()["To your protein goal"], "54 g")
        with closing(sqlite3.connect(self.db)) as con:
            row = con.execute("SELECT food,grams,protein,carbs,fat,source FROM meals").fetchone()
        self.assertEqual(row, ("Chicken rice (steamed)", 200.0, 24.0, 46.0, 10.0, "BiteWise starter estimate"))
        self.app = AppTest.from_file(str(ROOT / "app.py")).run()
        self.assertEqual(self.metrics()["Meals logged"], "1")
        self.button("Remove").click().run()
        self.assertHealthy()
        self.assertEqual(self.metrics()["Meals logged"], "0")
        self.assertEqual(self.metrics()["Protein today"], "0 g")

    def test_profile_persists_and_recomputes_target(self):
        self.number("Weight (kg)").set_value(80.0).run()
        self.button("Save profile").click().run()
        self.app = AppTest.from_file(str(ROOT / "app.py")).run()
        self.assertHealthy()
        self.assertEqual(self.number("Weight (kg)").value, 80.0)
        self.assertEqual(self.metrics()["To your protein goal"], "96 g")

    def test_photo_without_model_allows_manual_entry(self):
        with patch("recognition.model_status", return_value=(False, "Missing")):
            self.app.file_uploader[0].set_value(("meal.png", photo_bytes(), "image/png")).run()
        self.assertHealthy()
        self.assertIsNone(self.app.selectbox[0].value)
        self.assertTrue(any("not configured" in info.value for info in self.app.info))
        self.assertEqual([p.name for p in Path(self.tmp.name).iterdir()], ["test.sqlite3"])

    def test_corrupt_upload_does_not_crash(self):
        self.app.file_uploader[0].set_value(("broken.png", b"not an image", "image/png")).run()
        self.assertHealthy()
        self.assertTrue(any("could not be opened" in warning.value for warning in self.app.warning))
        self.assertTrue(self.button("Add to today's meals").disabled)

    def test_broken_model_allows_manual_correction(self):
        with patch("recognition.model_status", return_value=(True, "Ready")), patch("recognition.predict_image", side_effect=RuntimeError("bad checkpoint")):
            self.app.file_uploader[0].set_value(("meal.png", photo_bytes(), "image/png")).run()
            self.assertHealthy()
            self.assertTrue(any("could not process" in warning.value for warning in self.app.warning))
            self.app.selectbox[0].select("Laksa").run()
            self.button("Add to today's meals").click().run()
            self.assertHealthy()
            self.assertEqual(self.metrics()["Meals logged"], "1")

    def test_low_score_requires_a_choice(self):
        with patch("recognition.model_status", return_value=(True, "Ready")), patch("recognition.predict_image", return_value=[("Laksa", .45), ("Mee rebus", .35)]):
            self.app.file_uploader[0].set_value(("meal.png", photo_bytes(), "image/png")).run()
        self.assertHealthy()
        self.assertIsNone(self.app.selectbox[0].value)
        self.assertTrue(any("Uncertain" in warning.value for warning in self.app.warning))

    def test_suggestion_can_be_corrected_and_is_not_recomputed(self):
        with patch("recognition.model_status", return_value=(True, "Ready")), patch("recognition.predict_image", return_value=[("Laksa", .8), ("Mee rebus", .2)]) as predict:
            self.app.file_uploader[0].set_value(("meal.png", photo_bytes(), "image/png")).run()
            self.assertEqual(self.app.selectbox[0].value, "Laksa")
            self.app.selectbox[0].select("Mee rebus").run()
            self.number("Serving size (g)").set_value(100).run()
            self.button("Add to today's meals").click().run()
            self.assertHealthy()
            self.assertEqual(predict.call_count, 1)
            self.assertEqual(self.app.selectbox[0].value, "Mee rebus")
            with closing(sqlite3.connect(self.db)) as con:
                self.assertEqual(con.execute("SELECT food,protein FROM meals").fetchone(), ("Mee rebus", 5.5))

    def test_unmapped_prediction_does_not_choose_an_unrelated_nutrition_entry(self):
        with patch("recognition.model_status", return_value=(True, "Ready")), patch("recognition.predict_image", return_value=[("Apple pie", .95), ("Pizza", .03)]):
            self.app.file_uploader[0].set_value(("meal.png", photo_bytes(), "image/png")).run()
        self.assertHealthy()
        self.assertIsNone(self.app.selectbox[0].value)
        self.assertTrue(self.button("Add to today's meals").disabled)
        self.assertTrue(any("no nutrition entry" in warning.value for warning in self.app.warning))

    def test_new_photo_replaces_previous_correction_with_its_own_suggestion(self):
        results = [[("Laksa", .8)], [("Pizza", .9)]]
        with patch("recognition.model_status", return_value=(True, "Ready")), patch("recognition.predict_image", side_effect=results) as predict:
            self.app.file_uploader[0].set_value(("first.png", photo_bytes("green"), "image/png")).run()
            self.app.selectbox[0].select("Mee rebus").run()
            self.app.file_uploader[0].set_value(("second.png", photo_bytes("red"), "image/png")).run()
            self.assertHealthy()
            self.assertEqual(self.app.selectbox[0].value, "Pizza")
            self.assertEqual(predict.call_count, 2)


if __name__ == "__main__":
    unittest.main()
