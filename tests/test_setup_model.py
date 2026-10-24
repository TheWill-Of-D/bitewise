"""Pretrained artifact integrity, caching, and disk budget checks."""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import setup_model


class SetupChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.folder = self.root / "model"
        self.folder.mkdir()
        self.payload = b"verified model content"
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(json.dumps({
            "repo_id": "example/food", "revision": "pinned-revision",
            "artifacts": {"weights.bin": {"size": len(self.payload), "sha256": hashlib.sha256(self.payload).hexdigest()}},
        }))
        self.stack = contextlib.ExitStack()
        self.stack.enter_context(patch.object(setup_model, "MODEL_DIR", self.folder))
        self.stack.enter_context(patch.object(setup_model, "MANIFEST", self.manifest))
        self.stack.enter_context(patch.object(setup_model, "space_sources", return_value=[("Disk", self.root)]))
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))

    def tearDown(self):
        self.stack.close()
        self.tmp.cleanup()

    def test_verified_files_are_reused_without_network_or_extra_space(self):
        (self.folder / "weights.bin").write_bytes(self.payload)
        with patch.object(setup_model.shutil, "disk_usage", return_value=SimpleNamespace(free=4 * 1024**3)), patch.object(setup_model, "urlopen") as network:
            setup_model.install()
        network.assert_not_called()

    def test_mismatch_does_not_replace_existing_file(self):
        target = self.folder / "weights.bin"
        target.write_bytes(b"old checkpoint")
        with patch.object(setup_model.shutil, "disk_usage", return_value=SimpleNamespace(free=9 * 1024**3)), patch.object(setup_model, "urlopen", return_value=io.BytesIO(b"x" * len(self.payload))):
            with self.assertRaisesRegex(RuntimeError, "Integrity check failed"):
                setup_model.install()
        self.assertEqual(target.read_bytes(), b"old checkpoint")
        self.assertFalse((self.folder / "weights.bin.part").exists())

    def test_disk_reserve_blocks_download(self):
        with patch.object(setup_model.shutil, "disk_usage", return_value=SimpleNamespace(free=4 * 1024**3)), patch.object(setup_model, "urlopen") as network:
            with self.assertRaisesRegex(RuntimeError, "physical disk space"):
                setup_model.install()
        network.assert_not_called()

    def test_download_is_verified_before_installation(self):
        with patch.object(setup_model.shutil, "disk_usage", return_value=SimpleNamespace(free=9 * 1024**3)), patch.object(setup_model, "urlopen", return_value=io.BytesIO(self.payload)):
            setup_model.install()
        self.assertEqual((self.folder / "weights.bin").read_bytes(), self.payload)
        self.assertFalse((self.folder / "weights.bin.part").exists())
