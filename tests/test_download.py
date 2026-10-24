"""Protect the physical Windows host disk, not only the WSL virtual filesystem."""

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import download_foodsg as downloader


class Response(io.BytesIO):
    status = 200
    headers = {}


class DownloadSpaceChecks(unittest.TestCase):
    def test_wsl_registry_identifies_physical_drive(self):
        registry = SimpleNamespace(stdout=json.dumps([{"DistributionName": "Ubuntu", "BasePath": "C:\\Users\\person\\wsl\\Ubuntu"}]))
        with patch.dict(os.environ, {"WSL_DISTRO_NAME": "Ubuntu"}), patch.object(downloader.subprocess, "run", return_value=registry), patch.object(Path, "is_mount", return_value=True):
            sources = downloader.space_sources(Path("data"))
        self.assertEqual(sources[-1][1], Path("/mnt/c"))

    def test_detection_failure_does_not_silently_use_virtual_free_space(self):
        with patch.dict(os.environ, {"WSL_DISTRO_NAME": "Ubuntu"}), patch.object(downloader.subprocess, "run", side_effect=subprocess.SubprocessError("PowerShell unavailable")):
            with self.assertRaises(subprocess.SubprocessError):
                downloader.space_sources(Path("data"))

    def test_host_below_reserve_prevents_network_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "dataset.zip"
            host = Path(tmp) / "host"
            def usage(path):
                return SimpleNamespace(free=(4 if path == host else 883) * 1024**3)
            with patch.object(sys, "argv", ["download", "--output", str(output), "--expected-md5", "unused"]), patch.dict(os.environ, {"BITEWISE_DATASET_URL": "https://example.invalid/dataset"}), patch.object(downloader, "space_sources", return_value=[("WSL", output.parent), ("Windows", host)]), patch.object(downloader.shutil, "disk_usage", side_effect=usage), patch.object(downloader, "urlopen") as network, contextlib.redirect_stdout(io.StringIO()):
                result = downloader.main()
            self.assertEqual(result, 2)
            network.assert_not_called()
            self.assertFalse(output.with_suffix(".zip.part").exists())

    def test_host_reserve_stops_an_active_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "dataset.zip"
            host = Path(tmp) / "host"
            read_count = [0]
            class StreamingResponse(Response):
                def read1(self, size):
                    read_count[0] += 1
                    return super().read1(size)
            def usage(path):
                free = (4 if read_count[0] >= 2 else 9) if path == host else 883
                return SimpleNamespace(free=free * 1024**3)
            with patch.object(sys, "argv", ["download", "--output", str(output), "--expected-md5", "unused"]), patch.dict(os.environ, {"BITEWISE_DATASET_URL": "https://example.invalid/dataset"}), patch.object(downloader, "space_sources", return_value=[("WSL", output.parent), ("Windows", host)]), patch.object(downloader.shutil, "disk_usage", side_effect=usage), patch.object(downloader, "urlopen", return_value=StreamingResponse(b"x" * (3 * 1024**2))), contextlib.redirect_stdout(io.StringIO()):
                result = downloader.main()
            self.assertEqual(result, 2)
            self.assertFalse(output.exists())
            self.assertEqual(output.with_suffix(".zip.part").stat().st_size, 1024**2)

    def test_successful_download_requires_matching_checksum(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "dataset.zip"
            content = b"sample archive bytes"
            with patch.object(sys, "argv", ["download", "--output", str(output), "--expected-md5", hashlib.md5(content).hexdigest()]), patch.dict(os.environ, {"BITEWISE_DATASET_URL": "https://example.invalid/dataset"}), patch.object(downloader, "space_sources", return_value=[("Disk", output.parent)]), patch.object(downloader.shutil, "disk_usage", return_value=SimpleNamespace(free=9 * 1024**3)), patch.object(downloader, "urlopen", return_value=Response(content)), contextlib.redirect_stdout(io.StringIO()):
                result = downloader.main()
            self.assertEqual(result, 0)
            self.assertEqual(output.read_bytes(), content)
            self.assertFalse(output.with_suffix(".zip.part").exists())


if __name__ == "__main__":
    unittest.main()
