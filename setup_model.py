"""Install a small, pinned pretrained Food-101 model for subsequent offline use."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from urllib.request import urlopen

from scripts.download_foodsg import space_sources


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "data" / "pretrained_model.json"
MODEL_DIR = ROOT / "models" / "food101"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def install() -> None:
    manifest = json.loads(MANIFEST.read_text())
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    sources = space_sources(MODEL_DIR)
    for label, path in sources:
        print(f"{label}: {shutil.disk_usage(path).free / 1024**3:.2f} GiB free", flush=True)
    for name, metadata in manifest["artifacts"].items():
        target = MODEL_DIR / name
        if target.is_file() and target.stat().st_size == metadata["size"] and file_hash(target) == metadata["sha256"]:
            print(f"Already verified: {name}", flush=True)
            continue
        # Keep the user's 5 GiB reserve, plus room for the entire artifact and buffering.
        minimum = 5 * 1024**3 + metadata["size"] + 64 * 1024**2
        if min(shutil.disk_usage(path).free for _, path in sources) < minimum:
            raise RuntimeError("Not enough physical disk space to preserve the 5 GiB reserve.")
        url = f"https://huggingface.co/{manifest['repo_id']}/resolve/{manifest['revision']}/{name}"
        temporary = target.with_suffix(target.suffix + ".part")
        count = 0
        print(f"Downloading {name} ({metadata['size'] / 10**6:.2f} MB)…", flush=True)
        try:
            with urlopen(url, timeout=60) as response, temporary.open("wb") as output:
                while chunk := response.read(1024**2):
                    count += len(chunk)
                    if count > metadata["size"]:
                        raise RuntimeError(f"Unexpected response size for {name}.")
                    if min(shutil.disk_usage(path).free for _, path in sources) <= 5 * 1024**3 + 64 * 1024**2 + len(chunk):
                        raise RuntimeError("Download stopped to preserve the physical disk reserve.")
                    output.write(chunk)
            if count != metadata["size"] or file_hash(temporary) != metadata["sha256"]:
                raise RuntimeError(f"Integrity check failed for {name}.")
            temporary.replace(target)
            print(f"SHA-256 verified: {name}", flush=True)
        finally:
            temporary.unlink(missing_ok=True)
    print("Pretrained Food-101 model installed. Run: python -m streamlit run app.py", flush=True)


if __name__ == "__main__":
    try:
        install()
    except Exception as exc:
        raise SystemExit(f"Model setup failed: {exc}") from exc
