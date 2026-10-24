"""Optional real-browser smoke check: pip install playwright; playwright install chromium."""

import io
import argparse
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from urllib.parse import urlsplit
from urllib.request import urlopen

from PIL import Image
from playwright.sync_api import expect, sync_playwright


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-photo", type=Path, help="Optional local photo to exercise the installed pretrained model.")
    parser.add_argument("--expected-food", default="Pizza")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        url = f"http://127.0.0.1:{port}"
        env = {**os.environ, "BITEWISE_DB_PATH": str(Path(tmp) / "browser.sqlite3"), "BITEWISE_MODEL": "pretrained" if args.real_photo else "none"}
        with (Path(tmp) / "server.log").open("w+") as log:
            server = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", str(port)], cwd=ROOT, env=env, stdout=log, stderr=log)
            try:
                for _ in range(100):
                    try:
                        with urlopen(url + "/_stcore/health", timeout=1) as response:
                            if response.read() == b"ok":
                                break
                    except OSError:
                        time.sleep(.2)
                else:
                    raise RuntimeError("Test server did not start")

                with sync_playwright() as p:
                    browser = p.chromium.launch(args=["--no-sandbox"])
                    page = browser.new_page(viewport={"width": 1440, "height": 1050}, device_scale_factor=1)
                    external_requests = []
                    errors = []
                    page.on("request", lambda request: external_requests.append(request.url) if urlsplit(request.url).hostname not in ("127.0.0.1", "localhost", None) else None)
                    page.on("pageerror", lambda error: errors.append(str(error)))
                    page.goto(url)
                    expect(page.get_by_role("heading", name="BiteWise", exact=True)).to_be_visible()
                    add = page.get_by_role("button", name="Add to today's meals", exact=True)
                    expect(add).to_be_disabled()
                    page.get_by_role("combobox").click()
                    page.get_by_role("combobox").fill("Laksa")
                    page.get_by_role("option", name="Laksa", exact=True).click()
                    serving = page.get_by_role("spinbutton", name="Serving size (g)", exact=True)
                    serving.fill("200")
                    serving.press("Enter")
                    expect(page.get_by_text("Estimated for 200 g", exact=True)).to_be_visible()
                    expect(page.get_by_text("300 kcal", exact=False).first).to_be_visible()
                    add.click()
                    metric = page.get_by_test_id("stMetric").filter(has=page.get_by_text("Protein today", exact=True))
                    expect(metric).to_contain_text("12 g")
                    expect(page.get_by_role("button", name="Remove", exact=True)).to_be_visible()
                    (ROOT / "artifacts").mkdir(exist_ok=True)
                    page.wait_for_timeout(700)  # Let the progress animation settle for the preview.
                    page.screenshot(path=str(ROOT / "artifacts" / "bitewise-preview.png"), full_page=True)
                    page.reload()
                    expect(metric).to_contain_text("12 g")
                    page.get_by_role("button", name="Remove", exact=True).click()
                    expect(metric).to_contain_text("0 g")
                    image = io.BytesIO()
                    Image.new("RGB", (80, 60), "green").save(image, format="PNG")
                    if args.real_photo:
                        page.locator('input[type="file"]').set_input_files(str(args.real_photo))
                        expect(page.get_by_text(f"Suggested dish: {args.expected_food}", exact=False)).to_be_visible(timeout=30000)
                        expect(add).to_be_enabled()
                        add.click()
                        expect(page.get_by_role("button", name="Remove", exact=True)).to_be_visible()
                        expect(page.get_by_text(args.expected_food, exact=True).first).to_be_visible()
                    else:
                        page.locator('input[type="file"]').set_input_files({"name": "meal.png", "mimeType": "image/png", "buffer": image.getvalue()})
                        expect(page.get_by_text("Photo added. Recognition is not configured yet. Choose the dish below to continue.", exact=True)).to_be_visible()
                        expect(add).to_be_disabled()
                    mobile = browser.new_page(viewport={"width": 390, "height": 844})
                    mobile.goto(url)
                    expect(mobile.get_by_role("heading", name="BiteWise", exact=True)).to_be_visible()
                    expect(mobile.get_by_role("button", name="Add to today's meals", exact=True)).to_be_attached()
                    assert mobile.evaluate("document.documentElement.scrollWidth <= window.innerWidth"), "Mobile page overflows horizontally"
                    mobile.screenshot(path=str(ROOT / "artifacts" / "bitewise-mobile.png"), full_page=True)
                    assert not errors, errors
                    assert not external_requests, external_requests
                    print("Browser checks passed: load, serving preview, add, reload persistence, remove, photo upload, mobile layout, no external requests.")
                    browser.close()
            finally:
                server.terminate()
                server.wait(timeout=10)


if __name__ == "__main__":
    main()
