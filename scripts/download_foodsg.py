"""Download an approved dataset with a free-space reserve and streaming MD5 check.

The access URL is read from BITEWISE_DATASET_URL rather than stored in this file.
Partial downloads are retained on interruption; the server may not support resume.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import threading
import time
from urllib.request import Request, urlopen


class SpaceLimit(Exception):
    pass


def space_sources(destination, host_volume=None):
    """Monitor the physical VHD backing volume as well as the WSL filesystem."""
    sources = [("Destination filesystem", destination)]
    is_wsl = bool(os.environ.get("WSL_DISTRO_NAME")) or "microsoft" in platform.release().lower()
    if host_volume is None and is_wsl:
        # Query every registered distribution without interpolating user data into PowerShell.
        command = 'Get-ChildItem "HKCU:\\Software\\Microsoft\\Windows\\CurrentVersion\\Lxss" | ForEach-Object { $d = Get-ItemProperty $_.PSPath; [PSCustomObject]@{DistributionName=$d.DistributionName; BasePath=$d.BasePath} } | ConvertTo-Json -Compress'
        result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command], capture_output=True, text=True, timeout=15, check=True)
        distributions = json.loads(result.stdout)
        if isinstance(distributions, dict):
            distributions = [distributions]
        distro_name = os.environ.get("WSL_DISTRO_NAME")
        matches = [entry for entry in distributions if entry.get("DistributionName") == distro_name]
        if len(matches) != 1:
            raise ValueError("Cannot determine this WSL distribution's backing Windows volume. Supply --host-volume /mnt/<drive>.")
        drive = re.search(r"([A-Za-z]):[\\/]", matches[0]["BasePath"])
        if not drive:
            raise ValueError("Cannot map the WSL backing path to a Windows drive. Supply --host-volume /mnt/<drive>.")
        host_volume = Path("/mnt") / drive.group(1).lower()
    if host_volume is not None:
        host_volume = Path(host_volume)
        if not host_volume.is_mount():
            raise ValueError(f"Windows host volume is not a mounted filesystem: {host_volume}")
        sources.append(("Physical Windows host volume", host_volume))
    return sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/foodsg-233.zip"))
    parser.add_argument("--minimum-free-gib", type=float, default=5.0)
    parser.add_argument("--expected-md5", required=True)
    parser.add_argument("--host-volume", type=Path, help="Physical Windows volume backing WSL, e.g. /mnt/c; auto-detected when omitted.")
    args = parser.parse_args()
    url = os.environ.get("BITEWISE_DATASET_URL")
    if not url:
        parser.error("Set BITEWISE_DATASET_URL to your approved download URL.")
    if args.minimum_free_gib < 5:
        parser.error("The minimum disk reserve must be at least 5 GiB.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        sources = space_sources(args.output.parent, args.host_volume)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        parser.error(f"Cannot safely check physical free space: {exc}")
    partial = args.output.with_suffix(args.output.suffix + ".part")
    if args.output.exists() or partial.exists():
        parser.error(f"Refusing to overwrite an existing download: {args.output} or {partial}")

    reserve = int(args.minimum_free_gib * 1024**3)
    # Stop a little early to leave room for buffers and other local processes.
    buffer = 64 * 1024**2
    chunk_size = 1024**2
    stop_monitor = threading.Event()
    size = 0
    digest = hashlib.md5()
    started = last_report = time.monotonic()
    monitor_error = []

    def available_space():
        return min(shutil.disk_usage(path).free for _, path in sources)

    def on_low_space(signum, frame):
        raise SpaceLimit(monitor_error[0] if monitor_error else "Free disk space reached the protected reserve.")

    previous_handler = signal.signal(signal.SIGUSR1, on_low_space)

    def monitor():
        while not stop_monitor.wait(.5):
            try:
                low_space = available_space() <= reserve + buffer
            except OSError as exc:
                monitor_error.append(f"Cannot read free disk space: {exc}")
                low_space = True
            if low_space:
                os.kill(os.getpid(), signal.SIGUSR1)
                return

    for label, path in sources:
        print(f"{label} ({path}): {shutil.disk_usage(path).free / 1024**3:.2f} GiB free", flush=True)
    free = available_space()
    print(f"Usable free-space limit: {free / 1024**3:.2f} GiB. Protected reserve: {reserve / 1024**3:.2f} GiB (+64 MiB buffer).", flush=True)
    if free <= reserve + buffer:
        print("Not starting: insufficient free disk space.", flush=True)
        signal.signal(signal.SIGUSR1, previous_handler)
        return 2
    watchdog = threading.Thread(target=monitor, daemon=True)
    watchdog.start()
    try:
        with urlopen(Request(url, headers={"Accept-Encoding": "identity"}), timeout=120) as response:
            print(f"Connected: HTTP {response.status}; reported size: {response.headers.get('Content-Length', 'unknown')}", flush=True)
            with partial.open("xb") as destination:
                while True:
                    if available_space() <= reserve + buffer + chunk_size:
                        raise SpaceLimit("Stopping before the next chunk would consume the disk reserve.")
                    chunk = response.read1(chunk_size)
                    if not chunk:
                        break
                    if available_space() <= reserve + buffer + len(chunk):
                        raise SpaceLimit("Stopping before writing the next chunk to preserve disk space.")
                    destination.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
                    now = time.monotonic()
                    if now - last_report >= 10:
                        free = available_space()
                        print(f"Downloaded {size / 1024**3:.3f} GiB ({size / 10**9:.3f} GB); average {size / 1024**2 / (now-started):.2f} MiB/s; available {free / 1024**3:.2f} GiB", flush=True)
                        last_report = now
        actual_md5 = digest.hexdigest()
        print(f"Download finished: {size:,} bytes / {size / 10**9:.3f} GB / {size / 1024**3:.3f} GiB", flush=True)
        print(f"MD5: {actual_md5}", flush=True)
        if actual_md5.lower() != args.expected_md5.lower():
            print(f"Checksum does NOT match. Unverified file retained at {partial}", flush=True)
            return 3
        partial.rename(args.output)
        print(f"MD5 matches. Saved to {args.output.resolve()}", flush=True)
        print(f"Remaining free space (limiting volume): {available_space() / 1024**3:.2f} GiB", flush=True)
        return 0
    except (KeyboardInterrupt, Exception) as exc:
        print(f"Download stopped: {type(exc).__name__}: {exc}", flush=True)
        try:
            free_label = f"{available_space() / 1024**3:.2f} GiB"
        except OSError:
            free_label = "unavailable"
        print(f"Retained {size:,} bytes at {partial}; free space {free_label}", flush=True)
        return 2
    finally:
        stop_monitor.set()
        watchdog.join(timeout=1)
        signal.signal(signal.SIGUSR1, previous_handler)


if __name__ == "__main__":
    raise SystemExit(main())
