#!/usr/bin/env python3
import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()


def find_project_root(path: Path) -> Path:
    for parent in path.parents:
        if all((parent / name).is_dir() for name in ("data", "models", "configs")):
            return parent
    return path.parents[1]


ROOT = find_project_root(SCRIPT_PATH)
DEFAULT_SOURCES = ROOT / "configs" / "model_sources.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, destination: Path, timeout: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_file = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(url, timeout=timeout) as response, temp_file.open("wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    temp_file.replace(destination)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Detectron2 model weights.")
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--force", action="store_true", help="Redownload even if file exists.")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    sources_path = args.sources
    if not sources_path.is_absolute():
        sources_path = (ROOT / sources_path).resolve()

    if not sources_path.exists():
        print(f"Sources file not found: {sources_path}")
        return 1

    with sources_path.open("r", encoding="utf-8") as file_handle:
        payload = json.load(file_handle)
    models = payload.get("models", [])
    if not models:
        print(f"No model entries found in: {sources_path}")
        return 1

    failures = []
    missing_urls = []

    for item in models:
        name = item.get("name", "unnamed")
        rel_path = item.get("file_path", "").strip()
        url = item.get("url", "").strip()
        expected_hash = item.get("sha256", "").strip().lower()

        if not rel_path:
            failures.append(f"[{name}] missing file_path in {sources_path}")
            continue
        if not url:
            missing_urls.append(f"[{name}] set url for {rel_path}")
            continue

        output_path = (ROOT / rel_path).resolve()
        if ROOT not in output_path.parents and output_path != ROOT:
            failures.append(f"[{name}] file_path escapes project root: {rel_path}")
            continue

        if output_path.exists() and not args.force:
            if expected_hash:
                current_hash = sha256_file(output_path)
                if current_hash == expected_hash:
                    print(f"[skip] {name}: already present and hash is valid")
                    continue
                print(f"[warn] {name}: existing file hash mismatch, redownloading")
            else:
                print(f"[skip] {name}: already present (no hash configured)")
                continue

        print(f"[download] {name} -> {output_path}")
        try:
            download_file(url, output_path, args.timeout)
        except Exception as exc:
            failures.append(f"[{name}] download failed: {exc}")
            continue

        if expected_hash:
            got_hash = sha256_file(output_path)
            if got_hash != expected_hash:
                output_path.unlink(missing_ok=True)
                failures.append(
                    f"[{name}] sha256 mismatch (expected {expected_hash}, got {got_hash})"
                )
                continue
            print(f"[ok] {name}: hash verified")
        else:
            print(f"[ok] {name}: downloaded (sha256 not configured)")

    if missing_urls:
        print("\nMissing model URLs:")
        for message in missing_urls:
            print(f"- {message}")
        print(f"Update: {sources_path}")

    if failures:
        print("\nErrors:")
        for message in failures:
            print(f"- {message}")

    return 1 if failures or missing_urls else 0


if __name__ == "__main__":
    sys.exit(main())
