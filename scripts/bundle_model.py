"""Copies the RE-USE model pack into build/REUSEModel for the App Store build.

The App Store build ships the model instead of downloading it. The pack is
taken from the copy the app or `voicecleaner download` installed, and every
file is checked against ModelCatalog.json.

usage: python3 scripts/bundle_model.py
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
catalog = json.loads((ROOT / "Sources/CleanerCore/Resources/ModelCatalog.json").read_text())
models = Path.home() / "Library/Application Support/VoiceCleaner/REUSEModels"
destination = ROOT / "build/REUSEModel"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def matches(folder):
    return all((folder / f["path"]).is_file() and sha256(folder / f["path"]) == f["sha256"]
               for f in catalog["files"])


source = next((folder for folder in sorted(models.glob("*")) if folder.is_dir() and matches(folder)), None)
if source is None:
    sys.exit(f"No complete model pack in {models}. Run the app once, or `voicecleaner download`.")
if destination.exists():
    shutil.rmtree(destination)
for f in catalog["files"]:
    target = destination / f["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / f["path"], target)
assert matches(destination)
size = sum(f["bytes"] for f in catalog["files"])
print(f"{destination}  ({len(catalog['files'])} files, {size / 1e6:.1f} MB)")
