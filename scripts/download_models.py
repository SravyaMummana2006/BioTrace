#!/usr/bin/env python3
"""Download and verify the model assets required by the BIOTrace AI engine.

Usage (from the repository root):
    python scripts/download_models.py            # download missing assets
    python scripts/download_models.py --check    # report status only
    python scripts/download_models.py --force    # re-download the landmarker
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "ai-engine" / "models"

FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
FACE_LANDMARKER_PATH = MODELS_DIR / "mediapipe" / "face_landmarker.task"
FACE_LANDMARKER_SHA256 = (
    "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
)

# Committed model artifacts and their expected digests.
REQUIRED_MODELS = {
    MODELS_DIR / "baseline" / "best_model.pth": (
        "332846ff1479d1f6cdb1313e3247e484ec4a832d5e7c2146fb3a2e0719f29cbc"
    ),
    MODELS_DIR / "physiological" / "best_model.joblib": (
        "0a7466d8efbac00ca70128eb058a420f810601b1108fabd30212ec97b0cbd18b"
    ),
    MODELS_DIR / "physiological" / "preprocessor.joblib": (
        "a669f5a14a207ab10748c997e56f9a1570a95f532d472823b896ef6ab1fd5788"
    ),
    MODELS_DIR / "fusion" / "phase10_fusion.joblib": (
        "a6fbd3596756af8534c7232c719096a31f43e64d882b3d404a043d83ce7c66f1"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def report_required_models() -> bool:
    ok = True
    for path, expected in REQUIRED_MODELS.items():
        if not path.exists():
            print(f"MISSING  {path.relative_to(REPO_ROOT)}")
            ok = False
            continue
        actual = sha256(path)
        if actual != expected:
            print(
                f"DIGEST MISMATCH  {path.relative_to(REPO_ROOT)}\n"
                f"  expected {expected}\n  actual   {actual}"
            )
            ok = False
        else:
            print(f"OK       {path.relative_to(REPO_ROOT)}")
    return ok


def download_face_landmarker(force: bool = False) -> bool:
    if FACE_LANDMARKER_PATH.exists() and not force:
        actual = sha256(FACE_LANDMARKER_PATH)
        if actual == FACE_LANDMARKER_SHA256:
            print(f"OK       {FACE_LANDMARKER_PATH.relative_to(REPO_ROOT)}")
            return True
        print(
            f"DIGEST MISMATCH  {FACE_LANDMARKER_PATH.relative_to(REPO_ROOT)} "
            "(re-downloading)"
        )

    FACE_LANDMARKER_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"DOWNLOAD {FACE_LANDMARKER_URL}")
    try:
        with tempfile.NamedTemporaryFile(
            dir=FACE_LANDMARKER_PATH.parent, delete=False, suffix=".task.part"
        ) as tmp:
            tmp_path = Path(tmp.name)
            with urllib.request.urlopen(FACE_LANDMARKER_URL, timeout=120) as response:
                while chunk := response.read(1024 * 1024):
                    tmp.write(chunk)
        actual = sha256(tmp_path)
        if actual != FACE_LANDMARKER_SHA256:
            tmp_path.unlink(missing_ok=True)
            print(
                "ERROR: downloaded face landmarker digest mismatch\n"
                f"  expected {FACE_LANDMARKER_SHA256}\n  actual   {actual}"
            )
            return False
        tmp_path.replace(FACE_LANDMARKER_PATH)
        print(f"OK       {FACE_LANDMARKER_PATH.relative_to(REPO_ROOT)}")
        return True
    except Exception as exc:  # pragma: no cover - network dependent
        print(f"ERROR: unable to download face landmarker: {exc}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="Report model status without downloading"
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-download the face landmarker"
    )
    args = parser.parse_args()

    print(f"BIOTrace model assets in {MODELS_DIR}")
    ok = report_required_models()

    if args.check:
        if FACE_LANDMARKER_PATH.exists():
            actual = sha256(FACE_LANDMARKER_PATH)
            if actual == FACE_LANDMARKER_SHA256:
                print(f"OK       {FACE_LANDMARKER_PATH.relative_to(REPO_ROOT)}")
            else:
                print(f"DIGEST MISMATCH  {FACE_LANDMARKER_PATH.relative_to(REPO_ROOT)}")
                ok = False
        else:
            print(f"MISSING  {FACE_LANDMARKER_PATH.relative_to(REPO_ROOT)}")
            ok = False
    else:
        ok = download_face_landmarker(force=args.force) and ok

    if not ok:
        print(
            "\nModel setup incomplete. Re-run after resolving the failures above.",
            file=sys.stderr,
        )
        return 1
    print("\nAll required model assets are present and verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
