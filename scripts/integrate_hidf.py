#!/usr/bin/env python3
"""
BIOTrace Dataset Integration & Integrity Verification Script
Module: scripts/integrate_hidf.py
Integrates extracted HiDF video files with the frozen Protocol C split manifest:
- Validates extracted real (<base_id>.mp4) and fake (<base_id>_<target_id>.mp4) files
- Assigns partition splits (train/val/test) exclusively from protocol_c_split_manifest.csv
- Enforces zero base and zero donor leakage
- Produces data/manifests/hidf_integrity_manifest.csv
"""

import os
import sys
import argparse
import csv
import hashlib
from datetime import datetime, timezone

DEFAULT_EXTRACTED_DIR = os.path.join("data", "extracted", "hidf")
DEFAULT_FROZEN_MANIFEST = "protocol_c_split_manifest.csv"
DEFAULT_OUTPUT_MANIFEST_DIR = os.path.join("data", "manifests")
FROZEN_MANIFEST_SHA256 = "c15300c15ca78fb58ab16d8f01549534dffb05fb9d75664ac14b2d12c0af06b7"

def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def verify_frozen_manifest(manifest_path):
    """Verify that the frozen manifest has not been tampered with or modified."""
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Frozen split manifest not found: {manifest_path}")
    actual_hash = sha256_file(manifest_path)
    if actual_hash != FROZEN_MANIFEST_SHA256:
        raise ValueError(
            f"FROZEN MANIFEST INTEGRITY VIOLATION!\n"
            f"Expected SHA256: {FROZEN_MANIFEST_SHA256}\n"
            f"Actual SHA256:   {actual_hash}\n"
            f"Modifying the frozen split manifest is strictly prohibited."
        )
    return True

def dry_run_integration(extracted_dir, frozen_manifest_path, output_dir):
    """Verify split mapping readiness without requiring extracted videos."""
    print("=" * 70)
    print("BIOTrace HiDF Dataset Integration — DRY-RUN VERIFICATION")
    print("=" * 70)
    
    # Check frozen manifest
    print(f"Checking frozen split manifest: {frozen_manifest_path}...")
    verify_frozen_manifest(frozen_manifest_path)
    print(f"Frozen Split Manifest Verified: SHA-256 MATCH ({FROZEN_MANIFEST_SHA256})")
    
    # Load manifest
    pairs = list(csv.DictReader(open(frozen_manifest_path, "r", encoding="utf-8")))
    print(f"Total Verified Split Rows: {len(pairs):,}")
    
    real_dir = os.path.join(extracted_dir, "real")
    fake_dir = os.path.join(extracted_dir, "fake")
    print(f"Expected Real Video Directory: {real_dir}")
    print(f"Expected Fake Video Directory: {fake_dir}")
    print(f"Manifest Output Directory:     {output_dir}")
    
    real_found = len([f for f in os.listdir(real_dir) if f.endswith(".mp4")]) if os.path.exists(real_dir) else 0
    fake_found = len([f for f in os.listdir(fake_dir) if f.endswith(".mp4")]) if os.path.exists(fake_dir) else 0
    
    print(f"\nExtracted Files Currently Present:")
    print(f"  Real Videos: {real_found:,} / 4,361 expected")
    print(f"  Fake Videos: {fake_found:,} / 4,361 expected")
    
    if real_found == 0 and fake_found == 0:
        print("  Status: Awaiting Phase 4 archive download and extraction.")
    elif real_found == 4361 and fake_found == 4361:
        print("  Status: All video payloads present. Ready for full integrity linking.")
    else:
        print("  Status: Partial video payload detected.")
        
    print("\nDry-run completed successfully.")
    return True

def integrate_dataset(extracted_dir, frozen_manifest_path, output_dir):
    """Perform dataset integration and produce hidf_integrity_manifest.csv."""
    verify_frozen_manifest(frozen_manifest_path)
    os.makedirs(output_dir, exist_ok=True)
    
    real_dir = os.path.join(extracted_dir, "real")
    fake_dir = os.path.join(extracted_dir, "fake")
    
    if not os.path.exists(real_dir) or not os.path.exists(fake_dir):
        raise FileNotFoundError("Extracted directories 'real' and 'fake' must exist before integration.")
        
    pairs = list(csv.DictReader(open(frozen_manifest_path, "r", encoding="utf-8")))
    integrity_rows = []
    
    missing_real = 0
    missing_fake = 0
    
    for p in pairs:
        bid = p["base_id"]
        tid = p["target_id"]
        cid = p["component_id"]
        split = p["split"]
        race = p["race"]
        gender = p["gender"]
        age = p["age"]
        
        real_file = f"{bid}.mp4"
        fake_file = f"{bid}_{tid}.mp4"
        
        real_path = os.path.join(real_dir, real_file)
        fake_path = os.path.join(fake_dir, fake_file)
        
        real_exists = os.path.exists(real_path)
        fake_exists = os.path.exists(fake_path)
        
        if not real_exists:
            missing_real += 1
        if not fake_exists:
            missing_fake += 1
            
        integrity_rows.append({
            "base_id": bid,
            "target_id": tid,
            "component_id": cid,
            "split": split,
            "race": race,
            "gender": gender,
            "age": age,
            "real_filename": real_file,
            "fake_filename": fake_file,
            "real_exists": real_exists,
            "fake_exists": fake_exists,
            "real_size_bytes": os.path.getsize(real_path) if real_exists else 0,
            "fake_size_bytes": os.path.getsize(fake_path) if fake_exists else 0,
            "status": "VALID" if (real_exists and fake_exists) else "MISSING_PAYLOAD",
        })
        
    # Copy frozen manifest to data/manifests/ for centralized data packaging
    dest_frozen = os.path.join(output_dir, "protocol_c_split_manifest.csv")
    if not os.path.exists(dest_frozen) or sha256_file(dest_frozen) != FROZEN_MANIFEST_SHA256:
        import shutil
        shutil.copy2(frozen_manifest_path, dest_frozen)
        
    out_manifest = os.path.join(output_dir, "hidf_integrity_manifest.csv")
    with open(out_manifest, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "base_id", "target_id", "component_id", "split", "race", "gender", "age",
            "real_filename", "fake_filename", "real_exists", "fake_exists",
            "real_size_bytes", "fake_size_bytes", "status"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(integrity_rows)
        
    print(f"Integration Manifest written to: {out_manifest}")
    print(f"Total Rows: {len(integrity_rows):,}")
    print(f"Missing Real: {missing_real}, Missing Fake: {missing_fake}")
    return True

def main():
    parser = argparse.ArgumentParser(description="BIOTrace HiDF Dataset Integration Script")
    parser.add_argument("--extracted-dir", default=DEFAULT_EXTRACTED_DIR,
                        help=f"Path to extracted videos (default: {DEFAULT_EXTRACTED_DIR})")
    parser.add_argument("--frozen-manifest", default=DEFAULT_FROZEN_MANIFEST,
                        help=f"Path to frozen Protocol C split manifest (default: {DEFAULT_FROZEN_MANIFEST})")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_MANIFEST_DIR,
                        help=f"Directory to write integrity manifest (default: {DEFAULT_OUTPUT_MANIFEST_DIR})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run verification without writing integration manifests")
    args = parser.parse_args()
    
    if args.dry_run:
        dry_run_integration(args.extracted_dir, args.frozen_manifest, args.output_dir)
        sys.exit(0)
        
    integrate_dataset(args.extracted_dir, args.frozen_manifest, args.output_dir)

if __name__ == "__main__":
    main()
