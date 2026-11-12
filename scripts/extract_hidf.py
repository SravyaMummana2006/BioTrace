#!/usr/bin/env python3
"""
BIOTrace Safe Dataset Extraction Script
Module: scripts/extract_hidf.py

Extracts HiDF Real-vid.zip and Fake-vid.zip safely with:
- Pre-extraction size, MD5, and ZIP integrity verification
- Strict path traversal prevention (ZipSlip defense)
- Temporary staging directory extraction preventing partial extractions
- Double verification per member (file size and CRC32)
- Manifest validation against frozen Protocol C split manifest
- Atomic directory promotion on full success
- Detailed extraction manifest generation
"""

import os
import sys
import argparse
import re
import zipfile
import zlib
import hashlib
import csv
import json
import shutil
from datetime import datetime, timezone

DEFAULT_RAW_DIR = os.path.join("data", "raw", "hidf")
DEFAULT_EXTRACTED_DIR = os.path.join("data", "extracted", "hidf")
DEFAULT_SPLIT_MANIFEST = "protocol_c_split_manifest.csv"

EXTRACTION_CONFIG = {
    "real": {
        "archive_name": "Real-vid.zip",
        "dest_subdir": "real",
        "expected_files": 4361,
        "expected_size": 842885152,
        "expected_md5": "ee53b75850a7f54b45d61a78085eda0a",
        "expected_sha256": "d7ecd565b638e699ac7ec928eae954cc206ca68837a1939587696dbee9ae560c",
        "default_manifest": "real_video_extraction_manifest.csv",
        "description": "Authentic base videos",
    },
    "fake": {
        "archive_name": "Fake-vid.zip",
        "dest_subdir": "fake",
        "expected_files": 4361,
        "expected_size": 715291222,
        "expected_md5": "708890ccaedc3a6cccee0c1d3e64a711",
        "expected_sha256": "a5833c23e0fe98285146f313bcc3ecd9f0af6b098de87aeb775af8960655cd6c",
        "default_manifest": "fake_video_extraction_manifest.csv",
        "description": "Deepfake derivative videos",
    },
}

def is_safe_path(target_dir, member_path):
    """
    Prevent ZipSlip vulnerability (path traversal attack).
    Ensures that extracting member_path remains strictly inside target_dir.

    Both POSIX and Windows separators are normalized, and absolute/drive-prefixed
    members are rejected regardless of the host platform.
    """
    normalized_member = str(member_path).replace("\\", "/")
    if normalized_member.startswith("/") or re.match(r"^[A-Za-z]:", normalized_member):
        return False

    resolved_dest = os.path.abspath(target_dir)
    resolved_target = os.path.abspath(os.path.join(target_dir, normalized_member))
    return os.path.commonpath([resolved_dest]) == os.path.commonpath([resolved_dest, resolved_target])

def compute_file_crc32(filepath):
    """Compute CRC32 checksum of an extracted local file."""
    crc = 0
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            crc = zlib.crc32(chunk, crc)
    return crc & 0xFFFFFFFF

def compute_file_md5(filepath):
    """Compute MD5 checksum of a file."""
    md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            md5.update(chunk)
    return md5.hexdigest()

def compute_file_sha256(filepath):
    """Compute SHA-256 checksum of a file."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            sha.update(chunk)
    return sha.hexdigest()

def load_split_manifest(split_manifest_path):
    """Load and index the frozen Protocol C split manifest."""
    if not os.path.exists(split_manifest_path):
        raise FileNotFoundError(f"Split manifest not found: {split_manifest_path}")
        
    records_by_base = {}
    records_by_pair = {}
    with open(split_manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records_by_base[row["base_id"]] = row
            records_by_pair[(row["base_id"], row["target_id"])] = row
    return records_by_base, records_by_pair

def dry_run_extraction(raw_dir, extract_dir, targets, split_manifest_path=DEFAULT_SPLIT_MANIFEST):
    """Inspect archives and pre-flight conditions without extracting any files."""
    print("=" * 70)
    print("BIOTrace HiDF Extraction — DRY-RUN PREPARATION")
    print("=" * 70)
    print(f"Raw Archive Directory:      {os.path.abspath(raw_dir)}")
    print(f"Target Extraction Root:     {os.path.abspath(extract_dir)}")
    print(f"Split Manifest:             {os.path.abspath(split_manifest_path)}")
    
    records_by_base, records_by_pair = load_split_manifest(split_manifest_path) if os.path.exists(split_manifest_path) else ({}, {})
    print(f"Split Records Loaded:       {len(records_by_base):,}")
    
    for t in targets:
        cfg = EXTRACTION_CONFIG[t]
        archive_path = os.path.join(raw_dir, cfg["archive_name"])
        target_dest = os.path.join(extract_dir, cfg["dest_subdir"])
        print(f"\nTarget: {t.upper()} ({cfg['description']})")
        print(f"  Archive Path:    {archive_path}")
        print(f"  Destination Dir: {target_dest}")
        print(f"  Expected Files:  {cfg['expected_files']}")
        
        if not os.path.exists(archive_path):
            print("  Archive Status:  ARCHIVE NOT FOUND")
            continue
            
        archive_size = os.path.getsize(archive_path)
        print(f"  Archive Size:    {archive_size:,} bytes (expected {cfg['expected_size']:,})")
        if archive_size != cfg["expected_size"]:
            print("  CRITICAL: Archive size does not match expected size!")
            continue
            
        # Verify ZIP headers and members
        try:
            with zipfile.ZipFile(archive_path, "r") as zf:
                infolist = zf.infolist()
                files = [m for m in infolist if not m.is_dir()]
                print(f"  ZIP Members:     {len(infolist)} total ({len(files)} files)")
                
                # Check ZipSlip safety for all members
                traversal_attempts = [m.filename for m in infolist if not is_safe_path(target_dest, os.path.basename(m.filename))]
                if traversal_attempts:
                    print(f"  CRITICAL: {len(traversal_attempts)} members violate path traversal boundary!")
                else:
                    print("  Path Traversal:  SAFE (100% of member basenames stay within destination directory)")
        except Exception as e:
            print(f"  ZIP Error:       {e}")
            
    print("\nDry-run completed successfully. Zero files extracted.")
    return True

def extract_archive(target_key, raw_dir, extract_dir, split_manifest_path=DEFAULT_SPLIT_MANIFEST):
    """
    Safely extract single archive with:
    - Pre-extraction size, MD5, and testzip verification
    - Staging temporary directory extraction preventing partial extraction
    - Per-file CRC32 and size verification
    - Comprehensive manifest cross-referencing against frozen split
    - Atomic directory promotion upon full completion
    """
    cfg = EXTRACTION_CONFIG[target_key]
    archive_path = os.path.join(raw_dir, cfg["archive_name"])
    target_dest = os.path.join(extract_dir, cfg["dest_subdir"])
    staging_dest = os.path.join(extract_dir, f".{cfg['dest_subdir']}_staging")
    backup_dest = os.path.join(extract_dir, f".{cfg['dest_subdir']}_backup")
    expected_count = cfg["expected_files"]
    expected_size = cfg["expected_size"]
    expected_md5 = cfg["expected_md5"]
    
    if not os.path.exists(archive_path):
        raise FileNotFoundError(f"Archive not found: {archive_path}")
        
    print(f"\n[{cfg['archive_name']}] Step 1: Pre-extraction archive verification...")
    actual_size = os.path.getsize(archive_path)
    if actual_size != expected_size:
        raise ValueError(f"Archive size mismatch for {cfg['archive_name']}: expected {expected_size}, got {actual_size}")
    print(f"  -> File size verified: {actual_size:,} bytes")
    
    actual_md5 = compute_file_md5(archive_path)
    if actual_md5 != expected_md5:
        raise ValueError(f"MD5 mismatch for {cfg['archive_name']}: expected {expected_md5}, got {actual_md5}")
    print(f"  -> MD5 verified: {actual_md5}")
    
    # Load split manifest for cross-referencing
    records_by_base, records_by_pair = load_split_manifest(split_manifest_path)
    print(f"  -> Split manifest loaded: {len(records_by_base):,} records from {split_manifest_path}")
    
    # Step 2: Test ZIP structure
    print(f"[{cfg['archive_name']}] Step 2: Testing ZIP archive integrity (testzip)...")
    with zipfile.ZipFile(archive_path, "r") as zf:
        first_bad = zf.testzip()
        if first_bad is not None:
            raise ValueError(f"Corrupt ZIP file detected! First defective member: {first_bad}")
        print("  -> testzip() PASSED: 0 corrupted members.")
        
        infolist = zf.infolist()
        file_members = [m for m in infolist if not m.is_dir()]
        print(f"  -> Total files to extract: {len(file_members):,} (expected {expected_count})")
        if len(file_members) != expected_count:
            raise ValueError(f"Member count mismatch: found {len(file_members)}, expected {expected_count}")
            
        # Step 3: Clean/prepare staging temporary directory
        if os.path.exists(staging_dest):
            print(f"[{cfg['archive_name']}] Cleaning existing staging directory {staging_dest}...")
            shutil.rmtree(staging_dest, ignore_errors=True)
        os.makedirs(staging_dest, exist_ok=True)
        
        print(f"[{cfg['archive_name']}] Step 3: Extracting {len(file_members):,} members into staging directory...")
        print(f"  Staging directory: {os.path.abspath(staging_dest)}")
        
        manifest_entries = []
        total_uncompressed_bytes = 0
        
        for idx, member in enumerate(file_members, 1):
            pure_filename = os.path.basename(member.filename)
            stem = os.path.splitext(pure_filename)[0]
            
            if target_key == "real":
                base_id = stem
                target_id = ""
                split_info = records_by_base.get(base_id, {})
                matched_frozen = bool(base_id in records_by_base)
            else:  # fake
                parts = stem.split("_")
                base_id = parts[0]
                target_id = parts[1] if len(parts) > 1 else ""
                split_info = records_by_pair.get((base_id, target_id), {})
                matched_frozen = bool((base_id, target_id) in records_by_pair)
            
            # Path Traversal Guard (ZipSlip Prevention)
            if not is_safe_path(staging_dest, pure_filename):
                shutil.rmtree(staging_dest, ignore_errors=True)
                raise PermissionError(f"Security Alert: Path traversal detected in member '{member.filename}'!")
                
            staged_file_path = os.path.join(staging_dest, pure_filename)
            tmp_dest = staged_file_path + ".tmp"
            
            # Extract to temporary file first
            with zf.open(member) as source_stream, open(tmp_dest, "wb") as target_file:
                shutil.copyfileobj(source_stream, target_file)
                
            # Verify extracted file size
            extracted_size = os.path.getsize(tmp_dest)
            if extracted_size != member.file_size:
                os.remove(tmp_dest)
                shutil.rmtree(staging_dest, ignore_errors=True)
                raise ValueError(f"Extracted size mismatch for {pure_filename}: expected {member.file_size}, got {extracted_size}")
                
            # Verify extracted file CRC32
            extracted_crc = compute_file_crc32(tmp_dest)
            if extracted_crc != member.CRC:
                os.remove(tmp_dest)
                shutil.rmtree(staging_dest, ignore_errors=True)
                raise ValueError(f"CRC32 mismatch for {pure_filename}: expected {member.CRC:08x}, got {extracted_crc:08x}")
                
            # Atomic promote within staging
            os.rename(tmp_dest, staged_file_path)
            total_uncompressed_bytes += extracted_size
            
            manifest_entries.append({
                "filename": pure_filename,
                "base_id": base_id,
                "target_id": target_id,
                "file_size": extracted_size,
                "crc32": f"{extracted_crc:08x}",
                "split": split_info.get("split", "UNKNOWN"),
                "component_id": split_info.get("component_id", "UNKNOWN"),
                "race": split_info.get("race", "UNKNOWN"),
                "gender": split_info.get("gender", "UNKNOWN"),
                "age": split_info.get("age", "UNKNOWN"),
                "matched_frozen": matched_frozen,
                "status": "EXTRACTED_VERIFIED",
                "extracted_path": os.path.join(target_dest, pure_filename),
            })
            
            if idx % 500 == 0 or idx == len(file_members):
                print(f"  Processed {idx:,} / {len(file_members):,} ({idx / len(file_members) * 100:.1f}%) — {total_uncompressed_bytes / (1024**2):.1f} MB decompressed")
                
    # Step 4: Validate staging directory before atomic completion
    print(f"\n[{cfg['archive_name']}] Step 4: Validating staged files before atomic promotion...")
    staged_files = [f for f in os.listdir(staging_dest) if os.path.isfile(os.path.join(staging_dest, f))]
    if len(staged_files) != expected_count:
        shutil.rmtree(staging_dest, ignore_errors=True)
        raise ValueError(f"Staged file count mismatch: expected {expected_count}, got {len(staged_files)}")
        
    staged_set = set(staged_files)
    if target_key == "real":
        expected_set = {f"{b}.mp4" for b in records_by_base.keys()}
    else:
        expected_set = {f"{b}_{t}.mp4" for (b, t) in records_by_pair.keys()}
        
    missing_from_staged = expected_set - staged_set
    unexpected_in_staged = staged_set - expected_set
    
    if missing_from_staged:
        shutil.rmtree(staging_dest, ignore_errors=True)
        raise ValueError(f"Validation failed: {len(missing_from_staged)} expected files missing from staging!")
    if unexpected_in_staged:
        shutil.rmtree(staging_dest, ignore_errors=True)
        raise ValueError(f"Validation failed: {len(unexpected_in_staged)} unexpected files present in staging!")
        
    print(f"  -> Exactly {len(staged_files):,} files validated in staging.")
    print("  -> Zero missing, zero unexpected, zero duplicate files.")
    
    # Step 5: Atomic Directory Promotion
    print(f"\n[{cfg['archive_name']}] Step 5: Atomically promoting staging to destination directory...")
    if os.path.exists(target_dest):
        if os.path.exists(backup_dest):
            shutil.rmtree(backup_dest, ignore_errors=True)
        os.rename(target_dest, backup_dest)
        os.rename(staging_dest, target_dest)
        shutil.rmtree(backup_dest, ignore_errors=True)
    else:
        os.rename(staging_dest, target_dest)
        
    print(f"  -> Promotion complete: {target_dest} now active and verified.")
    return manifest_entries

def main():
    parser = argparse.ArgumentParser(description="BIOTrace HiDF Safe Extraction Script")
    parser.add_argument("--target", choices=["real", "fake", "all"], default="real",
                        help="Which archive to extract (default: real)")
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR,
                        help=f"Raw archive directory (default: {DEFAULT_RAW_DIR})")
    parser.add_argument("--extract-dir", default=DEFAULT_EXTRACTED_DIR,
                        help=f"Extraction directory (default: {DEFAULT_EXTRACTED_DIR})")
    parser.add_argument("--split-manifest", default=DEFAULT_SPLIT_MANIFEST,
                        help=f"Frozen Protocol C split manifest (default: {DEFAULT_SPLIT_MANIFEST})")
    parser.add_argument("--manifest-path", default=None,
                        help="Output manifest file path (defaults to target specific manifest)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Perform pre-flight verification without extracting files")
    args = parser.parse_args()
    
    targets = ["real", "fake"] if args.target == "all" else [args.target]
    
    if args.dry_run:
        dry_run_extraction(args.raw_dir, args.extract_dir, targets, args.split_manifest)
        sys.exit(0)
        
    print("=" * 70)
    print("BIOTrace HiDF Safe Extraction Execution Mode")
    print("=" * 70)
    print(f"Target:                {args.target}")
    print(f"Raw Directory:         {os.path.abspath(args.raw_dir)}")
    print(f"Extraction Root:       {os.path.abspath(args.extract_dir)}")
    print(f"Split Manifest:        {os.path.abspath(args.split_manifest)}")
    
    os.makedirs(args.extract_dir, exist_ok=True)
    
    for t in targets:
        cfg = EXTRACTION_CONFIG[t]
        manifest_rows = extract_archive(t, args.raw_dir, args.extract_dir, args.split_manifest)
        
        # Determine output manifest path
        out_manifest = args.manifest_path if args.manifest_path else cfg["default_manifest"]
        
        fieldnames = [
            "filename", "base_id", "target_id", "file_size", "crc32", "split",
            "component_id", "race", "gender", "age", "matched_frozen", "status", "extracted_path"
        ]
        with open(out_manifest, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(manifest_rows)
        print(f"\nWrote primary extraction manifest to {out_manifest} ({len(manifest_rows):,} records)")
        
        # Also write compatibility manifest in extracted dir
        compat_path = os.path.join(args.extract_dir, f"{t}_extraction_manifest.csv")
        with open(compat_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(manifest_rows)
        print(f"Wrote in-tree extraction manifest to {compat_path}")

if __name__ == "__main__":
    main()
