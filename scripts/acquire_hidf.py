#!/usr/bin/env python3
"""
BIOTrace Controlled Dataset Acquisition Script
Module: scripts/acquire_hidf.py
Safe, resumable, verified downloader for official HiDF archives from Zenodo.
Strictly requires explicit command arguments. Supports dry-run verification mode.
Features robust auto-resume and retry loops for resilient broadband acquisition.
"""

import os
import sys
import argparse
import hashlib
import json
import time
import shutil
import urllib.parse
from datetime import datetime, timezone
import requests
from urllib3.exceptions import IncompleteRead, ProtocolError

TARGET_CONFIG = {
    "real": {
        "filename": "Real-vid.zip",
        "url": "https://zenodo.org/records/16140829/files/Real-vid.zip",
        "expected_size": 842885152,
        "expected_md5": "ee53b75850a7f54b45d61a78085eda0a",
        "description": "HiDF Authentic Base Videos Archive",
    },
    "fake": {
        "filename": "Fake-vid.zip",
        "url": "https://zenodo.org/records/16140829/files/Fake-vid.zip",
        "expected_size": 715291222,
        "expected_md5": "708890ccaedc3a6cccee0c1d3e64a711",
        "description": "HiDF Deepfake Derivative Videos Archive",
    },
}

DEFAULT_RAW_DIR = os.path.join("data", "raw", "hidf")
CHUNK_SIZE = 1024 * 1024  # 1 MB streaming chunks

def compute_hashes(filepath):
    """Compute both MD5 and SHA-256 in a single streaming pass."""
    md5_hash = hashlib.md5()
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(CHUNK_SIZE):
            md5_hash.update(chunk)
            sha256_hash.update(chunk)
    return md5_hash.hexdigest(), sha256_hash.hexdigest()

def check_disk_space(target_dir, required_bytes):
    """Check whether destination volume has sufficient free space."""
    os.makedirs(target_dir, exist_ok=True)
    total, used, free = shutil.disk_usage(target_dir)
    return free, free >= (required_bytes + 1024 * 1024 * 1024)  # 1 GB safety buffer

def perform_dry_run(targets, raw_dir):
    """Execute pre-flight environment and parameter check without downloading."""
    print("=" * 70, flush=True)
    print("BIOTrace HiDF Acquisition — DRY-RUN PRE-FLIGHT VERIFICATION", flush=True)
    print("=" * 70, flush=True)
    print(f"Destination Directory: {os.path.abspath(raw_dir)}", flush=True)
    
    total_required = sum(TARGET_CONFIG[t]["expected_size"] for t in targets)
    free_bytes, has_space = check_disk_space(raw_dir, total_required)
    
    print(f"Target Archives Selected: {', '.join(targets)}", flush=True)
    print(f"Total Required Size: {total_required:,} bytes ({total_required / (1024**2):.2f} MB)", flush=True)
    print(f"Available Disk Space: {free_bytes:,} bytes ({free_bytes / (1024**3):.2f} GB)", flush=True)
    print(f"Disk Space Check: {'PASS' if has_space else 'FAIL'}", flush=True)
    
    for t in targets:
        cfg = TARGET_CONFIG[t]
        dest_final = os.path.join(raw_dir, cfg["filename"])
        dest_part = dest_final + ".part"
        print(f"\nTarget: {t.upper()} ({cfg['description']})", flush=True)
        print(f"  URL: {cfg['url']}", flush=True)
        print(f"  Final Destination: {dest_final}", flush=True)
        print(f"  Expected Size: {cfg['expected_size']:,} bytes", flush=True)
        print(f"  Expected MD5: {cfg['expected_md5']}", flush=True)
        
        if os.path.exists(dest_final):
            actual_size = os.path.getsize(dest_final)
            print(f"  Current Status: FINAL FILE EXISTS ({actual_size:,} bytes)", flush=True)
        elif os.path.exists(dest_part):
            part_size = os.path.getsize(dest_part)
            print(f"  Current Status: PARTIAL DOWNLOAD EXISTS ({part_size:,} bytes)", flush=True)
        else:
            print("  Current Status: NOT YET DOWNLOADED", flush=True)
            
    print("\nDry-run completed successfully. Zero bytes downloaded.", flush=True)
    return True

def download_target(target_key, raw_dir, log_entries):
    """Execute streaming download with resume, verification, and atomic rename."""
    cfg = TARGET_CONFIG[target_key]
    filename = cfg["filename"]
    url = cfg["url"]
    expected_size = cfg["expected_size"]
    expected_md5 = cfg["expected_md5"]
    
    os.makedirs(raw_dir, exist_ok=True)
    final_path = os.path.join(raw_dir, filename)
    part_path = final_path + ".part"
    
    # 1. Check if final file already exists and is valid
    if os.path.exists(final_path):
        size = os.path.getsize(final_path)
        if size == expected_size:
            print(f"[{filename}] Final file already exists with expected size ({size:,} B). Verifying hash...", flush=True)
            md5_val, sha256_val = compute_hashes(final_path)
            if md5_val == expected_md5:
                print(f"[{filename}] MD5 MATCH: {md5_val}. Already acquired and verified.", flush=True)
                log_entries.append({
                    "target": target_key,
                    "filename": filename,
                    "status": "ALREADY_VERIFIED",
                    "size": size,
                    "md5": md5_val,
                    "sha256": sha256_val,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                return True
            else:
                print(f"[{filename}] WARNING: Existing final file has MD5 mismatch ({md5_val} != {expected_md5}). Re-downloading.", flush=True)
                os.remove(final_path)
        else:
            print(f"[{filename}] WARNING: Existing final file has size mismatch ({size} != {expected_size}). Re-downloading.", flush=True)
            os.remove(final_path)

    # 2. Resumable streaming loop with automatic retry
    max_retries = 100
    retry_delay = 3
    attempt = 0
    start_time = time.time()
    
    while attempt < max_retries:
        existing_bytes = 0
        if os.path.exists(part_path):
            existing_bytes = os.path.getsize(part_path)
            if existing_bytes > expected_size:
                print(f"[{filename}] Partial size ({existing_bytes:,} B) > expected ({expected_size:,} B). Resetting.", flush=True)
                os.remove(part_path)
                existing_bytes = 0
            elif existing_bytes == expected_size:
                print(f"[{filename}] Partial file size matches expected size. Proceeding to hash verification.", flush=True)
                break

        headers = {"User-Agent": "BIOTrace-Dataset-Acquisition/1.0"}
        mode = "wb"
        if existing_bytes > 0:
            headers["Range"] = f"bytes={existing_bytes}-"
            mode = "ab"
            print(f"[{filename}] Resuming from byte {existing_bytes:,} of {expected_size:,} ({(existing_bytes/expected_size)*100:.2f}%)...", flush=True)
        else:
            print(f"[{filename}] Starting download from offset 0...", flush=True)

        try:
            session = requests.Session()
            resp = session.get(url, headers=headers, stream=True, timeout=60)
            
            if existing_bytes > 0 and resp.status_code != 206:
                print(f"[{filename}] Server returned {resp.status_code} (Range not accepted). Restarting from offset 0.", flush=True)
                mode = "wb"
                existing_bytes = 0
                resp = session.get(url, headers={"User-Agent": "BIOTrace-Dataset-Acquisition/1.0"}, stream=True, timeout=60)
                
            if resp.status_code not in (200, 206):
                raise RuntimeError(f"HTTP request failed with status code {resp.status_code}")

            downloaded = existing_bytes
            last_print = time.time()
            session_start = time.time()
            session_downloaded = 0
            
            with open(part_path, mode) as f:
                for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        f.flush()
                        downloaded += len(chunk)
                        session_downloaded += len(chunk)
                        now = time.time()
                        if now - last_print >= 5.0 or downloaded == expected_size:
                            pct = (downloaded / expected_size) * 100.0
                            s_elapsed = now - session_start
                            speed = session_downloaded / (s_elapsed if s_elapsed > 0 else 1) / (1024**2)
                            remaining_mb = (expected_size - downloaded) / (1024**2)
                            eta_sec = (remaining_mb / speed) if speed > 0 else 0
                            print(f"[{filename}] {downloaded:,} / {expected_size:,} B ({pct:.2f}%) | {speed:.2f} MB/s | ETA: {eta_sec:.0f}s ({eta_sec/60:.1f}m)", flush=True)
                            last_print = now
                            
            # Check if finished
            if os.path.exists(part_path) and os.path.getsize(part_path) == expected_size:
                print(f"[{filename}] Full stream received ({expected_size:,} bytes).", flush=True)
                break
                
        except (requests.exceptions.RequestException, ProtocolError, ConnectionError, IncompleteRead) as e:
            attempt += 1
            cur_sz = os.path.getsize(part_path) if os.path.exists(part_path) else 0
            print(f"\n[{filename}] Transient network drop at {cur_sz:,} bytes: {type(e).__name__}. Retrying in {retry_delay}s (Attempt {attempt}/{max_retries})...", flush=True)
            time.sleep(retry_delay)
            continue
            
    # 3. Post-download Verification
    if not os.path.exists(part_path):
        raise FileNotFoundError(f"[{filename}] Download file missing after download loop!")
        
    actual_size = os.path.getsize(part_path)
    if actual_size != expected_size:
        raise ValueError(f"[{filename}] Downloaded size mismatch: expected {expected_size}, got {actual_size} bytes!")
        
    print(f"\n[{filename}] Download complete ({actual_size:,} bytes). Verifying checksums...", flush=True)
    md5_val, sha256_val = compute_hashes(part_path)
    print(f"[{filename}] Calculated MD5:    {md5_val}", flush=True)
    print(f"[{filename}] Expected MD5:      {expected_md5}", flush=True)
    print(f"[{filename}] Calculated SHA256: {sha256_val}", flush=True)
    
    if md5_val != expected_md5:
        raise ValueError(f"[{filename}] CRITICAL CHECKSUM ERROR: MD5 mismatch ({md5_val} != {expected_md5})!")
        
    # 4. Atomic Promotion
    os.rename(part_path, final_path)
    print(f"[{filename}] Integrity verified. Promoted to {final_path}.", flush=True)
    
    log_entries.append({
        "target": target_key,
        "filename": filename,
        "status": "SUCCESSFULLY_ACQUIRED",
        "size": actual_size,
        "md5": md5_val,
        "sha256": sha256_val,
        "download_duration_sec": round(time.time() - start_time, 2),
        "timestamp": datetime.now(timezone.utc).isoformat()
    })
    return True

def main():
    parser = argparse.ArgumentParser(description="BIOTrace HiDF Dataset Acquisition Script")
    parser.add_argument("--target", choices=["real", "fake", "all"], default="all",
                        help="Which archive to acquire (default: all)")
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR,
                        help=f"Target directory for raw archives (default: {DEFAULT_RAW_DIR})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run pre-flight verification without downloading data")
    args = parser.parse_args()
    
    targets = ["real", "fake"] if args.target == "all" else [args.target]
    
    if args.dry_run:
        perform_dry_run(targets, args.raw_dir)
        sys.exit(0)
        
    print("=" * 70, flush=True)
    print("BIOTrace HiDF Acquisition Execution Mode", flush=True)
    print("=" * 70, flush=True)
    
    log_entries = []
    log_path = os.path.join(args.raw_dir, "acquisition_log.json")
    
    for t in targets:
        print(f"\nProcessing target: {t}...", flush=True)
        download_target(t, args.raw_dir, log_entries)
        
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_entries, f, indent=2)
    print(f"\nWrote acquisition metadata log to {log_path}", flush=True)

if __name__ == "__main__":
    main()
