#!/usr/bin/env python3
"""
BIOTrace Phase 5 — Deterministic High-Throughput Frame Extractor.
Extracts balanced, temporally uniform frames from authentic and deepfake videos
under Protocol C frozen bipartite split partitions.
"""

import argparse
import hashlib
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import pandas as pd


def compute_sha256(filepath: Path) -> str:
    """Computes SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def calculate_sample_indices(total_frames: int, num_samples: int) -> List[int]:
    """
    Computes strictly increasing, temporally uniform frame indices across clip.
    Formula: round(i * (N - 1) / (K - 1))
    """
    if total_frames <= 0:
        return []
    if total_frames <= num_samples:
        return list(range(total_frames))
    indices = []
    for i in range(num_samples):
        idx = int(round(i * (total_frames - 1) / (num_samples - 1)))
        indices.append(idx)
    # Remove any potential duplicates while preserving order
    unique_indices = []
    for idx in indices:
        if not unique_indices or idx > unique_indices[-1]:
            unique_indices.append(idx)
    return unique_indices


def process_single_video(task: Tuple) -> Tuple[List[Dict], Optional[str]]:
    """
    Worker function to process a single video:
    Decodes target frames, resizes to target resolution, saves JPEG, and records metadata.
    """
    (
        video_id,
        base_id,
        target_id,
        label,
        split,
        component_id,
        video_path_str,
        output_dir_str,
        num_samples,
        target_size,
        resume,
    ) = task

    video_path = Path(video_path_str)
    output_dir = Path(output_dir_str)
    split_dir = output_dir / split / label

    if not video_path.is_file():
        err_msg = f"Video file not found: {video_path}"
        return [], err_msg

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        err_msg = f"Failed to open video: {video_path}"
        return [], err_msg

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120:
        fps = 24.0  # Fallback standard fps
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        # Fallback: count frames by reading
        pass

    target_indices = calculate_sample_indices(total_frames if total_frames > 0 else 72, num_samples)
    target_set = set(target_indices)

    # Check resume state
    if resume and target_indices:
        all_exist = True
        existing_records = []
        for idx in target_indices:
            frame_filename = f"{video_id}_f{idx:04d}.jpg"
            frame_path = split_dir / frame_filename
            if not frame_path.is_file() or frame_path.stat().st_size == 0:
                all_exist = False
                break
            frame_sha = compute_sha256(frame_path)
            ts = idx / fps
            existing_records.append({
                "frame_id": f"{video_id}_f{idx:04d}",
                "video_id": video_id,
                "base_id": base_id,
                "target_id": target_id if target_id else "",
                "label": label,
                "split": split,
                "component_id": component_id,
                "source_video_path": str(video_path).replace("\\", "/"),
                "frame_path": str(frame_path).replace("\\", "/"),
                "timestamp_seconds": round(ts, 4),
                "original_width": orig_w,
                "original_height": orig_h,
                "output_width": target_size,
                "output_height": target_size,
                "frame_sha256": frame_sha,
                "extraction_status": "CACHED_RESUMED",
            })
        if all_exist:
            cap.release()
            return existing_records, None

    records = []
    curr_frame = 0
    extracted_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if curr_frame in target_set:
            frame_filename = f"{video_id}_f{curr_frame:04d}.jpg"
            frame_path = split_dir / frame_filename

            # Resample frame
            if orig_w > target_size and orig_h > target_size:
                interp = cv2.INTER_AREA
            else:
                interp = cv2.INTER_LINEAR

            resized = cv2.resize(frame, (target_size, target_size), interpolation=interp)
            cv2.imwrite(str(frame_path), resized, [cv2.IMWRITE_JPEG_QUALITY, 95])

            frame_sha = compute_sha256(frame_path)
            ts = curr_frame / fps
            status = "SUCCESS" if total_frames >= num_samples else "WARNING_SHORT_CLIP"

            records.append({
                "frame_id": f"{video_id}_f{curr_frame:04d}",
                "video_id": video_id,
                "base_id": base_id,
                "target_id": target_id if target_id else "",
                "label": label,
                "split": split,
                "component_id": component_id,
                "source_video_path": str(video_path).replace("\\", "/"),
                "frame_path": str(frame_path).replace("\\", "/"),
                "timestamp_seconds": round(ts, 4),
                "original_width": orig_w,
                "original_height": orig_h,
                "output_width": target_size,
                "output_height": target_size,
                "frame_sha256": frame_sha,
                "extraction_status": status,
            })
            extracted_count += 1

        curr_frame += 1
        if extracted_count >= len(target_indices):
            break

    cap.release()

    if not records:
        return [], f"Zero frames decoded from {video_path}"

    return records, None


def main():
    parser = argparse.ArgumentParser(description="BIOTrace Deterministic Frame Extraction Engine")
    parser.add_argument("--manifest", type=str, default="protocol_c_split_manifest.csv")
    parser.add_argument("--real-dir", type=str, default="data/extracted/hidf/real")
    parser.add_argument("--fake-dir", type=str, default="data/extracted/hidf/fake")
    parser.add_argument("--output-dir", type=str, default="data/derived/frames")
    parser.add_argument("--frames-per-video", type=int, default=10)
    parser.add_argument("--target-size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 4, 8))
    parser.add_argument("--pilot", action="store_true", help="Run on pilot subset only")
    parser.add_argument("--pilot-count", type=int, default=2, help="Pairs per split for pilot")
    parser.add_argument("--resume", action="store_true", default=True, help="Skip already extracted frames")
    parser.add_argument("--output-manifest", type=str, default="phase5_frame_extraction_manifest.csv")
    parser.add_argument("--stats-csv", type=str, default="phase5_extraction_statistics.csv")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    real_dir = Path(args.real_dir)
    fake_dir = Path(args.fake_dir)
    output_dir = Path(args.output_dir)

    print("=================================================================")
    print("BIOTRACE PHASE 5: FRAME EXTRACTION PIPELINE")
    print("=================================================================")
    print(f"Manifest:          {manifest_path}")
    print(f"Real Videos Dir:   {real_dir}")
    print(f"Fake Videos Dir:   {fake_dir}")
    print(f"Output Frames Dir: {output_dir}")
    print(f"Frames per Video:  {args.frames_per_video}")
    print(f"Target Size:       {args.target_size}x{args.target_size}")
    print(f"Worker Processes:  {args.workers}")
    print(f"Pilot Mode:        {args.pilot}")
    print("=================================================================")

    if not manifest_path.is_file():
        print(f"CRITICAL ERROR: Manifest {manifest_path} not found!", file=sys.stderr)
        sys.exit(1)

    df_manifest = pd.read_csv(manifest_path)
    print(f"Loaded frozen manifest with {len(df_manifest)} video pairs.")

    # Create destination partition directories
    for split in ["train", "val", "test"]:
        for label in ["real", "fake"]:
            (output_dir / split / label).mkdir(parents=True, exist_ok=True)

    # Filter for pilot if requested
    if args.pilot:
        print(f"Configuring pilot subset: {args.pilot_count} pairs per split...")
        pilot_rows = []
        for s in ["train", "val", "test"]:
            subset = df_manifest[df_manifest["split"] == s].head(args.pilot_count)
            pilot_rows.append(subset)
        df_target = pd.concat(pilot_rows, ignore_index=True)
    else:
        df_target = df_manifest

    print(f"Total pairs to process: {len(df_target)} ({len(df_target)*2} videos total)")

    # Build work tasks (1 task per video)
    tasks = []
    for _, row in df_target.iterrows():
        b_id = row["base_id"]
        t_id = row["target_id"]
        split = row["split"]
        comp_id = row["component_id"]

        # 1. Real video task
        real_v_path = real_dir / f"{b_id}.mp4"
        tasks.append((
            b_id,            # video_id
            b_id,            # base_id
            "",              # target_id (empty for real)
            "real",          # label
            split,
            comp_id,
            str(real_v_path),
            str(output_dir),
            args.frames_per_video,
            args.target_size,
            args.resume,
        ))

        # 2. Fake video task
        fake_v_id = f"{b_id}_{t_id}"
        fake_v_path = fake_dir / f"{fake_v_id}.mp4"
        tasks.append((
            fake_v_id,       # video_id
            b_id,            # base_id
            t_id,            # target_id
            "fake",          # label
            split,
            comp_id,
            str(fake_v_path),
            str(output_dir),
            args.frames_per_video,
            args.target_size,
            args.resume,
        ))

    print(f"Dispatched task queue of {len(tasks)} videos to {args.workers} worker processes...")
    start_time = time.time()

    all_records = []
    failed_videos = []
    completed_count = 0
    total_tasks = len(tasks)

    # Process in parallel with ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_video, t): t for t in tasks}

        for future in as_completed(futures):
            task = futures[future]
            video_id = task[0]
            try:
                records, err = future.result()
                if err:
                    failed_videos.append({"video_id": video_id, "error": err})
                else:
                    all_records.extend(records)
            except Exception as ex:
                failed_videos.append({"video_id": video_id, "error": str(ex)})

            completed_count += 1
            if completed_count % 500 == 0 or completed_count == total_tasks:
                elapsed = time.time() - start_time
                fps_rate = completed_count / elapsed if elapsed > 0 else 0
                print(f"[{completed_count}/{total_tasks}] videos processed ({fps_rate:.1f} vids/s, {len(all_records)} frames generated, {len(failed_videos)} failures)...")

    elapsed_total = time.time() - start_time
    print(f"\nExtraction completed in {elapsed_total:.2f} seconds.")
    print(f"Total frames extracted: {len(all_records)}")
    print(f"Total failed videos:    {len(failed_videos)}")

    if failed_videos:
        print("WARNING: Some videos failed extraction:")
        for fv in failed_videos[:5]:
            print(f"  • {fv['video_id']}: {fv['error']}")
        if len(failed_videos) > 5:
            print(f"  ... and {len(failed_videos) - 5} more.")

    # Write frame extraction manifest
    df_records = pd.DataFrame(all_records)
    if not df_records.empty:
        # Sort by split, label, video_id, frame_id
        df_records.sort_values(by=["split", "label", "video_id", "frame_id"], inplace=True)
        manifest_out = Path(args.output_manifest)
        df_records.to_csv(manifest_out, index=False)
        print(f"Wrote frame metadata manifest: {manifest_out} ({len(df_records)} rows)")

        # Calculate statistics
        stats = []
        for split in ["train", "val", "test"]:
            for label in ["real", "fake"]:
                subset = df_records[(df_records["split"] == split) & (df_records["label"] == label)]
                unique_vids = subset["video_id"].nunique()
                count_frames = len(subset)
                stats.append({
                    "split": split,
                    "label": label,
                    "videos_processed": unique_vids,
                    "frames_extracted": count_frames,
                    "avg_frames_per_video": count_frames / unique_vids if unique_vids > 0 else 0,
                })
        df_stats = pd.DataFrame(stats)
        stats_out = Path(args.stats_csv)
        df_stats.to_csv(stats_out, index=False)
        print(f"Wrote extraction statistics:  {stats_out}")
        print("\n=== EXTRACTION SUMMARY BY SPLIT & LABEL ===")
        print(df_stats.to_string(index=False))

    return len(failed_videos)


if __name__ == "__main__":
    sys.exit(main())
