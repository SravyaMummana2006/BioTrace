"""Feature extraction script using ImageNet-pretrained EfficientNet-B0 backbone.

Precomputes and caches 1280-dimensional feature representations for extracted frames,
enabling rapid, reproducible multi-epoch training and video-level evaluation on CPU.
"""

import argparse
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.models as models

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AI_ENGINE_SRC = PROJECT_ROOT / "ai-engine" / "src"
if str(AI_ENGINE_SRC) not in sys.path:
    sys.path.insert(0, str(AI_ENGINE_SRC))

from biotrace.vision.image_preprocessing import letterbox_bgr


def extract_split(
    df_split: pd.DataFrame,
    split_name: str,
    output_path: Path,
    model: torch.nn.Module,
    batch_size: int = 128,
    num_threads: int = 8,
):
    if output_path.exists():
        print(f"[{split_name.upper()}] Features already cached at {output_path}. Checking length...")
        data = torch.load(output_path, weights_only=True)
        if len(data["features"]) == len(df_split):
            print(f"[{split_name.upper()}] Valid cache verified ({len(data['features'])} samples). Skipping.")
            return
        else:
            print(f"[{split_name.upper()}] Incomplete cache ({len(data['features'])} != {len(df_split)}). Recomputing.")

    torch.set_num_threads(num_threads)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    total_samples = len(df_split)
    paths = df_split["frame_path"].tolist()
    labels = [1.0 if l.strip().lower() == "fake" else 0.0 for l in df_split["label"].tolist()]
    video_ids = df_split["video_id"].astype(str).tolist()
    frame_ids = df_split["frame_id"].astype(str).tolist()

    print(f"[{split_name.upper()}] Starting feature extraction for {total_samples} frames...")
    start_time = time.time()
    feats_list = []

    with torch.no_grad():
        for i in range(0, total_samples, batch_size):
            chunk_paths = paths[i : i + batch_size]
            imgs = []
            for p in chunk_paths:
                img = cv2.imread(p)
                if img is None:
                    raise FileNotFoundError(f"Frame file not found or corrupted: {p}")
                img, _ = letterbox_bgr(img, target_size=(224, 224))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                imgs.append(img)

            arr = np.stack(imgs).transpose(0, 3, 1, 2)
            batch = torch.from_numpy(arr).float().div_(255.0).sub_(mean).div_(std)
            feat = model.avgpool(model.features(batch)).flatten(1)
            feats_list.append(feat)

            done = min(i + batch_size, total_samples)
            if done % 2000 == 0 or done == total_samples:
                elapsed = time.time() - start_time
                fps = done / max(0.1, elapsed)
                remaining = (total_samples - done) / max(0.1, fps)
                print(f"[{split_name.upper()}] Processed {done}/{total_samples} frames ({fps:.1f} fps, {remaining:.0f}s remaining)")

    all_features = torch.cat(feats_list, dim=0)
    output_data = {
        "features": all_features,
        "labels": torch.tensor(labels, dtype=torch.float32),
        "video_ids": video_ids,
        "frame_ids": frame_ids,
        "split": split_name,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_data, str(output_path))
    total_time = time.time() - start_time
    print(f"[{split_name.upper()}] Extraction complete in {total_time:.1f}s. Saved to {output_path} (Shape: {all_features.shape}).")


def main():
    parser = argparse.ArgumentParser(description="Extract EfficientNet-B0 features for BIOTrace frames.")
    parser.add_argument("--manifest", type=str, default="reports/data/phase5_frame_extraction_manifest.csv")
    parser.add_argument("--output-dir", type=str, default="data/cache/features")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        sys.exit(f"Manifest not found: {manifest_path}")

    print(f"Reading manifest from {manifest_path}...")
    df = pd.read_csv(manifest_path)
    print(f"Loaded {len(df)} manifest rows.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Initializing EfficientNet-B0 feature extractor with ImageNet pretrained weights...")
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.eval()

    # Extract in order: val first, then test, then train (so val/test are cached first)
    for split in ["val", "test", "train"]:
        df_split = df[df["split"] == split].reset_index(drop=True)
        out_path = out_dir / f"{split}_features.pt"
        extract_split(
            df_split=df_split,
            split_name=split,
            output_path=out_path,
            model=model,
            batch_size=args.batch_size,
            num_threads=args.threads,
        )

    print("All splits extracted and verified successfully.")


if __name__ == "__main__":
    main()
