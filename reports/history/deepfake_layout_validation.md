# Deepfake Analysis: Layout, Preprocessing, and Verdict Validation

## What changed

1. **Aspect-ratio-preserving preprocessing**
   - Added `biotrace.vision.image_preprocessing.letterbox_bgr` / `preprocess_letterboxed`.
   - Inference (`BaselineInferenceService`, `ImageNormalizer`, `FaceCropExtractor`) now letterboxes to 224×224 with symmetric padding instead of stretching.
   - Training/validation feature generation (`scripts/extract_features.py`) and frame extraction (`scripts/extract_hidf_frames.py`) use the same letterbox contract.
   - `ai-engine/scripts/predict_video.py` uses the same contract.

2. **Rotation metadata**
   - `biotrace.vision.video_capture.open_video_capture` requests `CAP_PROP_ORIENTATION_AUTO`.
   - Verified OpenCV/FFmpeg decodes a `rotation=-90` portrait clip into display orientation (`1080×1920`) without a second manual rotation.

3. **Verdict gate removed**
   - Removed the `non_square_input` UNASSESSED gate from the FastAPI service and Java UI.
   - Added evidence-based verdict derivation:
     - `FAKE` → **Likely Fake**
     - `REAL` → **Likely Authentic**
     - visual/physiological conflict without fusion → **Inconclusive**
     - concrete model/media failure → **Unassessed** with `verdict_reason`
   - API report now carries `verdict_basis` and `verdict_reason`; the Java UI renders both.
   - `BaselineInferenceResult` now carries `status`, `error_message`, and `preprocessing`.

## Measured layout results

Clips: a 6-second portrait source (`IMG_2915.mp4`) converted to 864×1920, 1920×1080, and 1080×1080, plus a 2.6-second rotated phone clip (`Snapchat-1342977448.mp4`, coded 1920×1080 with rotation metadata −90).

These are **unlabeled personal clips**, not a labeled real/fake benchmark. Physiological evidence was unavailable on these short clips, so the overall verdict is visual-only. No false-positive/false-negative accuracy claim can be made from them.

| Layout | Before (stretch) | Before verdict | After (letterbox) | After verdict | After basis |
|---|---:|---|---:|---|---|
| Portrait 864×1920 | 0.1391 | UNASSESSED (aspect gate) | 0.1194 | REAL | visual_only |
| Landscape 1920×1080 | 0.1672 | UNASSESSED (aspect gate) | 0.1880 | REAL | visual_only |
| Square 1080×1080 | 0.2045 | REAL | 0.2045 | REAL | visual_only |
| Rotated phone (display 1080×1920) | 0.3185 | UNASSESSED (aspect gate) | 0.2508 | REAL | visual_only |

Runtime and peak RSS (same clips, one process per run; model load excluded from per-clip runtime):

| Metric | Before (stretch) | After (letterbox) |
|---|---:|---:|
| Portrait runtime | 1.869 s | 2.022 s |
| Landscape runtime | 1.531 s | 1.557 s |
| Square runtime | 1.286 s | 1.357 s |
| Rotated runtime | 2.472 s | 2.611 s |
| Peak RSS | 1,001,734,144 B (~955 MiB) | 972,865,536 B (~928 MiB) |

The after run was slightly slower per clip (≈1–10%) due to padding/resize work; peak RSS was not higher.

## What is validated

Validated in this checkout:

- MP4/MOV decoding through OpenCV/FFmpeg.
- Portrait, landscape, and square frame geometry.
- Rotated phone video with rotation metadata (display orientation applied).
- Letterbox geometry and normalized tensor shape.
- Verdict consistency: API `overall_verdict`/`verdict_basis`/`verdict_reason` drive the Java banner.
- Unreadable media returns HTTP 200 with `UNASSESSED` and a concrete reason, not fabricated scores.
- Python: `test_layout_and_verdicts.py` and `test_multimodal_api.py`.
- Java: `ForensicStudioRunIdentityTest` (11 tests).

## Blocked: retraining and fusion revalidation

The existing `best_model.pth` and `phase10_fusion.joblib` were trained on the legacy stretched frames. Matching letterbox retraining and fusion revalidation cannot be completed because these inputs are missing:

- `data/extracted/hidf/real/` and `data/extracted/hidf/fake/` — HiDF source videos.
- `data/derived/frames/` — Phase 5 extracted frame corpus.
- `data/cache/features/train_features.pt`, `val_features.pt`, `test_features.pt` — cached EfficientNet features.
- `data/derived/fusion/phase10_multimodal_scores.csv` — per-video visual/physiological scores for fusion fitting and test.
- The user's exact 864×1920 failing clip — not present in the repository or the local Downloads search.

Representative failures:

```
python scripts/extract_hidf_frames.py
# CRITICAL ERROR: Manifest ... or source video directory missing

python scripts/extract_features.py
# Frame files referenced by the manifest are missing

python scripts/train_phase10_fusion.py
# data/derived/fusion/phase10_multimodal_scores.csv not found
```

Because the exact failing clip is unavailable, this change **cannot claim that the clip is now detected**. Letterboxing is implemented and validated for geometry/rotation/verdict behavior, but the current weights still return REAL on the available portrait and rotated real clips. Retraining the visual model on letterboxed HiDF frames and refitting Phase 10 fusion is required before claiming improved deepfake recall.

## Reproduction

```bash
# Python layout/verdict tests
cd ai-engine
python -m pytest tests/test_layout_and_verdicts.py tests/test_multimodal_api.py -q

# Java UI verdict tests
cd ../java-client
mvn -Dtest=ForensicStudioRunIdentityTest,Chapter10ScreenshotsTest test
```

The layout clips can be regenerated from any source video with `ffmpeg` using `scale=...:force_original_aspect_ratio=decrease,pad=...`; rotation was tested with a clip carrying `rotation=-90` metadata.
