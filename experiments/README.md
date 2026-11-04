# BIOTrace Experimental Framework & Evaluation Protocols

This directory houses reproducible evaluation protocols, ablation studies, and benchmarking suites.

## Evaluation Metrics

To avoid biased evaluation on imbalanced test sets, BIOTrace will evaluate across:
- **ROC-AUC (Receiver Operating Characteristic - Area Under Curve)**: Primary classification discriminability metric.
- **EER (Equal Error Rate)**: Threshold where False Acceptance Rate equals False Rejection Rate.
- **F1-Score, Precision, and Recall**: Standard precision-recall trade-offs.
- **Cross-Dataset Generalization Gap**: Difference in ROC-AUC when trained on Dataset A and evaluated on unseen Dataset B.
- **Inference Latency & FPS**: Wall-clock runtime benchmarks per processing stage.

## Planned Ablation Studies

1. **Modality Contribution**:
   - rPPG only vs. Kinematics only vs. Multimodal Fusion.
2. **Color Space Comparison**:
   - Green-channel baseline vs. CHROM vs. POS vs. learned neural rPPG representations.
3. **Compression Resilience**:
   - Evaluation on Raw (c0), High Quality (c23), and Low Quality (c40) video compressions.
4. **ROI Spatial Granularity**:
   - Full-face ROI vs. Multi-patch facial grid (forehead, left cheek, right cheek, chin).
