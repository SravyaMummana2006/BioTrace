"""BIOTrace Phase 8 — Comprehensive Scientific and Statistical Validation Script.

Performs:
1. Exact Benjamini-Hochberg FDR correction across all extracted features.
2. Multicollinearity and redundancy detection.
3. Paired dataset structure and donor/target verification.
4. Confounding factor analysis (file size, compression, bitrate, execution time).
"""

import hashlib
import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

CSV_PATH = "phase8_sample_features.csv"
MANIFEST_PATH = "protocol_c_split_manifest.csv"

df = pd.read_csv(CSV_PATH)
manifest_df = pd.read_csv(MANIFEST_PATH)

print("=" * 80)
print("1. STATISTICAL VALIDATION & BENJAMINI-HOCHBERG FDR CORRECTION")
print("=" * 80)

# Identify numerical features
exclude_cols = [
    "ground_truth_class", "video_frame_count", "video_width", "video_height",
    "rppg_extractor_success", "facial_extractor_success", "spatial_extractor_success",
    "rppg_is_valid_pulse", "spatial_is_measurable", "video_duration_s"
]

candidate_cols = [
    c for c in df.columns
    if df[c].dtype in (np.float64, np.int64)
    and c not in exclude_cols
    and df[c].notnull().sum() >= 20
]

records = []
for col in candidate_cols:
    real_s = df[df["label"] == "real"][col].dropna()
    fake_s = df[df["label"] == "fake"][col].dropna()

    n_r, n_f = len(real_s), len(fake_s)
    if n_r < 5 or n_f < 5:
        continue

    m_r, s_r = float(real_s.mean()), float(real_s.std())
    m_f, s_f = float(fake_s.mean()), float(fake_s.std())

    if s_r < 1e-6 and s_f < 1e-6:
        continue

    # Cohen's d with pooled std
    s_pooled = np.sqrt(((n_r - 1) * s_r**2 + (n_f - 1) * s_f**2) / (n_r + n_f - 2))
    cohen_d = (m_f - m_r) / s_pooled if s_pooled > 1e-7 else 0.0

    # Mann-Whitney U test
    u_stat, p_val_mwu = stats.mannwhitneyu(real_s, fake_s, alternative="two-sided")
    # Welch t-test
    t_stat, p_val_t = stats.ttest_ind(real_s, fake_s, equal_var=False)

    records.append({
        "feature": col,
        "n_real": n_r,
        "n_fake": n_f,
        "real_mean": m_r,
        "real_std": s_r,
        "fake_mean": m_f,
        "fake_std": s_f,
        "real_median": float(real_s.median()),
        "fake_median": float(fake_s.median()),
        "cohen_d": float(cohen_d),
        "p_val_mwu": float(p_val_mwu),
        "p_val_t": float(p_val_t),
        "missing_pct": float(df[col].isnull().mean() * 100),
    })

res = pd.DataFrame(records)
res = res.sort_values(by="p_val_mwu").reset_index(drop=True)
m = len(res)

# Benjamini-Hochberg FDR correction
res["rank"] = range(1, m + 1)
raw_q = (m / res["rank"]) * res["p_val_mwu"]
q_mwu = np.minimum.accumulate(raw_q.values[::-1])[::-1]
res["q_val_mwu"] = np.clip(q_mwu, 0.0, 1.0)

print(f"Total features evaluated: {m}")
print(f"{'Feature':35s} | {'Real Mean':10s} | {'Fake Mean':10s} | {'Cohen d':8s} | {'p (MWU)':10s} | {'q (FDR)':10s}")
print("-" * 92)
for _, r in res.iterrows():
    sig = "***" if r["q_val_mwu"] < 0.01 else ("**" if r["q_val_mwu"] < 0.05 else ("*" if r["q_val_mwu"] < 0.10 else "ns"))
    print(f"{r['feature']:35s} | {r['real_mean']:10.4f} | {r['fake_mean']:10.4f} | {r['cohen_d']:8.3f} | {r['p_val_mwu']:10.4e} | {r['q_val_mwu']:10.4e} {sig}")

print("\n" + "=" * 80)
print("2. MULTICOLLINEARITY & REDUNDANCY AUDIT")
print("=" * 80)
corr_matrix = df[[r["feature"] for r in records]].corr()
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        col_a = corr_matrix.columns[i]
        col_b = corr_matrix.columns[j]
        r_val = corr_matrix.iloc[i, j]
        if abs(r_val) >= 0.80:
            high_corr_pairs.append((col_a, col_b, r_val))

high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
print(f"Pairs with |Pearson r| >= 0.80 (Total {len(high_corr_pairs)}):")
for c1, c2, r_val in high_corr_pairs:
    print(f"  {c1:30s} <-> {c2:30s}: r = {r_val:+.4f}")

print("\n" + "=" * 80)
print("3. DATASET STRUCTURE, PAIRED IDENTITY & CONFOUNDING AUDIT")
print("=" * 80)

# Check manifest pairing
train_manifest = manifest_df[manifest_df["split"] == "train"]
first_100_pairs = train_manifest.iloc[:100]

sample_real_ids = set(df[df["label"] == "real"]["base_id"])
sample_fake_ids = set(df[df["label"] == "fake"]["base_id"])
manifest_base_ids = set(first_100_pairs["base_id"])

print(f"Unique base_ids in real samples: {len(sample_real_ids)}")
print(f"Unique base_ids in fake samples: {len(sample_fake_ids)}")
print(f"Real base_ids match manifest: {sample_real_ids == manifest_base_ids}")
print(f"Fake base_ids match manifest: {sample_fake_ids == manifest_base_ids}")
print(f"Real and fake share EXACT same base_ids (1-to-1 paired): {sample_real_ids == sample_fake_ids}")

# Donor / Target analysis
target_ids = df[df["label"] == "fake"]["target_id"].tolist()
print(f"Unique target_ids in fake samples: {len(set(target_ids))}")
overlap_base_target = sample_real_ids.intersection(set(target_ids))
print(f"Overlap between base_ids and target_ids in sample: {len(overlap_base_target)}")

# Confounder analysis: File size, duration, execution time
print("\nConfounder Checks:")
for conf in ["file_size_bytes", "video_fps", "total_wall_time_s", "rppg_exec_time_s"]:
    r_val = df[df["label"] == "real"][conf].dropna()
    f_val = df[df["label"] == "fake"][conf].dropna()
    u, p = stats.mannwhitneyu(r_val, f_val)
    print(f"  {conf:20s}: Real mean={r_val.mean():10.2f}, Fake mean={f_val.mean():10.2f}, p-val={p:.4e}")

# Save results for artifact writing
validation_summary = {
    "total_features": m,
    "features_p_lt_05": int((res["p_val_mwu"] < 0.05).sum()),
    "features_q_lt_05": int((res["q_val_mwu"] < 0.05).sum()),
    "features_q_lt_10": int((res["q_val_mwu"] < 0.10).sum()),
    "table": res.to_dict(orient="records"),
    "high_corr": [(c1, c2, float(r)) for c1, c2, r in high_corr_pairs],
}

with open("phase8_validation_stats.json", "w") as f:
    json.dump(validation_summary, f, indent=2)

print("\nValidation script finished successfully!")
