"""Comprehensive analysis of Phase 8 physiological feature feasibility results."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

CSV_PATH = "phase8_sample_features.csv"
df = pd.read_csv(CSV_PATH)

print("=" * 80)
print("BIOTrace Phase 8 — Comprehensive Feasibility Feature Analysis")
print("=" * 80)

# 1. Dataset Overview
n_total = len(df)
n_real = (df["label"] == "real").sum()
n_fake = (df["label"] == "fake").sum()
print(f"Sample Size: {n_total} videos (Real: {n_real}, Fake: {n_fake})")
print(f"Duration: mean={df['video_duration_s'].mean():.2f}s, min={df['video_duration_s'].min():.2f}s, max={df['video_duration_s'].max():.2f}s")
print(f"FPS: mean={df['video_fps'].mean():.2f}, Frame count: mean={df['video_frame_count'].mean():.1f}")

# 2. Extractor Execution Success Rates
print("\n" + "-" * 40)
print("1. EXTRACTOR PIPELINE EXECUTION SUCCESS RATES")
print("-" * 40)
for mod in ["rppg_extractor_success", "facial_extractor_success", "spatial_extractor_success"]:
    tot = df[mod].mean() * 100
    r_val = df[df["label"] == "real"][mod].mean() * 100
    f_val = df[df["label"] == "fake"][mod].mean() * 100
    print(f"{mod:30s}: Total={tot:5.1f}% | Real={r_val:5.1f}% | Fake={f_val:5.1f}%")

# 3. Clinical / Forensic Quality Gating Rates
print("\n" + "-" * 40)
print("2. RESEARCH MODULE GATING OUTCOMES (Standard Thresholds)")
print("-" * 40)
print("rPPG Quality Grades:")
print(df["rppg_quality_grade"].value_counts(dropna=False))

print("\nrPPG Gated Heart Rate Valid (is_valid_pulse):")
print(f"  Valid: {df['rppg_is_valid_pulse'].sum()} / {n_total} ({df['rppg_is_valid_pulse'].mean()*100:.1f}%)")
print(f"  Real valid: {df[df['label']=='real']['rppg_is_valid_pulse'].sum()}% | Fake valid: {df[df['label']=='fake']['rppg_is_valid_pulse'].sum()}%")

print("\nrPPG Quality Flags Distribution:")
all_flags = []
for f_str in df["rppg_quality_flags"].dropna():
    for f in f_str.split("; "):
        if f.strip():
            all_flags.append(f.strip())
flags_series = pd.Series(all_flags).value_counts()
print(flags_series)

print("\nFacial Dynamics Calibration States:")
print(df["blink_calibration_state"].value_counts(dropna=False))

print("\nSpatial Consistency Global States:")
print(df["spatial_global_state"].value_counts(dropna=False))

# 4. Missing Value Analysis
print("\n" + "-" * 40)
print("3. MISSING VALUE PROFILE")
print("-" * 40)
missing_pct = df.isnull().mean() * 100

high_missing = missing_pct[missing_pct > 50].sort_values(ascending=False)
mod_missing = missing_pct[(missing_pct > 20) & (missing_pct <= 50)].sort_values(ascending=False)
low_missing = missing_pct[(missing_pct > 0) & (missing_pct <= 20)].sort_values(ascending=False)
zero_missing = missing_pct[missing_pct == 0]

print(f"Features with > 50% missing values ({len(high_missing)}):")
for k, v in high_missing.items():
    print(f"  {k:40s}: {v:5.1f}% missing")

print(f"\nFeatures with 20-50% missing values ({len(mod_missing)}):")
for k, v in mod_missing.items():
    print(f"  {k:40s}: {v:5.1f}% missing")

print(f"\nFeatures with 0-20% missing values ({len(low_missing)}):")
for k, v in low_missing.items():
    print(f"  {k:40s}: {v:5.1f}% missing")

print(f"\nFeatures with 0% missing values ({len(zero_missing)}):")
print(f"  Total: {len(zero_missing)} features (Container, Video, Metadata, Landmark & Pose Statistics)")

# 5. Discriminative Power Analysis (Real vs. Fake)
print("\n" + "-" * 40)
print("4. FEATURE DISCRIMINABILITY ANALYSIS (Real vs Fake)")
print("-" * 40)

# Identify numerical features with < 30% missingness
candidate_features = [
    col for col in df.columns
    if df[col].dtype in (np.float64, np.int64)
    and missing_pct[col] < 30
    and col not in ["ground_truth_class", "video_frame_count", "video_width", "video_height",
                    "rppg_extractor_success", "facial_extractor_success", "spatial_extractor_success"]
]

results = []
for feat in candidate_features:
    real_vals = df[df["label"] == "real"][feat].dropna()
    fake_vals = df[df["label"] == "fake"][feat].dropna()

    if len(real_vals) < 10 or len(fake_vals) < 10:
        continue

    r_mean, r_std = real_vals.mean(), real_vals.std()
    f_mean, f_std = fake_vals.mean(), fake_vals.std()
    r_med, r_iqr = real_vals.median(), real_vals.quantile(0.75) - real_vals.quantile(0.25)
    f_med, f_iqr = fake_vals.median(), fake_vals.quantile(0.75) - fake_vals.quantile(0.25)

    # Variance check
    pooled_var = ((len(real_vals) - 1) * r_std**2 + (len(fake_vals) - 1) * f_std**2) / (len(real_vals) + len(fake_vals) - 2)
    pooled_std = np.sqrt(pooled_var) if pooled_var > 0 else 1e-9

    # Cohen's d effect size
    cohen_d = (f_mean - r_mean) / pooled_std if pooled_std > 1e-7 else 0.0

    # Mann-Whitney U test (non-parametric)
    try:
        u_stat, p_val = stats.mannwhitneyu(real_vals, fake_vals, alternative="two-sided")
    except Exception:
        p_val = 1.0

    # Near-zero variance check
    is_nzv = (r_std < 1e-4 and f_std < 1e-4)

    results.append({
        "feature": feat,
        "real_mean": r_mean,
        "real_std": r_std,
        "fake_mean": f_mean,
        "fake_std": f_std,
        "real_median": r_med,
        "fake_median": f_med,
        "cohen_d": cohen_d,
        "p_val": p_val,
        "is_nzv": is_nzv,
        "missing_pct": missing_pct[feat],
    })

res_df = pd.DataFrame(results).sort_values(by="p_val")
print("\nTop 15 Most Discriminative Features (Ranked by Mann-Whitney p-value):")
print(f"{'Feature':35s} | {'Real Mean':10s} | {'Fake Mean':10s} | {'Cohen d':8s} | {'p-value':10s}")
print("-" * 80)
for _, r in res_df.head(15).iterrows():
    sig = "***" if r["p_val"] < 0.001 else ("**" if r["p_val"] < 0.01 else ("*" if r["p_val"] < 0.05 else "ns"))
    print(f"{r['feature']:35s} | {r['real_mean']:10.4f} | {r['fake_mean']:10.4f} | {r['cohen_d']:8.3f} | {r['p_val']:9.2e} {sig}")

print("\nNear-Zero Variance Features:")
nzv_df = res_df[res_df["is_nzv"]]
if len(nzv_df) == 0:
    print("  None detected among candidate features.")
else:
    for _, r in nzv_df.iterrows():
        print(f"  {r['feature']}")

# 6. Computational Feasibility & Timing Profiling
print("\n" + "-" * 40)
print("5. COMPUTATIONAL FEASIBILITY & RUNTIME BREAKDOWN")
print("-" * 40)
t_tot_mean = df["total_wall_time_s"].mean()
t_tot_std = df["total_wall_time_s"].std()
t_rppg_mean = df["rppg_exec_time_s"].mean()
t_facial_mean = df["facial_exec_time_s"].mean()
t_spatial_mean = df["spatial_exec_time_s"].mean()

print(f"Total time per video : {t_tot_mean:.3f}s +/- {t_tot_std:.3f}s")
print(f"  - rPPG extraction  : {t_rppg_mean:.3f}s ({t_rppg_mean/t_tot_mean*100:.1f}%)")
print(f"  - Facial dynamics  : {t_facial_mean:.3f}s ({t_facial_mean/t_tot_mean*100:.1f}%)")
print(f"  - Spatial consist. : {t_spatial_mean:.3f}s ({t_spatial_mean/t_tot_mean*100:.1f}%)")

full_dataset_n = 8722
proj_hours_single = (full_dataset_n * t_tot_mean) / 3600.0
proj_hours_4core = proj_hours_single / 4.0
proj_hours_8core = proj_hours_single / 8.0

print(f"\nFull HiDF Dataset Projection (8,722 videos):")
print(f"  - Single process   : {proj_hours_single:.2f} hours ({proj_hours_single*60:.1f} minutes)")
print(f"  - 4 parallel cores : {proj_hours_4core:.2f} hours ({proj_hours_4core*60:.1f} minutes)")
print(f"  - 8 parallel cores : {proj_hours_8core:.2f} hours ({proj_hours_8core*60:.1f} minutes)")

# Save detailed analysis JSON for report inclusion
analysis_dict = {
    "total_videos": n_total,
    "real_videos": n_real,
    "fake_videos": n_fake,
    "duration_mean": float(df["video_duration_s"].mean()),
    "fps_mean": float(df["video_fps"].mean()),
    "rppg_success_pct": float(df["rppg_extractor_success"].mean() * 100),
    "facial_success_pct": float(df["facial_extractor_success"].mean() * 100),
    "spatial_success_pct": float(df["spatial_extractor_success"].mean() * 100),
    "rppg_valid_pulse_pct": float(df["rppg_is_valid_pulse"].mean() * 100),
    "spatial_measurable_pct": float(df["spatial_is_measurable"].mean() * 100),
    "timing": {
        "mean_wall_time_s": float(t_tot_mean),
        "mean_rppg_s": float(t_rppg_mean),
        "mean_facial_s": float(t_facial_mean),
        "mean_spatial_s": float(t_spatial_mean),
        "projected_hours_single": float(proj_hours_single),
        "projected_hours_4core": float(proj_hours_4core),
    },
    "top_discriminative_features": res_df.head(20).to_dict(orient="records"),
    "high_missing_features": high_missing.to_dict(),
    "low_missing_features": low_missing.to_dict(),
    "quality_flags": flags_series.to_dict(),
}

def convert_to_serializable(obj):
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    return str(obj)

with open("phase8_feasibility_metrics.json", "w") as f:
    json.dump(analysis_dict, f, indent=2, default=convert_to_serializable)

print("\nAnalysis complete! Metrics saved to phase8_feasibility_metrics.json")
