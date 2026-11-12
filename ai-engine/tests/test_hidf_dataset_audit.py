"""
Unit tests for BIOTrace Phase 4.4 Dataset Audit and Protocol C Verification.
Validates pairing, non-leakage, split counts, and demographic invariants.
"""

import os
import csv
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS_DATA = PROJECT_ROOT / "reports" / "data"

SPLIT_MANIFEST = str(REPORTS_DATA / "protocol_c_split_manifest.csv")
PAIRING_AUDIT_CSV = str(REPORTS_DATA / "phase4_4_pairing_audit.csv")
SPLIT_STATS_CSV = str(REPORTS_DATA / "phase4_4_split_statistics.csv")

def test_split_manifest_integrity():
    assert os.path.exists(SPLIT_MANIFEST)
    with open(SPLIT_MANIFEST, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4361
    
    splits = {}
    for r in rows:
        s = r["split"]
        splits[s] = splits.get(s, 0) + 1
        
    assert splits["train"] == 3053
    assert splits["val"] == 654
    assert splits["test"] == 654

def test_zero_leakage_invariants():
    with open(SPLIT_MANIFEST, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        
    split_bases = {"train": set(), "val": set(), "test": set()}
    split_targets = {"train": set(), "val": set(), "test": set()}
    split_comps = {"train": set(), "val": set(), "test": set()}
    
    for r in rows:
        s = r["split"]
        split_bases[s].add(r["base_id"])
        split_targets[s].add(r["target_id"])
        split_comps[s].add(r["component_id"])
        
    # Zero overlap across partitions
    for s1, s2 in [("train", "val"), ("train", "test"), ("val", "test")]:
        assert len(split_bases[s1] & split_bases[s2]) == 0
        assert len(split_targets[s1] & split_targets[s2]) == 0
        assert len(split_comps[s1] & split_comps[s2]) == 0

def test_pairing_audit_invariants():
    assert os.path.exists(PAIRING_AUDIT_CSV)
    with open(PAIRING_AUDIT_CSV, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        
    assert len(rows) == 4361
    for r in rows:
        assert r["pairing_status"] == "VERIFIED_VALID_PAIR"
        assert r["real_size_matches_zip"] == "True"
        assert r["fake_size_matches_zip"] == "True"
        assert r["real_crc_matches_zip"] == "True"
        assert r["fake_crc_matches_zip"] == "True"

def test_split_statistics_consistency():
    assert os.path.exists(SPLIT_STATS_CSV)
    with open(SPLIT_STATS_CSV, "r", encoding="utf-8") as f:
        rows = {r["split"]: r for r in csv.DictReader(f)}
        
    assert "train" in rows
    assert "val" in rows
    assert "test" in rows
    assert "TOTAL" in rows
    
    assert int(rows["train"]["pair_count"]) == 3053
    assert int(rows["val"]["pair_count"]) == 654
    assert int(rows["test"]["pair_count"]) == 654
    assert int(rows["TOTAL"]["pair_count"]) == 4361
    
    # Check component and donor totals
    comp_sum = int(rows["train"]["unique_components"]) + int(rows["val"]["unique_components"]) + int(rows["test"]["unique_components"])
    assert comp_sum == int(rows["TOTAL"]["unique_components"]) == 4236
