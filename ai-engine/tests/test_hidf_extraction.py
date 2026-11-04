"""
Unit tests for HiDF Safe Extraction Script (scripts/extract_hidf.py)
Validates ZipSlip defenses, CRC32 computations, staging, and atomic completion for both real and fake datasets.
"""

import os
import sys
import zipfile
import zlib
import pytest
from pathlib import Path

# Ensure workspace root and scripts are importable
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from scripts.extract_hidf import (
    is_safe_path,
    compute_file_crc32,
    compute_file_md5,
    compute_file_sha256,
    load_split_manifest,
    extract_archive,
    EXTRACTION_CONFIG,
)

def test_is_safe_path_valid():
    target_dir = os.path.abspath("data/extracted/hidf/fake")
    assert is_safe_path(target_dir, "c00003_c11507.mp4") is True
    assert is_safe_path(target_dir, "sub/video.mp4") is True

def test_is_safe_path_traversal_attacks():
    target_dir = os.path.abspath("data/extracted/hidf/fake")
    assert is_safe_path(target_dir, "../c00003_c11507.mp4") is False
    assert is_safe_path(target_dir, "../../etc/passwd") is False
    assert is_safe_path(target_dir, "..\\..\\windows\\system32") is False

def test_compute_file_crc32(tmp_path):
    test_file = tmp_path / "test.bin"
    payload = b"BIOTrace Research Protocol C 2026 Test Data Payload"
    test_file.write_bytes(payload)
    
    expected_crc = zlib.crc32(payload) & 0xFFFFFFFF
    calculated_crc = compute_file_crc32(str(test_file))
    assert calculated_crc == expected_crc

def test_compute_hashes(tmp_path):
    test_file = tmp_path / "hash_test.bin"
    payload = b"Cryptographic Integrity Check"
    test_file.write_bytes(payload)
    
    import hashlib
    assert compute_file_md5(str(test_file)) == hashlib.md5(payload).hexdigest()
    assert compute_file_sha256(str(test_file)) == hashlib.sha256(payload).hexdigest()

def test_load_split_manifest(tmp_path):
    manifest_csv = tmp_path / "manifest.csv"
    manifest_csv.write_text(
        "base_id,target_id,component_id,split,race,gender,age\n"
        "c0001,c0002,COMP_01,train,White,Male,Adult\n"
        "c0003,c0004,COMP_02,test,Black,Female,Adult\n",
        encoding="utf-8"
    )
    records_by_base, records_by_pair = load_split_manifest(str(manifest_csv))
    assert len(records_by_base) == 2
    assert records_by_base["c0001"]["split"] == "train"
    assert records_by_pair[("c0003", "c0004")]["split"] == "test"

def test_mock_safe_real_extraction(tmp_path, monkeypatch):
    """Verifies staging, validation, and atomic promotion logic on a mock real archive."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    extract_dir = tmp_path / "extracted"
    extract_dir.mkdir()
    
    # Create mock zip archive
    archive_path = raw_dir / "Real-vid.zip"
    members = {
        "m001.mp4": b"video_data_1",
        "m002.mp4": b"video_data_2_long",
    }
    
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in members.items():
            zf.writestr(f"Real-vid/{name}", data)
            
    # Mock split manifest
    split_csv = tmp_path / "mock_split.csv"
    split_csv.write_text(
        "base_id,target_id,component_id,split,race,gender,age\n"
        "m001,t001,C1,train,White,Male,Adult\n"
        "m002,t002,C2,test,Asian,Female,Adult\n",
        encoding="utf-8"
    )
    
    import hashlib
    with open(archive_path, "rb") as f:
        data = f.read()
        mock_size = len(data)
        mock_md5 = hashlib.md5(data).hexdigest()
        
    mock_config = {
        "real": {
            "archive_name": "Real-vid.zip",
            "dest_subdir": "mock_real",
            "expected_files": 2,
            "expected_size": mock_size,
            "expected_md5": mock_md5,
            "description": "Mock authentic base videos",
        }
    }
    monkeypatch.setattr("scripts.extract_hidf.EXTRACTION_CONFIG", mock_config)
    
    manifest_rows = extract_archive("real", str(raw_dir), str(extract_dir), str(split_csv))
    
    assert len(manifest_rows) == 2
    final_dest = extract_dir / "mock_real"
    assert final_dest.exists()
    assert (final_dest / "m001.mp4").read_bytes() == b"video_data_1"
    assert (final_dest / "m002.mp4").read_bytes() == b"video_data_2_long"
    
    staging_dir = extract_dir / ".mock_real_staging"
    assert not staging_dir.exists()

def test_mock_safe_fake_extraction(tmp_path, monkeypatch):
    """Verifies staging, validation, and atomic promotion logic on a mock fake archive."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    extract_dir = tmp_path / "extracted"
    extract_dir.mkdir()
    
    archive_path = raw_dir / "Fake-vid.zip"
    members = {
        "m001_t001.mp4": b"fake_data_1",
        "m002_t002.mp4": b"fake_data_2_deepfake",
    }
    
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in members.items():
            zf.writestr(f"Fake-vid/{name}", data)
            
    split_csv = tmp_path / "mock_split.csv"
    split_csv.write_text(
        "base_id,target_id,component_id,split,race,gender,age\n"
        "m001,t001,C1,train,White,Male,Adult\n"
        "m002,t002,C2,test,Asian,Female,Adult\n",
        encoding="utf-8"
    )
    
    import hashlib
    with open(archive_path, "rb") as f:
        data = f.read()
        mock_size = len(data)
        mock_md5 = hashlib.md5(data).hexdigest()
        
    mock_config = {
        "fake": {
            "archive_name": "Fake-vid.zip",
            "dest_subdir": "mock_fake",
            "expected_files": 2,
            "expected_size": mock_size,
            "expected_md5": mock_md5,
            "description": "Mock deepfake derivative videos",
        }
    }
    monkeypatch.setattr("scripts.extract_hidf.EXTRACTION_CONFIG", mock_config)
    
    manifest_rows = extract_archive("fake", str(raw_dir), str(extract_dir), str(split_csv))
    
    assert len(manifest_rows) == 2
    final_dest = extract_dir / "mock_fake"
    assert final_dest.exists()
    assert (final_dest / "m001_t001.mp4").read_bytes() == b"fake_data_1"
    assert (final_dest / "m002_t002.mp4").read_bytes() == b"fake_data_2_deepfake"
    
    staging_dir = extract_dir / ".mock_fake_staging"
    assert not staging_dir.exists()
