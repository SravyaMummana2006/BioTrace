# BIOTrace REST API Specification (Version 1)

Base URL: `http://127.0.0.1:8000/api/v1`

---

## 1. System Health Check

- **Endpoint**: `GET /health`
- **Description**: Returns server status, hardware capabilities, and version metadata.
- **Response `200 OK`**:
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "uptime_seconds": 124.5,
  "python_version": "3.11.9",
  "torch_version": "2.10.0",
  "cuda_available": false,
  "active_jobs": 0,
  "worker_threads": 4,
  "supported_modalities": ["rppg", "blink", "facial_dynamics", "respiratory"]
}
```

---

## 2. Submit Video Analysis Job

- **Endpoint**: `POST /jobs`
- **Description**: Submits a video file for biological & temporal forensic analysis.
- **Request Body**:
```json
{
  "video_path": "C:\\Forensics\\suspect_video.mp4",
  "requested_modalities": ["rppg", "blink", "facial_dynamics"],
  "sampling_fps": 30.0,
  "parameters": {}
}
```
- **Response `201 Created`**:
```json
{
  "job_id": "job-a1b2c3d4e5f6",
  "status": "PENDING",
  "created_at": "2026-11-03T05:00:00Z",
  "message": "Forensic analysis job submitted successfully and enqueued."
}
```

---

## 3. Query Job Progress & Forensic Verdict

- **Endpoint**: `GET /jobs/{job_id}`
- **Description**: Retrieves current processing stage, progress percentage, and final verdict when complete.
- **Response `200 OK`**:
```json
{
  "job_id": "job-a1b2c3d4e5f6",
  "video_path": "C:\\Forensics\\suspect_video.mp4",
  "status": "COMPLETED",
  "stage": "COMPLETED",
  "progress_percentage": 100.0,
  "created_at": "2026-11-03T05:00:00Z",
  "completed_at": "2026-11-03T05:00:03Z",
  "metadata": {
    "file_path": "C:\\Forensics\\suspect_video.mp4",
    "file_name": "suspect_video.mp4",
    "width": 1920,
    "height": 1080,
    "fps": 30.0,
    "frame_count": 300,
    "duration_seconds": 10.0,
    "codec": "avc1",
    "file_size_bytes": 10485760
  },
  "result": {
    "job_id": "job-a1b2c3d4e5f6",
    "overall_authenticity_score": 0.50,
    "is_manipulated": false,
    "modality_verdicts": [],
    "execution_time_seconds": 3.12,
    "summary_explanation": "Initial pipeline validation succeeded..."
  }
}
```

---

## 4. Cancel Job

- **Endpoint**: `POST /jobs/{job_id}/cancel`
- **Description**: Aborts an in-flight analysis job.
- **Response `200 OK`**:
```json
{
  "job_id": "job-a1b2c3d4e5f6",
  "message": "Analysis job cancelled successfully.",
  "status": "CANCELLED"
}
```
