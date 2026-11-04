# BIOTrace: Biological Intelligence & Digital Media Forensics Platform

[![Java 21](https://img.shields.io/badge/Java-21-007396?logo=openjdk&logoColor=white)](https://openjdk.org/)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Maven](https://img.shields.io/badge/Maven-3.9%2B-C71A36?logo=apachemaven&logoColor=white)](https://maven.apache.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **BIOTrace** is a research-grade biological intelligence and digital media forensics platform engineered to investigate video authenticity through physiological and temporal consistency signals.

---

## 1. Scientific Vision & Research Direction

The central research hypothesis guiding BIOTrace is:

> *"Can multimodal physiological and temporal facial evidence provide complementary information for detecting manipulated videos?"*

Generative facial manipulation models (such as deepfakes, diffusion reenactments, and face swaps) primarily optimize for spatial texture realism and perceptual appearance. However, real human physiology operates under strict biological constraints that are inherently difficult to synthesize consistently across facial regions and time:

- **Remote Photoplethysmography (rPPG)**: Subcutaneous blood volume pulse (BVP) fluctuations produce periodic, subtle chrominance changes across facial skin. In synthetic media, generative temporal jitter, face-swapping boundaries, and Poisson blending degrade pulse wave synchrony.
- **Eye Blinking & Facial Kinematics**: Natural human blinks adhere to constrained dynamic curves characterized by the Eye Aspect Ratio (EAR). Generative models frequently introduce phase shifts, unnatural durations, or omitted blinks.
- **Multimodal Evidence Fusion**: Under real-world constraints (lossy video compression, lighting variations, head pose changes), single-modality detectors suffer. Fusing orthogonal biological and kinematic evidence provides greater resilience.

---

## 2. System Architecture

BIOTrace is built as a clean, decoupled two-tier architecture:

```
+-------------------------------------------------------------------------+
|                         JAVA 21 DESKTOP CLIENT                          |
|                                                                         |
|  +------------------------+   +-------------------+   +---------------+ |
|  |   Forensic Studio UI   |   | History / Audit   |   | Diagnostics   | |
|  |  (Swing / FlatLaf Dark)|   | (JTable / Filter) |   | (Live Engine) | |
|  +-----------+------------+   +---------+---------+   +-------+-------+ |
|              |                          |                     |         |
|              v                          v                     v         |
|  +--------------------------------------------------------------------+ |
|  |              Application Context & Domain Services                 | |
|  |   * AsyncJobOrchestrator  * EventBus  * SqliteJobRepository (JDBC) | |
|  +----------------------------------+---------------------------------+ |
|                                     |                                   |
|                                     v                                   |
|  +--------------------------------------------------------------------+ |
|  |                 HttpForensicApiClient (java.net.http)              | |
|  +----------------------------------+---------------------------------+ |
+-------------------------------------|-----------------------------------+
                                      | HTTP / REST (JSON)
                                      | localhost:8000
+-------------------------------------|-----------------------------------+
|                                     v                                   |
|  +--------------------------------------------------------------------+ |
|  |                    FASTAPI REST GATEWAY (v1)                       | |
|  |    * POST /api/v1/jobs           * GET /api/v1/jobs/{id}           | |
|  |    * GET /api/v1/health          * POST /api/v1/jobs/{id}/cancel   | |
|  +----------------------------------+---------------------------------+ |
|                                     |                                   |
|                                     v                                   |
|  +--------------------------------------------------------------------+ |
|  |                  PYTHON AI ENGINE (biotrace-engine)                | |
|  |                                                                    | |
|  |  [JobManager & ThreadPool]  --->  [VideoInspector (OpenCV)]        | |
|  |                                                |                   | |
|  |                                                v                   | |
|  |  [Extensible Extractors]               [Validation Pipeline]       | |
|  |  * BaseSignalExtractor                 * Container Integrity       | |
|  |  * (rPPG / CHROM / POS)                * Resolution / FPS Guard    | |
|  |  * (Facial Kinematics / EAR)           * Frame Decodability        | |
|  +--------------------------------------------------------------------+ |
|                        PYTHON 3.11/3.12 / PYTORCH                       |
+-------------------------------------------------------------------------+
```

---

## 3. Technology Stack

### Java 21 Forensic Client (`java-client/`)
- **Runtime**: Java 21 LTS
- **Build System**: Apache Maven 3.9+
- **GUI Framework**: Java Swing with modern **FlatLaf** scientific dark look-and-feel.
- **Waveform Telemetry**: **JFreeChart** with custom dark styling for interactive physiological signal visualization.
- **Persistence**: **JDBC** with SQLite dialect, prepared statements, and transactional safety.
- **Modern Java Idioms**: Java 21 `record` classes, pattern matching, switch expressions, text blocks, and `java.net.http.HttpClient` asynchronous pipelines.
- **Testing**: JUnit 5 test suite.

### Python AI Engine (`ai-engine/`)
- **Runtime**: Python 3.11 / 3.12
- **REST Framework**: **FastAPI** with Pydantic v2 schemas and OpenAPI documentation.
- **Computer Vision & Video Validation**: **OpenCV** (`cv2.VideoCapture`) for container verification and decodability validation.
- **Signal & Machine Learning Foundations**: **NumPy**, **SciPy**, **scikit-learn**, **PyTorch**, and **MediaPipe**.
- **Concurrency**: Thread-safe asynchronous `JobManager` with thread pool execution and cooperative cancellation tokens.
- **Testing**: `pytest` suite with synthetic video decoding fixtures.

---

## 4. Repository Structure

```
BioTrace/
├── .gitignore                    # Comprehensive ignore rules for Java, Python, and datasets
├── README.md                     # Central project documentation
├── java-client/                  # Java 21 Swing Forensic Desktop Application
│   ├── pom.xml                   # Maven dependencies and build configuration
│   ├── src/main/java/org/biotrace/client/
│   │   ├── BioTraceApplication.java         # Main desktop entry point
│   │   ├── core/                            # Config, EventBus, Logging, Exceptions
│   │   ├── model/                           # Domain records, entities, and enums
│   │   ├── service/api/                     # HttpClient and JSON utilities
│   │   ├── service/repository/              # JDBC SQLite repository implementation
│   │   ├── service/orchestration/           # Multithreaded job orchestrator
│   │   └── ui/                              # Theme, Components, and Views
│   └── src/test/java/org/biotrace/client/   # JUnit 5 automated test suite
├── ai-engine/                    # Python FastAPI Backend & AI Foundation
│   ├── pyproject.toml            # Python packaging and pytest configuration
│   ├── requirements.txt          # Python dependencies
│   ├── .env.example              # Environment variables template
│   ├── src/biotrace/
│   │   ├── main.py                          # FastAPI application factory
│   │   ├── core/                            # Settings, Logging, Custom Exceptions
│   │   ├── domain/                          # Pydantic models and Enums
│   │   ├── services/                        # VideoInspector, JobManager, Extractor Stub
│   │   └── api/v1/                          # Health and Job REST endpoints
│   └── tests/                    # Pytest automated test suite
├── research/                     # Theoretical foundations, research hypotheses, literature notes
├── experiments/                  # Evaluation protocols, metrics (ROC-AUC, EER), ablation plans
├── datasets/                     # Guidelines for FaceForensics++, Celeb-DF, UBFC-rPPG
└── documentation/                # Architecture diagrams and REST API specifications
```

---

## 5. Quickstart & Verification

### Step 1: Clone Repository
```bash
git clone https://github.com/SravyaMummana2006/BioTrace.git
cd BioTrace
```

### Step 2: Start the Python AI Engine
```bash
cd ai-engine

# Verify or configure environment
cp .env.example .env

# Run automated tests
python -m pytest tests -v

# Launch the FastAPI service
python -m uvicorn biotrace.main:app --host 127.0.0.1 --port 8000 --reload
```
Interactive API documentation will be available at: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### Step 3: Build & Launch the Java Forensic Client
In a separate terminal:
```bash
cd java-client

# Run automated test suite
mvn clean test

# Launch the Desktop Application
mvn exec:java
```

---

## 6. Research Roadmap & Next Stages

- **Stage 1 (Current)**: Architecture & System Foundation, API Contracts, Video Ingestion & Validation, Asynchronous Job Scheduling, UI Skeleton, Automated Unit Tests.
- **Stage 2**: rPPG Signal Extraction Pipeline (Face landmark detection via MediaPipe, skin segmentation, CHROM & POS temporal filtering, power spectral density estimation).
- **Stage 3**: Temporal Kinematics & Eye-Blink Dynamics (EAR tracking, blink curve asymmetry modeling, micro-expression jitter analysis).
- **Stage 4**: Multimodal Evidence Fusion & Explainable AI (Bayesian/late fusion, confidence weighting, visual explainability maps).
- **Stage 5**: Benchmark Evaluation & Ablation Studies (Cross-dataset evaluation on FaceForensics++ and Celeb-DF, compression degradation analysis).
