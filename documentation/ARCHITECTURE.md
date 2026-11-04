# BIOTrace System Architecture

BIOTrace is engineered as a decoupled, multi-tier forensic system designed for rigorous scientific media authenticity investigation.

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

## 1. Java 21 Client (`java-client/`)

### Key Design Principles:
- **Separation of Concerns**: UI components are decoupled from background communication via an `EventBus` and `JobOrchestrator`.
- **Non-blocking Concurrency**: All network I/O runs asynchronously using `java.net.http.HttpClient` and `CompletableFuture`. Swing Event Dispatch Thread (EDT) is never blocked.
- **Persistence via JDBC**: Analysis history is tracked using SQLite JDBC with schema migration, prepared statements, transactions, and functional stream filtering.
- **Aesthetic Forensic Visuals**: Uses FlatLaf dark theme, emerald/cyan accents, and JFreeChart waveform telemetry.

---

## 2. Python AI Engine (`ai-engine/`)

### Key Design Principles:
- **FastAPI Modern Asynchronous Gateway**: High-throughput REST endpoints with strict Pydantic model validation.
- **Worker Concurrency**: In-memory job scheduler powered by Python's `concurrent.futures.ThreadPoolExecutor` and cooperative cancellation tokens.
- **Media Inspection & Validation**: Uses OpenCV to test container integrity, frame decodability, spatial resolution, and fps prior to analysis.
- **Extensible Pluggable Architecture**: Modular signal extraction abstractions (`BaseSignalExtractor`) allow gradual introduction of physiological algorithms without artificial mock results.
