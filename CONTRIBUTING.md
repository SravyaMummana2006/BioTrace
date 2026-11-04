# Contributing to BIOTrace

Thank you for your interest in contributing to **BIOTrace** (Biological Intelligence & Multimodal Media Authenticity). We welcome scientific contributions, performance optimizations, bug reports, and forensic methodology enhancements.

BIOTrace adheres to rigorous scientific standards, strict protocol validation (zero-leakage splits), and reproducible forensic engineering. Please review this guide before submitting changes.

---

## 1. Code of Conduct & Research Ethics

- **Scientific Honesty**: Always report held-out test evaluations objectively. Never report validation or training metrics as generalization performance, and never cherry-pick test cases.
- **Data Protection & Privacy**: Forensic media, facial imagery, and biometric datasets must adhere to ethical research protocols and dataset distribution agreements. Never commit private facial media, API credentials, or raw video corpora to the repository.
- **Forensic Integrity**: BIOTrace provides investigative support and heuristic analysis. Contributions must maintain clear distinction between empirical research models and certified forensic proof.

---

## 2. Development Prerequisites

- **Java**: JDK 21 LTS or higher (JDK 26 tested with Java 21 bytecode compliance).
- **Maven**: Apache Maven 3.9+ (or use the repository script `.\scripts\run_mvn.ps1` on Windows).
- **Python**: Python 3.11 or 3.12 with `pip`.
- **Operating System**: Cross-platform (Windows 11, macOS, Ubuntu 22.04+).

---

## 3. Getting Started & Repository Setup

1. **Fork and Clone**:
   ```bash
   git clone https://github.com/SravyaMummana2006/BioTrace.git
   cd BioTrace
   ```

2. **Python Environment Setup**:
   ```bash
   cd ai-engine
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate

   pip install -r requirements.txt
   ```

3. **Java Client Build**:
   ```bash
   cd ../java-client
   # Using Maven or the Windows helper script:
   mvn clean compile
   # or
   ..\scripts\run_mvn.ps1 -f pom.xml clean compile
   ```

---

## 4. Coding Standards

### Python (`ai-engine/`)
- Follow **PEP 8** style guidelines and maintain full type hinting (`typing` / Python 3.11+ syntax).
- Format docstrings using **Google style**.
- Use **Pydantic v2** models for API contracts and request/response validation.
- All deep learning layers must explicitly handle tensor device placement (`cpu` / `cuda`).
- Write unit tests under `ai-engine/tests/` using **pytest**.

### Java (`java-client/`)
- Adhere to modern **Java 21** idioms:
  - Prefer immutable `record` types for data transfer objects (DTOs) and domain events.
  - Use pattern matching for `instanceof` and switch expressions.
  - Utilize text blocks `"""..."""` for structured queries or JSON templates.
  - Use `java.net.http.HttpClient` for non-blocking HTTP operations.
- UI styling must adhere to `ForensicTheme.java` and **FlatLaf Dark** design conventions.
- Write tests under `java-client/src/test/` using **JUnit 5**.

---

## 5. Running the Test Suites

Before opening a pull request, verify that all test suites pass with zero regressions:

### Python Test Suite (301 Tests)
```bash
cd ai-engine
pytest tests -v
```

### Java Test Suite (43 Tests)
```bash
cd java-client
mvn test
# Or on Windows:
..\scripts\run_mvn.ps1 -f pom.xml test
```

---

## 6. Dataset & Protocol C Integrity

BIOTrace evaluates models on the frozen **Protocol C** manifest (`datasets/protocol_c_manifest.json`):
- **Frozen Manifest SHA-256**: `c15300c15ca78fb58ab16d8f01549534dffb05fb9d75664ac14b2d12c0af06b7`
- Any change to dataset splits, video IDs, or partition assignments invalidates benchmark comparability and will be rejected.
- Test splits must remain strictly held-out until final benchmarking. Never use test data for feature selection, scaling, threshold tuning, or hyperparameter exploration.

---

## 7. Submitting Pull Requests

1. Create a feature branch from `master`:
   ```bash
   git checkout -b feat/your-feature-name
   ```
2. Commit your changes with conventional commit messages:
   - `feat(...)`: New capability or feature
   - `fix(...)`: Bug fix
   - `docs(...)`: Documentation updates
   - `test(...)`: Adding or updating test cases
   - `refactor(...)`: Code cleanup without functional change
3. Ensure all automated tests pass.
4. Submit a Pull Request targeting `master` on `https://github.com/SravyaMummana2006/BioTrace`.
5. Clearly describe the motivation, methodology, test results, and any relevant metrics in the PR description.
