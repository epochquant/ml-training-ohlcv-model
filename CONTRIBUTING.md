# Contributing to EpochQuant Kronos ML Training

Thank you for your interest in contributing to **EpochQuant Kronos ML Training**! We welcome contributions from developers, researchers, and quantitative trading enthusiasts.

## How Can I Contribute?

- **Reporting Bugs:** Submit an issue detailing the bug, environment, and steps to reproduce.
- **Suggesting Enhancements:** Open an issue describing your proposed feature or improvement.
- **Code & Documentation:** Submit a Pull Request (PR) with code cleanups, new features, or documentation improvements.

## Development Workflow

### 1. Fork & Clone

Fork the repository on GitHub and clone your fork locally:

```bash
git clone https://github.com/YOUR-USERNAME/ml-training-ohlcv-model.git
cd ml-training-ohlcv-model
```

### 2. Environment Setup

Create a virtual environment or containerized environment using **Podman** or Docker:

```bash
# Using Python venv
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Or building locally with Podman
podman build -t kronos-ml-training:local .
```

### 3. Code Style & Standards

- **PEP 8 Compliance:** Maintain clean, readable Python code.
- **Type Annotations:** Use type hints wherever applicable.
- **No Hardcoded Secrets:** Never commit GCP credentials, API keys, or project-specific emails. All secrets must be loaded via `.env`.

### 4. Running Security & Lint Checks

Before committing, ensure your changes pass secret scanning:

```bash
# Test local container build with Podman
podman build -t kronos-ml-training:test .

# Run pre-commit gitleaks check if installed
gitleaks detect --verbose
```

### 5. Submitting a Pull Request

1. Create a descriptive feature branch (`git checkout -b feature/amazing-feature`).
2. Commit your changes with a clear commit message (`git commit -m "feat: add support for custom loss function"`).
3. Push to your branch (`git push origin feature/amazing-feature`).
4. Open a **Pull Request** against the `main` branch of `epochquant/ml-training-ohlcv-model`.

## Licensing

By contributing, you agree that your contributions will be licensed under the project's **Apache 2.0 License**.
