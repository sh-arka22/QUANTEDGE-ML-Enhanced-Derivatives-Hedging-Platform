# Day 14 — Docker, CI/CD, Deployment

## Files Created

| File | Purpose |
|------|---------|
| `Dockerfile` | Multi-stage build: Python 3.11-slim, compile deps in builder, copy to runtime |
| `.dockerignore` | Excludes tests, docs, git, venv, pycache |
| `.github/workflows/ci.yml` | GitHub Actions: pytest on push/PR to main |

## Dockerfile

Multi-stage build for minimal image size:
- **Builder stage**: installs gcc/g++ for compiled packages (numpy, scipy, etc.)
- **Runtime stage**: copies only installed packages + source code
- Healthcheck on Streamlit's `/_stcore/health` endpoint
- Runs headless on port 8501

## CI/CD Pipeline

```yaml
Trigger: push to main, pull_request to main
Job: test on ubuntu-latest, Python 3.11
Steps: checkout -> setup python -> install deps -> pytest tests/ -v
```

Streamlit Community Cloud auto-deploys from main — no separate deploy job needed.

## Docker Usage

```bash
# Build
docker build -t quant-platform .

# Run
docker run -p 8501:8501 quant-platform

# Run with FRED key
docker run -p 8501:8501 -e FRED_API_KEY=your_key quant-platform
```
