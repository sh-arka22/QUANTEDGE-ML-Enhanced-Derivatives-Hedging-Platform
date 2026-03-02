FROM python:3.11-slim AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /install /usr/local
COPY src/ ./src/
COPY .streamlit/ ./.streamlit/
EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1
CMD ["streamlit", "run", "src/ui/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
