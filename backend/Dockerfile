FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
# The persisted vector store exported by the notebook (section 2.7)
COPY data ./data

# Ollama runs on the host, not in the container (see README: "Run with Docker")
ENV OLLAMA_HOST=http://host.docker.internal:11434 \
    VECTOR_STORE_DIR=/app/data/vector_store

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health')" || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
