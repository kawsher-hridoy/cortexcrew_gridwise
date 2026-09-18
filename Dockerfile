FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 \
    PORT=8000

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY schemas.py solve.py llm.py app.py run_cases.py public_cases.json ./

# No credentials are baked in; AZURE_AI_API_KEY is supplied at runtime.
RUN useradd --create-home --uid 10001 gridwise
USER gridwise

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

CMD ["sh", "-c", "uvicorn app:app --host ${HOST} --port ${PORT}"]
