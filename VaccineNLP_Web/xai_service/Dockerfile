FROM python:3.10-slim
RUN addgroup --system app && adduser --system --ingroup app appuser
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir llama-cpp-python \
       --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
COPY --chown=appuser:app . .
USER appuser
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8001/',timeout=4)" || exit 1
EXPOSE 8001
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
