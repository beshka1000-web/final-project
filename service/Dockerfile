FROM python:3.12-slim

WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY artifacts ./artifacts

ENV ARTIFACTS_DIR=/srv/artifacts
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request,sys; sys.exit(urllib.request.urlopen('http://localhost:8000/health').status != 200)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
