FROM python:3.12-slim

# Non-root user: if a dependency inside the container is compromised,
# the attacker must not get root (Docker best practice)
RUN addgroup --system app && adduser --system --ingroup app app

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

# memory_home mount point (identity.md): readable by the app user;
# a fresh volume inherits this ownership on first use
RUN mkdir -p /data/memory && chown app:app /data/memory

USER app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
