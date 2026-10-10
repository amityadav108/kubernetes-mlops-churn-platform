FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Create a non-root user.
RUN groupadd --system appuser && \
    useradd --system --gid appuser --home-dir /app appuser

# Install dependencies from the existing local wheelhouse.
COPY requirements.txt .
COPY wheelhouse/ /wheelhouse/

RUN python -m pip install \
    --no-index \
    --find-links=/wheelhouse \
    -r requirements.txt && \
    rm -rf /wheelhouse

# Copy application source and trained model.
COPY src/ /app/src/
COPY models/churn_model.joblib /app/models/churn_model.joblib

# Run without root privileges.
RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

CMD ["uvicorn", "churn_mlops.serving.main:app", "--host", "0.0.0.0", "--port", "8000"]
