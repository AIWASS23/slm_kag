FROM python:3.14-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY main.py ./
COPY .streamlit ./.streamlit
COPY src ./src
ARG SLM_KAG_EXTRAS=api,documents
RUN pip install --no-cache-dir ".[${SLM_KAG_EXTRAS}]"

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/data \
    && chown -R appuser:appuser /app
USER appuser

ENV SLM_KAG_DB_PATH=/app/data/slm_kag.db
EXPOSE 8000 8501
CMD ["slm-kag", "serve", "--host", "0.0.0.0", "--port", "8000"]
