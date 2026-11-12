FROM python:3.11-slim AS build
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --prefix=/install .

FROM python:3.11-slim
RUN useradd --create-home --uid 1000 app
COPY --from=build /install /usr/local
WORKDIR /app
COPY --chown=app:app data ./data
COPY --chown=app:app eval/fixtures ./eval/fixtures
USER app
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "legal_rag.api.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
