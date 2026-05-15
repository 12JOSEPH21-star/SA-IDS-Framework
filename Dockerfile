# ── stage 1: dependency layer ─────────────────────────────────────────────────
FROM python:3.11-slim AS deps

WORKDIR /build

COPY requirements-research.txt pyproject.toml ./

RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        torch==2.10.0+cpu \
        gpytorch==1.15.2 && \
    pip install --no-cache-dir \
        pandas \
        xarray \
        "netCDF4<2" \
        "cdsapi>=0.7,<1" \
        matplotlib

# ── stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# copy installed packages from the deps stage
COPY --from=deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# copy source
COPY task_cli/ ./task_cli/
COPY experiment.py models.py pipeline.py policy.py reliability.py benchmark_suite.py ./
COPY framework_config.json sample_weather.csv ./

# install the package itself (no deps, already present)
COPY pyproject.toml LICENSE README.md ./
RUN pip install --no-deps -e .

# default: run the framework on the bundled demo config
ENTRYPOINT ["python", "-m", "task_cli"]
CMD ["framework-run", "--config", "framework_config.json"]
