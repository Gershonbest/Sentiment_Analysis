# Ray base image with CUDA, PyTorch, and Ray preinstalled.
FROM rayproject/ray:latest-gpu

WORKDIR /app

USER root

# Install uv and resolve dependencies from lockfile into a project venv.
RUN pip install --no-cache-dir --upgrade pip uv

# Copy dependency metadata first for better build cache reuse.
COPY pyproject.toml uv.lock /app/
RUN uv sync --frozen --no-dev --no-install-project

# Copy project source.
COPY . /app

# Drop privileges for runtime.
RUN chown -R ray:users /app
USER ray

ENV PYTHONUNBUFFERED=1 \
	PYTHONDONTWRITEBYTECODE=1 \
	PATH="/app/.venv/bin:$PATH" \
	RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0 \
	ML_HOST=0.0.0.0 \
	ML_HTTP_PORT=8000

EXPOSE 8000 8265


CMD ["python", "run.py"]