FROM ghcr.io/astral-sh/uv:latest AS uv
FROM python:3.12-slim

# Install uv
COPY --from=uv /uv /usr/local/bin/uv

WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock ./
COPY src/ src/

# Install dependencies
RUN uv sync --frozen --no-dev --extra protrider

ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT ["protadjust"]
CMD ["--help"]
