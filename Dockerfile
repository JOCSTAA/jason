# The brain's container (portability rule 3): the same image runs on Windows
# (Docker Desktop, WSL2 backend), on this Mac for testing, and on cloud Linux.
# CPU-only for M0; M1 switches the base to a CUDA image for the LLM server.

FROM python:3.12-slim-bookworm

# uv, pinned to the version that wrote uv.lock.
COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies first, in their own layer: rebuilding after a code change then
# skips the download and install step.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-dev --no-install-project

COPY jason ./jason
COPY config ./config
RUN uv sync --locked --no-dev

# Run as an ordinary user, not root, so a bug in Jason can't modify the image.
RUN useradd --create-home --uid 1000 jason
USER jason

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8765
# config/brain.toml expects the state folder at /state and the token at
# /secrets/link_token; compose.yaml mounts both from the host.
CMD ["jason", "run", "--role", "brain"]
