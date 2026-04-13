FROM ghcr.io/astral-sh/uv:python3.13-bookworm

RUN adduser --disabled-password agent
USER agent
WORKDIR /home/agent

COPY --chown=agent pyproject.toml uv.lock README.md ./
COPY --chown=agent src src

RUN --mount=type=cache,target=/home/agent/.cache/uv,uid=1000 \
    uv sync --locked

ENTRYPOINT ["uv", "run", "src/purple/server.py"]
CMD ["--host", "0.0.0.0", "--port", "8080"]
EXPOSE 8080
