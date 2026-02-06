FROM python:3.12-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash evaluator

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ src/

RUN pip install --no-cache-dir .

RUN mkdir -p /workspace /output && \
    chown -R evaluator:evaluator /workspace /output

USER evaluator

ENTRYPOINT ["claude-evaluator"]
