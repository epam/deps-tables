ARG REPOSITORY_URL=""
FROM ${REPOSITORY_URL:+$REPOSITORY_URL/base/}python:3.12-slim AS python-base


ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/lib" \
    PIP_NO_CACHE_DIR=off \
    PYTHONUNBUFFERED=1 \
    LC_ALL=C \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_PATH=/opt/poetry \
    VENV_PATH=/app/.venv
ENV PATH="$VENV_PATH/bin:$POETRY_PATH/bin:$PATH"

FROM python-base AS base

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        # deps for installing poetry
        curl \
        # deps for building python deps
        build-essential \
        # Install Poetry
        && pip install pip==23.2.1 \
        && curl -sSL https://install.python-poetry.org | POETRY_HOME=$POETRY_PATH python3 - --version 1.8.3 \
        && poetry config virtualenvs.in-project true \
        # cleanup
        && rm -rf /var/lib/apt/lists/*

FROM base AS build

WORKDIR /app/

COPY ./src/pyproject.toml ./src/poetry.lock* /app/
COPY ./vendors /vendors

RUN poetry install --no-interaction --no-ansi --no-root --only main && rm -rf ~/.cache/pypoetry/

FROM build AS develop

RUN poetry install --no-interaction --no-ansi --no-root && rm -rf ~/.cache/pypoetry/

COPY ./etc/deps-tables/models/ /models/
COPY ./src /app

ARG BUILD_HASH
ARG BUILD_TAG
ARG BUILD_DATE

ENV SERVICE_INFO_HASH=$BUILD_HASH \
    SERVICE_INFO_TAG=$BUILD_TAG \
    SERVICE_INFO_DATE=$BUILD_DATE

CMD ["python", "-m", "deps_tables", "serve"]

FROM python-base AS runtime
WORKDIR /app

COPY --from=build $VENV_PATH $VENV_PATH
COPY ./etc/deps-tables/models/ /models/
COPY ./vendors /vendors
COPY ./src /app

ARG BUILD_HASH
ARG BUILD_TAG
ARG BUILD_DATE

ENV SERVICE_INFO_HASH=$BUILD_HASH \
    SERVICE_INFO_TAG=$BUILD_TAG \
    SERVICE_INFO_DATE=$BUILD_DATE

RUN useradd -Ms /bin/bash deps
USER deps

CMD ["python", "-m", "deps_tables", "serve"]
