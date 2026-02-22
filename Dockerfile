FROM ubuntu:noble

RUN rm -f /etc/apt/apt.conf.d/docker-clean; \
    echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt update &&\
    apt full-upgrade -y

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt install -y curl bash git

USER ubuntu

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

RUN --mount=type=bind,source=./harvest-training-data.py,target=/script.py \
    bash -eu -o pipefail -x -c 'source "$HOME"/.local/bin/env ; uv run /script.py --help'

RUN --mount=type=bind,source=./git-smart-commit,target=/script.py \
    bash -eu -o pipefail -x -c 'source "$HOME"/.local/bin/env ; uv run /script.py --help'
