#!/bin/bash

ENV_DIR=$(dirname "$0")
cd $ENV_DIR

export DATA_DIR="$ENV_DIR/data"
export CORS_ALLOW_ORIGIN="*"
export HF_ENDPOINT="https://hf-mirror.com"

echo "Open: http://localhost:8080"

uv run open-webui serve
