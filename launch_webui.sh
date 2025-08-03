#!/bin/bash

ENV_DIR=$(dirname "$0")
cd $ENV_DIR
source .venv/bin/activate

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source /usr/local/Ascend/atb-models/set_env.sh
source /usr/local/Ascend/mindie/set_env.sh
source /usr/local/Ascend/mindie/latest/mindie-service/set_env.sh

export DATA_DIR="$ENV_DIR/data"
export CORS_ALLOW_ORIGIN="*"
export HF_ENDPOINT="https://hf-mirror.com"

echo "Open: http://localhost:8080"

open-webui serve
