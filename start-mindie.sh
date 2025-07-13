#!/bin/bash

ENV_DIR=$(dirname "$0")
cp $ENV_DIR/mindie-config.json /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source /usr/local/Ascend/atb-models/set_env.sh
source /usr/local/Ascend/mindie/set_env.sh
source /usr/local/Ascend/mindie/latest/mindie-service/set_env.sh

cd /usr/local/Ascend/mindie/latest/mindie-service/bin
./mindieservice_daemon
