#!/bin/bash

ENV_DIR=$(dirname "$0")
CONFIG_FILE=/usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
if ! [ -f "$CONFIG_FILE.bak" ]; then
    echo Backup config file
    cp "$CONFIG_FILE" "$CONFIG_FILE.bak"
fi
cp $ENV_DIR/mindie_config-FairyR1.json $CONFIG_FILE

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source /usr/local/Ascend/atb-models/set_env.sh
source /usr/local/Ascend/mindie/set_env.sh
source /usr/local/Ascend/mindie/latest/mindie-service/set_env.sh

cd /usr/local/Ascend/mindie/latest/mindie-service/bin
./mindieservice_daemon
