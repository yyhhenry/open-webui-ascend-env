#!/bin/bash

ENV_DIR=$(dirname "$0")
cp $ENV_DIR/mindie-config.json /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
chmod 640 /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json

cd /usr/local/Ascend/mindie/latest/mindie-service
source set_env.sh
cd bin
./mindieservice_daemon
