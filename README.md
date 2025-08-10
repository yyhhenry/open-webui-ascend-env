# open-webui-ascend-env

Open WebUI Environment with Ascend support.

## Quick Start

使用环境：[AutoDL MindIE 1.0.0环境](https://www.autodl.com/docs/huawei_mindie/)

硬件需求：910B2x鲲鹏920 * 2卡

### 基本内容下载

```bash
# 下载证书
yum install ca-certificates
update-ca-trust extract
ln -s /etc/pki/tls/certs/ca-bundle.crt /etc/ssl/certs/ca-certificates.crt

# 下载uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 下载模型权重
pip install modelscope
modelscope download --model PKU-DS-LAB/FairyR1-32B --local_dir ~/autodl-tmp/FairyR1-32B
chmod 750 ~/autodl-tmp/FairyR1-32B/config.json

# 下载运行脚本环境
git clone https://github.com/yyhhenry/open-webui-ascend-env.git
git clone https://github.com/yyhhenry/fairy-r1-wrapper

# 下载全局依赖
pip install "tokenizers==0.21.4" "transformer==4.54.1" "einops==0.8.1"
```

### 补丁

```bash
# 把模型加载选项中的 bfloat16 改成 float16
sed -i 's/"torch_dtype": "bfloat16"/"torch_dtype": "float16"/g' ~/autodl-tmp/FairyR1-32B/config.json

# 修改mindie源码以支持较新的transformers
sed -i 's/from transformers.modeling_utils import shard_checkpoint/from huggingface_hub import split_torch_state_dict_into_shards as shard_checkpoint/g' /usr/local/Ascend/atb-models/atb_llm/models/base/model_utils.py
```

### 启动推理服务

在单独的终端中执行

logs位于: `/usr/local/Ascend/mindie/latest/mindie-llm/logs/`

脚本会将本目录的模型配置复制到 `/usr/local/Ascend/mindie/latest/mindie-service/conf/config.json`

```bash
cd open-webui-ascend-env
bash start-mindie.sh

# 测试接口 1025
curl -X POST -H "Content-type: application/json" -d '{"model": "FairyR1","messages": [{"role": "user", "content": "介绍一下杭州"}],"max_tokens": 128}' http://127.0.0.1:1025/v1/chat/completions
```

### 启动包装器

在单独的终端中执行

包装器会作为一个反向代理，接管所有关于 Fairy R1 的请求并正确触发思维链

如果不使用包装器，Fairy R1 的思维链无法在 Open WebUI 中正确显示

```bash
cd fairy-r1-wrapper
uv run main.py


# 测试接口 1075
curl -N -X POST -H "Content-type: application/json" -d '{"model": "FairyR1", "messages": [{"role": "user", "content": "介绍一下杭州"}], "max_tokens": 128, "stream": true}' http://127.0.0.1:1075/v1/chat/completions
```

### 启动对话界面

```bash
cd open-webui-ascend-env
# 启动WebUI
# 安装uv可通过: curl -LsSf https://astral.sh/uv/install.sh | sh
# 可能会下载一些东西，完成后访问"http://localhost:8080"
# 注册admin@admin.com和admin_password或任意账号
# 管理员面板-设置-外部连接中，设置OpenAI为"http://127.0.0.1:1075/v1"
bash launch_webui.sh
```
