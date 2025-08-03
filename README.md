# open-webui-ascend-env

Open WebUI Environment with Ascend support.

## Usage (Draft)

```bash
# 补充不正常的证书
yum install ca-certificates
update-ca-trust extract
ln -s /etc/pki/tls/certs/ca-bundle.crt /etc/ssl/certs/ca-certificates.crt

# 下载所需的调试和正式模型权重
pip install modelscopemodelscope download --model PKU-DS-LAB/FairyR1-32B --local_dir ~/autodl-tmp/FairyR1-32B
chmod 750 ~/autodl-tmp/FairyR1-32B/config.json
# 或用于测试的小模型
modelscope download --model Qwen/Qwen2.5-0.5B-Instruct --local_dir ~/autodl-tmp/Qwen2.5-0.5B
chmod 750 ~/autodl-tmp/Qwen2.5-0.5B/config.json

# 【重要步骤】你需要把bfloat16改成float16


# 启动
pip install "tokenizers==0.21.4" "transformer==4.54.1" "einops==0.8.1"
git clone https://github.com/yyhhenry/open-webui-ascend-env.git
cd open-webui-ascend-env

# 【重要步骤】修改mindie源码
# 为了加载FairyR1需要比较新的transformers
# 但是MindIE的代码原本依赖比较旧的transformers
# 建议进行以下修改
# Filepath: /usr/local/Ascend/atb-models/atb_llm/models/base/model_utils.py
# -- from transformers.modeling_utils import shard_checkpoint
# ++ from huggingface_hub import split_torch_state_dict_into_shards as shard_checkpoint

# 启动推理服务
# logs位于: /usr/local/Ascend/mindie/latest/mindie-llm/logs/
bash start-mindie.sh

# 测试接口
curl -H "Accept: application/json" -H "Content-type: application/json"  -X POST -d '{"model": "FairyR1","messages": [{"role": "user", "content": "介绍一下杭州"}],"max_tokens": 128}' http://127.0.0.1:1025/v1/chat/completions

# 启动WebUI
# 安装uv可通过: curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
bash launch_webui.sh
```
