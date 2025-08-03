from importlib import metadata

from tokenizers import Tokenizer
from transformers import AutoTokenizer

print(metadata.version("transformers"))
print(metadata.version("tokenizers"))

tokenizer_core = Tokenizer.from_file("/root/autodl-tmp/FairyR1-32B/tokenizer.json")

tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/FairyR1-32B")

print(tokenizer)
