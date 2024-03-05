from transformers import AutoModel, AutoTokenizer

# model_name = "bigcode/starcoder2-7b"
model_name = "deepseek-ai/deepseek-coder-6.7b-base"
cache_dir = "/data01/home/xiao.cheng/models"
model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
