from transformers import AutoModel, AutoTokenizer

# model_name = "bigcode/starcoder2-7b"
model_name = "deepseek-ai/deepseek-coder-6.7b-base"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
