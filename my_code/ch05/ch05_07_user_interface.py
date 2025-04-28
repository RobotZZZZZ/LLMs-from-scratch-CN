import tiktoken
import torch
import chainlit

# 运行命令: chainlit run ch05_07_user_interface.py
# 使用sys.path添加上级目录
import sys
import os
package_path = os.path.dirname(os.path.dirname(os.getcwd()))
file_path = os.path.join(package_path, "ch05", "06_user_interface")
print(file_path)
sys.path.append(file_path)

from previous_chapters import (
    download_and_load_gpt2,
    generate,
    GPTModel,
    load_weights_into_gpt,
    text_to_token_ids,
    token_ids_to_text,
)

if torch.cuda.is_available():
   device = torch.device("cuda")
elif torch.backends.mps.is_available():
   device = torch.device("mps")
else:
   device = torch.device("cpu")

def get_model_and_tokenizer():
   
   CHOOSE_MODEL = "gpt2-xl (1558M)"

   BASE_CONFIG = {
      "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Context length
        "drop_rate": 0.0,        # Dropout rate
        "qkv_bias": True         # Query-key-value bias
   }

   model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }
   
   model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")

   BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

   settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

   gpt = GPTModel(BASE_CONFIG)
   load_weights_into_gpt(gpt, params)
   gpt.to(device)
   gpt.eval()

   tokenizer = tiktoken.get_encoding("gpt2")

   return tokenizer, gpt, BASE_CONFIG

# 获取模型和分词器
tokenizer, model, model_config = get_model_and_tokenizer()

@chainlit.on_message
async def main(message: chainlit.Message):
   token_ids = generate(
      model=model,
      idx=text_to_token_ids(message.content, tokenizer).to(device),
      max_new_tokens=50,
      context_size=model_config["context_length"],
      top_k=1,
      temperature=0.0
   )

   text = token_ids_to_text(token_ids, tokenizer)

   await chainlit.Message(
      content=f"{text}",
   ).send()