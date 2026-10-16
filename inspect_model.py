
from transformers import AutoConfig

MODEL_ID = "Qwen/Qwen2.5-0.5B"

config = AutoConfig.from_pretrained(
    MODEL_ID,
    trust_remote_code=False,
)

fields = [
    "model_type",
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "max_position_embeddings",
    "tie_word_embeddings",
]

print(f"\nModel: {MODEL_ID}")
print("-" * 45)

for field in fields:
    print(f"{field:30s}: {getattr(config, field, 'N/A')}")

print("\nConfiguration loaded successfully.")
