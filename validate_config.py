from transformers import AutoConfig

MODEL_ID = "Qwen/Qwen2.5-0.5B"

config = AutoConfig.from_pretrained(
    MODEL_ID,
    trust_remote_code=False,
)

hidden = config.hidden_size
query_heads = config.num_attention_heads
kv_heads = config.num_key_value_heads
head_dim = getattr(config, "head_dim", None) or hidden // query_heads

assert hidden == query_heads * head_dim
assert hidden % query_heads == 0

print(f"Model: {MODEL_ID}")
print(f"Hidden size: {hidden}")
print(f"Intermediate size: {config.intermediate_size}")
print(f"Transformer blocks: {config.num_hidden_layers}")
print(f"Vocabulary size: {config.vocab_size}")
print(f"Query heads: {query_heads}")
print(f"KV heads: {kv_heads}")
print(f"Head dimension: {head_dim}")
print(f"Query projection width: {query_heads * head_dim}")
print(f"Key/value projection width: {kv_heads * head_dim}")
print(f"Tied embeddings: {config.tie_word_embeddings}")
print("Configuration checks passed.")