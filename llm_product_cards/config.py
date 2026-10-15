"""Project-wide constants: model, token budget, pricing assumptions."""

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# Hard limit on the rendered prompt (system + user + chat markup), tokens
MAX_INPUT_TOKENS = 1500

# Cost model assumptions; adjust for your provider
GPU_HOUR_RUB = 65.0
API_INPUT_RUB_PER_1M = 20.0
API_OUTPUT_RUB_PER_1M = 60.0
