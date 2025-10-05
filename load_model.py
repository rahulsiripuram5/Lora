import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Make sure you are logged in to Hugging Face for this to work
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

print("--- Starting Model and Tokenizer Loading ---")

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
print("Tokenizer loaded successfully.")

# Load the model
# device_map="auto" will automatically place the model on your A100
# torch_dtype=torch.bfloat16 is optimal for A100 GPUs
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

print("Model loaded successfully onto GPU.")

# Verify the model is on the GPU and get memory footprint
device = model.device
memory_footprint_gb = model.get_memory_footprint() / (1024**3)
print(f"Model is on device: {device}")
print(f"Model memory footprint: {memory_footprint_gb:.2f} GB")

print("--- Smoke test successful! ---")