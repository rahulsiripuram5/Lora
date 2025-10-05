import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
import math

# --- 1. Load Model and Tokenizer ---
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
print("--- Base Model and Tokenizer Loaded ---")

# --- 2. Load and Split the Dataset ---
dataset_name = "databricks/databricks-dolly-15k"
dataset = load_dataset(dataset_name, split="train").train_test_split(test_size=0.1, seed=42)

eval_dataset = dataset["test"]

# --- THIS SECTION CONTAINS THE FINAL FIX ---
def create_and_tokenize(batch):
    """
    Formats, tokenizes, and explicitly creates the 'labels' column.
    """
    # Create the full text for each sample in the batch
    full_texts = [
        f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{response}<|eot_id|>"""
        for instruction, response in zip(batch['instruction'], batch['response'])
    ]
    
    # Tokenize the full texts
    tokenized_results = tokenizer(
        full_texts,
        truncation=True,
        max_length=512,
        padding="max_length"
    )

    # *** THE DEFINITIVE FIX ***
    # The Trainer needs a 'labels' column to compute the loss. For Causal LM,
    # we just copy the input_ids. The model handles the rest.
    tokenized_results["labels"] = tokenized_results["input_ids"][:]
    
    return tokenized_results

tokenized_eval_dataset = eval_dataset.map(
    create_and_tokenize,
    batched=True,
    remove_columns=eval_dataset.column_names
)
# --- END OF FIX ---


# --- 3. Set up Trainer and Run Evaluation ---
trainer = Trainer(
    model=model,
    eval_dataset=tokenized_eval_dataset,
)

print("\n--- Starting Baseline Evaluation ---")
results = trainer.evaluate()
print("--- Baseline Evaluation Complete ---")

# --- 4. Print Results ---
print("\n--- Raw Results Dictionary ---")
print(results)

eval_loss = results.get('eval_loss')
if eval_loss is not None:
    perplexity = math.exp(eval_loss)
    print(f"\n--- Baseline Results ---")
    print(f"Loss: {eval_loss:.4f}")
    print(f"Perplexity: {perplexity:.4f}")
else:
    print("\n'eval_loss' not found in results. Cannot calculate perplexity.")