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
print("--- Model and Tokenizer Loaded ---")
print("--- Full Fine-tuning Setup: Training all parameters ---")

# --- 3. Load and Prepare the Dataset ---
dataset_name = "databricks/databricks-dolly-15k"
dataset = load_dataset(dataset_name, split="train").train_test_split(test_size=0.1, seed=42)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

def create_and_tokenize(batch):
    full_texts = [
        f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{response}<|eot_id|>"""
        for instruction, response in zip(batch['instruction'], batch['response'])
    ]
    tokenized_results = tokenizer(full_texts, truncation=True, max_length=512, padding="max_length")
    tokenized_results["labels"] = tokenized_results["input_ids"][:]
    return tokenized_results

tokenized_train_dataset = train_dataset.map(create_and_tokenize, batched=True, remove_columns=train_dataset.column_names)
tokenized_eval_dataset = eval_dataset.map(create_and_tokenize, batched=True, remove_columns=eval_dataset.column_names)

# --- 4. Set up Trainer and Start Training ---
training_args = TrainingArguments(
    output_dir="./full-llama3-8b-finetuned",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    num_train_epochs=1,
    learning_rate=1e-5,
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=50,
    save_steps=50,
    # *** THIS IS THE KEY CHANGE ***
    save_total_limit=1, # Only keep the single best checkpoint
    load_best_model_at_end=True,
    optim="paged_adamw_8bit"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
)

print("\n--- Starting Full Fine-tuning ---")
trainer.train()
print("--- Full Fine-tuning Complete ---")

# --- 5. Save the final best model ---
final_model_dir = "./full-llama3-8b-finetuned/final_model"
trainer.save_model(final_model_dir)
print(f"--- Best Full model saved to {final_model_dir} ---")