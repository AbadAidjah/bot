import os
import sys
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model

# Force unbuffered output
print("Starting training script...", flush=True)

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Much smarter model!

# 1. Real Alpaca dataset (use a small subset for faster training)
print("Loading dataset...", flush=True)
dataset = load_dataset("yahma/alpaca-cleaned", split="train[:500]")  # More samples
print(f"Dataset loaded: {len(dataset)} samples", flush=True)

# 2. Load lightweight model
print("Loading model and tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # GPT2 doesn't have a pad token
model = AutoModelForCausalLM.from_pretrained(model_name)
print("Model loaded!", flush=True)

# 3. Format and tokenize dataset
def format_and_tokenize(example):
    text = f"Instruction: {example['instruction']}\nAnswer: {example['output']}"
    tokenized = tokenizer(text, truncation=True, max_length=512, padding="max_length")
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

print("Tokenizing dataset...", flush=True)
dataset = dataset.map(format_and_tokenize, remove_columns=dataset.column_names)
print("Dataset tokenized!", flush=True)

# 4. LoRA (lightweight) - use correct target modules for LLaMA-based models
print("Setting up LoRA...", flush=True)
lora = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # LLaMA uses q_proj/v_proj
    lora_dropout=0.1,
)
model = get_peft_model(model, lora)
print("LoRA configured!", flush=True)

# 5. Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 6. Training arguments (CPU-friendly)
print("Setting up trainer...", flush=True)
args = TrainingArguments(
    output_dir="/app/model",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    logging_steps=5,
    fp16=False,
    save_steps=100,
    save_total_limit=1,
    disable_tqdm=False,
    log_level="info",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
    data_collator=data_collator,
)

print("Starting training...", flush=True)
trainer.train()
print("Training complete!", flush=True)

# 7. Merge LoRA weights and save full model
print("Saving model...", flush=True)
model = model.merge_and_unload()
model.save_pretrained("/app/model")
tokenizer.save_pretrained("/app/model")
print("Model saved to /app/model!", flush=True)
