import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# === Load and clean dataset ===
df = pd.read_csv("M:\\dizertatie1\\Phishing_dataset\\training_emails.csv")

# Normalize column names
df.columns = df.columns.str.strip().str.lower()
print("🧾 Columns found:", df.columns.tolist())

# Drop the index column if it exists
if 'unnamed: 0' in df.columns:
    df = df.drop(columns=['unnamed: 0'])

# Drop rows with missing or empty values in 'email text' or 'email type'
df = df.dropna(subset=['email text', 'email type'])
df = df[df['email text'].str.strip() != '']
df = df[df['email type'].str.strip() != '']

# Normalize the labels
df['email type'] = df['email type'].apply(
    lambda x: 'Phishing Email' if str(x).strip().lower() == 'phishing email' else 'Safe Email'
)

# Create training prompt text
df['text'] = df.apply(lambda row: f"Email: {row['email text']}\nLabel: {row['email type']}", axis=1)

# ✅ Count valid entries
print(f"✅ Total valid entries for training: {len(df)}")

# === Split into training and validation sets ===
train_texts, val_texts = train_test_split(df[['text']], test_size=0.2, random_state=42)

train_dataset = Dataset.from_pandas(train_texts.reset_index(drop=True))
val_dataset = Dataset.from_pandas(val_texts.reset_index(drop=True))

# === Load tokenizer ===
model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# === Tokenization with labels ===
def tokenize_function(example):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

train_dataset = train_dataset.map(tokenize_function, batched=True).remove_columns(["text"])
val_dataset = val_dataset.map(tokenize_function, batched=True).remove_columns(["text"])

# === Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# === Quantized model config (QLoRA style)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# === Load quantized model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

# === LoRA adapter config
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)

# === Enable LoRA + training settings
model.enable_input_require_grads()
model.gradient_checkpointing_enable()
model.print_trainable_parameters()
model.train()

# === DataLoaders
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=data_collator)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=data_collator)

# === Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# === Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# === Loss tracking
train_losses = []
val_losses = []

# === Training loop
for epoch in range(4):
    print(f"\n🔁 Epoch {epoch + 1}")
    model.train()
    total_train_loss = 0
    for batch in tqdm(train_loader, desc="Training"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        total_train_loss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    print(f"✅ Train Loss: {avg_train_loss:.4f}")

    # === Validation loop
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    print(f"📉 Validation Loss: {avg_val_loss:.4f}")

# === Save model and tokenizer
model.save_pretrained("./output/final_model")
tokenizer.save_pretrained("./output/final_model")
print("\n🎉 Fine-tuning complete! Model saved to './output/final_model'")

# === Plot Loss Curve
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label='Training Loss', marker='o')
plt.plot(val_losses, label='Validation Loss', marker='o')
plt.title("Loss vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_curve.png")  # Save the plot
plt.show()
