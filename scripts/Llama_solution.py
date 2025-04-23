import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)
import re
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import seaborn as sns

# === Plot font size settings ===
sns.set(font_scale=2)  # 2x larger fonts for seaborn
plt.rcParams.update({'font.size': 18})  # 2x larger fonts for matplotlib

# === Load the fine-tuned model and tokenizer ===
model_path = "./output/final_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
model.eval().to("cuda" if torch.cuda.is_available() else "cpu")

# === Load test set ===
df = pd.read_csv("M:\\dizertatie1\\Phishing_dataset\\test.csv")
df.columns = df.columns.str.strip().str.lower()
df['email_type'] = df['email_type'].apply(lambda x: 1 if str(x).strip().lower() == 'phishing email' else 0)

# === Clean email text
def clean_text(text):
    return re.sub(r'\s+', ' ', str(text))[:1000].strip()

df['email_text'] = df['email_text'].apply(clean_text)

# === Prompt builder
def build_prompt(email):
    return f"""Below is an email followed by a label.

Email: {email}
Label:"""

# === Extract model label
def extract_label(output_text):
    output_text = output_text.lower()
    if "phishing email" in output_text:
        return 1
    elif "safe email" in output_text:
        return 0
    elif "phishing" in output_text:
        return 1
    elif "safe" in output_text:
        return 0
    return -1  # Unknown

# === Inference
predictions = []
true_labels = df["email_type"].tolist()
probs = []

print("\n🚀 Classifying test emails...\n")
start_time = time.time()

for i, row in tqdm(df.iterrows(), total=len(df)):
    prompt = build_prompt(row["email_text"])
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_length=inputs["input_ids"].shape[1] + 20,
        do_sample=False,
        temperature=0.0,
        pad_token_id=tokenizer.pad_token_id
    )

    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    pred = extract_label(decoded_output)
    predictions.append(pred)
    probs.append(1.0 if pred == 1 else 0.0)

    print(f"\n📧 Email {i+1}")
    print(f"Text: {row['email_text'][:100]}...")
    print(f"✅ Prediction: {'Phishing Email' if pred == 1 else 'Safe Email'}")
    print(f"🧠 LLaMA Output: {decoded_output.strip()}")
    print("-" * 100)

# Fallback for unknowns
predictions = [p if p in [0, 1] else 0 for p in predictions]

# === Evaluation
end_time = time.time()
elapsed_time = end_time - start_time

accuracy = accuracy_score(true_labels, predictions)
report = classification_report(true_labels, predictions, target_names=["Safe Email", "Phishing Email"])
conf_matrix = confusion_matrix(true_labels, predictions)

print("\n📊 Classification Report:")
print(report)

print("\n🧮 Confusion Matrix:")
print(conf_matrix)

print(f"\n🎯 Accuracy: {accuracy:.4f}")
print(f"⏱️ Total Evaluation Time: {elapsed_time:.2f} seconds")

# === Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Safe Email", "Phishing Email"],
            yticklabels=["Safe Email", "Phishing Email"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("📊 Confusion Matrix")
plt.tight_layout()
plt.show()

# === ROC Curve
fpr, tpr, _ = roc_curve(true_labels, probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('🧪 ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()
