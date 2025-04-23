import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import time
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix

# === Plot font size settings ===
sns.set(font_scale=2)  # 2x larger fonts for seaborn
plt.rcParams.update({'font.size': 18})  # 2x larger fonts for matplotlib

# Start timing ⏱️
start_time = time.time()

# Load dataset
df = pd.read_csv("M:\\dizertatie1\\Phishing_dataset\\test_emails_big.csv", encoding="utf-8")
df = df.dropna(subset=["email_text"])

# Predefined stopwords
stopwords = set([
    "i", "you", "me", "my", "your", "yours", "he", "she", "it", "we", "they", "them", "us",
    "is", "are", "was", "were", "be", "been", "to", "from", "in", "on", "of", "for", "with",
    "at", "by", "a", "an", "the", "and", "or", "but", "if", "this", "that", "as", "so", "do",
    "does", "did", "not", "no", "yes", "all", "am"
])

# Tokenization and cleaning function
def tokenize(text):
    tokens = re.findall(r'\b\w+\b', text.lower())
    return [t for t in tokens if t not in stopwords]

# Separate spam/ham and build word counts
spam_words = Counter()
ham_words = Counter()
spam_total = 0
ham_total = 0

for i, row in df.iterrows():
    tokens = tokenize(row["email_text"])
    if row["email_type"] == "Phishing Email":
        spam_words.update(tokens)
        spam_total += len(tokens)
    else:
        ham_words.update(tokens)
        ham_total += len(tokens)

# Compute word spam probabilities
word_probs = {}
for word in set(spam_words.keys()).union(ham_words.keys()):
    spam_freq = spam_words[word] / spam_total if spam_total > 0 else 0
    ham_freq = ham_words[word] / ham_total if ham_total > 0 else 0
    if spam_freq + ham_freq > 0:
        prob = spam_freq / (spam_freq + ham_freq)
        word_probs[word] = max(0.01, min(0.99, prob))

# 🔥 Plot Top 30 Spam-Indicative Words
top_spammy = sorted(word_probs.items(), key=lambda x: x[1], reverse=True)[:30]
words, probs = zip(*top_spammy)
plt.figure(figsize=(12, 8))
sns.barplot(x=probs, y=words, palette="Reds_r")
plt.title("Top 30 Most Spam-Indicative Words")
plt.xlabel("P(Phishing | Word)")
plt.ylabel("Word")
plt.grid(True)
plt.tight_layout()
plt.show()

# Score each email using top N interesting words
scores = []
for text in df["email_text"]:
    tokens = tokenize(text)
    interesting = [word_probs[t] for t in tokens if t in word_probs]
    interesting = sorted(interesting, key=lambda p: abs(p - 0.5), reverse=True)[:15]

    if interesting:
        product = 1.0
        inv_product = 1.0
        for p in interesting:
            product *= p
            inv_product *= (1 - p)
        spam_score = product / (product + inv_product)
    else:
        spam_score = 0.5

    scores.append(spam_score)

df["bayesian_score"] = scores

# 📊 KDE Plot - Visualizing score distribution
plt.figure(figsize=(10, 5))
sns.kdeplot(df[df['email_type'] == 'Phishing Email']["bayesian_score"], label="Phishing Email", fill=True, color="red")
sns.kdeplot(df[df['email_type'] == 'Safe Email']["bayesian_score"], label="Safe Email", fill=True, color="green")
plt.axvline(x=0.5, linestyle='--', color='black', label='Threshold = 0.5')
plt.title("KDE Plot - Bayesian Spam Score by Email Type")
plt.xlabel("Spam Probability Score")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Classification
threshold = 0.5
df["predicted_label"] = df["bayesian_score"].apply(lambda x: "Phishing Email" if x >= threshold else "Safe Email")

# Evaluation
print("\n📊 Classification Report (Bayesian Score):")
print(classification_report(df["email_type"], df["predicted_label"]))

print("🧮 Confusion Matrix:")
cm = confusion_matrix(df["email_type"], df["predicted_label"], labels=["Phishing Email", "Safe Email"])
print(cm)

# Plot confusion matrix
plt.figure(figsize=(8, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Phishing Email", "Safe Email"],
            yticklabels=["Phishing Email", "Safe Email"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# Save output
df.to_csv("bayesian_only_scores.csv", index=False)
print("\n💾 Scores saved to 'bayesian_only_scores.csv'")

# ⏱️ End timing
end_time = time.time()
elapsed_time = end_time - start_time
print(f"\n⏱️ Total Evaluation Time: {elapsed_time:.3f} seconds")
