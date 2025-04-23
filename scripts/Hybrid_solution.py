import time
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import re
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
import numpy as np
import logging

# === Plot font size settings ===
sns.set(font_scale=2.5)  # 2x larger fonts for seaborn
plt.rcParams.update({'font.size': 18})  # 2x larger fonts for matplotlib

# === Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# === Global Timer Start
total_start = time.time()

# === Load dataset
df = pd.read_csv("M:\\dizertatie1\\Phishing_dataset\\test_emails_big.csv", encoding="utf-8")
df = df.dropna(subset=["email_text"])

# === Load Model Timer
load_start = time.time()
model_name = "./output/final_model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=False)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
load_time = time.time() - load_start

# === Evaluation Timer Start
eval_start = time.time()

# === Stopwords
stopwords = set([
    "i", "you", "me", "my", "your", "yours", "he", "she", "it", "we", "they", "them", "us",
    "is", "are", "was", "were", "be", "been", "to", "from", "in", "on", "of", "for", "with",
    "at", "by", "a", "an", "the", "and", "or", "but", "if", "this", "that", "as", "so", "do",
    "does", "did", "not", "no", "yes", "s", "x", "d", "all", "am", "n", "p", "v", "e", "o",
    "t", "st", "l", "j", "ex", "mm", "begin_of_text", "m", "w", "f", "c", "h", "b", "r", "ou", "k", "g", "u"
])

# === Token categories and patterns
token_categories = {
    "Finance": {"credit", "card", "account", "bank", "money", "loanofficer", "transaction", "wallet", "reward","skyrocket","free","tax"},
    "Adult Content": {"sex", "xxx", "horny", "erection", "penis", "enlarge", "actress", "hardcore", "adult", "content", "sexual", "cock", "cocks", "slut"},
    "Urgency": {"hurry", "now", "limited", "urgent","limited","time"},
    "Tech Scam": {"link", "login", "url", "access", "password", "selected", "account", "click", "here", "log", "0xff"},
    "Health Spam": {"pills", "viagra", "prescription", "pharmacy", "diabetic", "disfunction","depressed","stamina","diagnostic"},
    "Security": {"weapons", "camera", "secure","gun","burst","nationwide"},
    "Social": {"congratulations", "well done", "bravery", "prince","promo","code","loyalty","collaboration","special","offer","vip","celebrity"}
}

trigger_patterns = {
    r"\bcredit\s*card\b": 5,
    r"\byou'?ve\s*won\b": 5,
    r"\bskyrocket\b": 5,
    r"\byou\s*are\s*selected\b": 5,
    r"\blink\s*access\b": 4,
    r"\bpersonal\s*details\b": 4,
    r"\buser\s*account\b": 4,
    r"\bbank\s*account\b": 4,
    r"\bhurry\s*up\b": 4,
    r"\badult\s*content\b": 4,
    r"\bsexual\s*disfunction\b": 4,
    r"\berection\s*pills\b": 4,
    r"\bthis\s*transaction\b": 4,
    r"\bspecial\s*offer\b": 4,
    r"\bprince\b": 3,
    r"\boffer\s*expires\b": 4,
    r"\burgent\s*attention\b": 4,
    r"\belectronic\s*wallet\b": 5,
    r"\bwallet\b":4,
    r"\bsuspicious\s*activity\b": 4,
    r"\baccount\s*suspended\b": 4,
    r"\bloyality\s*bonus\b": 4,
    r"\bpromo\s*code\b": 4,
    r"\blog\s*in\b": 3,
    r"\bclick\s*here\b": 3,
    r"\bverify\s*identity\b": 4,
    r"\bwin\s*a\s*free\b": 5,
    r"\blimited\s*time\b": 5,
    r"\bclaim\s*your\s*reward\b": 4,
    r"\bcongratulations\b": 4,
    r"\bunusual\s*sign[- ]in\b": 4,
    r"\breward\b": 4,
    r"\bpromo\b": 4,
    r"\b0xff\b": 5,
    r"\bwallet\b": 3,
    r"\bsex\b": 3,
    r"\bslut\b": 3,
    r"\bhardcore\b": 3,
    r"\bhorny\b": 3,
    r"\bxxx\b": 3,
    r"\bpenis\b": 3,
    r"\bcocks\b": 3,
    r"\bcuff\b": 3,
    r"\bweapons\b": 3,
    r"\bcollaboration\b": 3,
    r"\bstamina\b": 4,
    r"\bdiagnostic\b": 4,
    r"\blimited\b": 4,
    r"\bloyalty\b": 3,
    r"\bgun\b": 3,
    r"\bvip\b": 4,
    r"\burst\b": 4,
    r"\bdepressed\b": 4
}

boost_multiplier = 8
trigger_boost_weight = 40
TOP_N_PATTERNS = 300

def clean_tokens(texts):
    encodings = tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
    all_tokens = []
    for i in range(len(texts)):
        token_ids = encodings['input_ids'][i]
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        normalized = [re.sub(r'^[^a-z]+|[^a-z]+$', '', t.lower()) for t in tokens]
        filtered = [t for t in normalized if t and t not in stopwords]
        all_tokens.append(filtered)
    return all_tokens

df["tokens"] = clean_tokens(df["email_text"].tolist())

vectorizer = TfidfVectorizer(max_features=5000, stop_words=list(stopwords), ngram_range=(1, 2))
tfidf_matrix = vectorizer.fit_transform(df["email_text"])
tfidf_features = vectorizer.get_feature_names_out()
labels = df["email_type"].apply(lambda x: 1 if x == "Phishing Email" else 0)
chi2_scores, _ = chi2(tfidf_matrix, labels)
feature_scores = dict(zip(tfidf_features, chi2_scores))

phishing_scores = []
for email, tokens in zip(df["email_text"], df["tokens"]):
    score = 0
    email_lower = email.lower()
    for pattern, boost in trigger_patterns.items():
        score += len(re.findall(pattern, email_lower)) * boost * boost_multiplier
    for t in tokens:
        score += feature_scores.get(t, 0)
        for category in token_categories.values():
            if t in category:
                score += 1
    phishing_scores.append(score)
df["phishing_score"] = phishing_scores

spam_words = Counter()
ham_words = Counter()
spam_total, ham_total = 0, 0

for i, row in df.iterrows():
    tokens = row["tokens"]
    if row["email_type"] == "Phishing Email":
        spam_words.update(tokens)
        spam_total += len(tokens)
    else:
        ham_words.update(tokens)
        ham_total += len(tokens)

for pattern, score in trigger_patterns.items():
    matches_spam = sum(1 for text in df[df['email_type'] == 'Phishing Email']["email_text"]
                       if re.search(pattern, text.lower()))
    spam_total += matches_spam * trigger_boost_weight * score

word_probs = {}
for word in set(spam_words) | set(ham_words):
    spam_freq = spam_words[word] / spam_total if spam_total > 0 else 0
    ham_freq = ham_words[word] / ham_total if ham_total > 0 else 0
    if spam_freq + ham_freq > 0:
        p = spam_freq / (spam_freq + ham_freq)
        word_probs[word] = max(0.01, min(0.99, p))

top_spam_words = sorted(word_probs.items(), key=lambda x: x[1], reverse=True)[:30]
words, probs = zip(*top_spam_words) if top_spam_words else ([], [])

if words and probs:
    plt.figure(figsize=(14, 10))
    sns.barplot(x=list(probs), y=list(words), hue=list(words), legend=False, palette="Reds_r")
    plt.title("Top 30 Most Spam-Indicative Words (Bayes Classifiers Method)")
    plt.xlabel("Spam Probability")
    plt.ylabel("Word")
    plt.tight_layout()
    plt.show()

bayesian_scores = []
for tokens in df["tokens"]:
    probs = [word_probs[t] for t in tokens if t in word_probs]
    probs = sorted(probs, key=lambda p: abs(p - 0.5), reverse=True)[:20]
    if probs:
        product = np.prod(probs)
        inv_product = np.prod([1 - p for p in probs])
        bayes_score = product / (product + inv_product)
    else:
        bayes_score = 0.5
    bayesian_scores.append(bayes_score)

df["bayesian_score"] = bayesian_scores
df["combined_score"] = df["phishing_score"] * (df["bayesian_score"] + 0.01)

def clean_regex_pattern(pattern):
    cleaned = re.sub(r"\\b", "", pattern)
    cleaned = re.sub(r"\\s\\*", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()

def categorize_trigger(phrase, categories):
    tokens = re.findall(r"\w+", phrase.lower())
    for token in tokens:
        for cat_name, keywords in categories.items():
            if token in keywords:
                return cat_name
    return "Uncategorized"

pattern_counts = {}
for pattern in trigger_patterns:
    total_matches = df["email_text"].str.lower().apply(lambda text: len(re.findall(pattern, text)))
    pattern_counts[pattern] = total_matches.sum()

top_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)[:TOP_N_PATTERNS]
cleaned_top_patterns = [(clean_regex_pattern(p), count) for p, count in top_patterns]

category_counts = {}
for pattern, count in cleaned_top_patterns:
    category = categorize_trigger(pattern, token_categories)
    category_counts[category] = category_counts.get(category, 0) + count

if category_counts:
    plt.figure(figsize=(12, 8))
    sns.barplot(x=list(category_counts.values()), y=list(category_counts.keys()),
                hue=list(category_counts.keys()), legend=False, palette="coolwarm")
    plt.title(f"Total Occurrences of Trigger Phrases by Category (Top {TOP_N_PATTERNS} Patterns)")
    plt.xlabel("Total Matches")
    plt.ylabel("Category")
    plt.tight_layout()
    plt.show()

# === Top 300 Uncategorized Tokens from Trigger Phrases
uncategorized_tokens = []
for pattern, count in cleaned_top_patterns:
    if categorize_trigger(pattern, token_categories) == "Uncategorized":
        tokens = re.findall(r"\w+", pattern.lower())
        uncategorized_tokens.extend(tokens * count)

uncategorized_counter = Counter(uncategorized_tokens)
top300_tokens = uncategorized_counter.most_common(300)

if top300_tokens:
    tokens, counts = zip(*top300_tokens)
    plt.figure(figsize=(12, 10))
    sns.barplot(x=list(counts), y=list(tokens), hue=list(tokens), legend=False, palette="mako")
    plt.title("Top 300 Tokens from Uncategorized Trigger Phrases")
    plt.xlabel("Weighted Frequency")
    plt.ylabel("Token")
    plt.tight_layout()
    plt.show()
else:
    logging.info("No Uncategorized trigger tokens found. Skipping plot.")

trigger_norm = {clean_regex_pattern(k): v / max(pattern_counts.values()) for k, v in pattern_counts.items()}
graham_norm = {k: v / max(word_probs.values()) for k, v in word_probs.items()}

top_trigger = sorted(trigger_norm.items(), key=lambda x: x[1], reverse=True)[:10]
top_graham = sorted(graham_norm.items(), key=lambda x: x[1], reverse=True)[:15]

combined_data = []
for phrase, score in top_trigger:
    combined_data.append({"Phrase/Word": phrase, "Score": score, "Source": "Trigger Phrases"})
for word, score in top_graham:
    combined_data.append({"Phrase/Word": word, "Score": score, "Source": "Bayes Classifiers"})

compare_df = pd.DataFrame(combined_data)

plt.figure(figsize=(14, 10))
sns.barplot(x="Score", y="Phrase/Word", hue="Source", data=compare_df)
plt.title("Comparison: Top Trigger Phrases vs Bayes Classifiers Spam Words")
plt.xlabel("Normalized Importance Score")
plt.ylabel("Phrase / Word")
plt.tight_layout()
plt.show()

fpr, tpr, thresholds = roc_curve(labels, df["combined_score"])
roc_auc = auc(fpr, tpr)
optimal_threshold = thresholds[np.argmax(tpr - fpr)]

df["predicted_label"] = df["combined_score"].apply(lambda x: "Phishing Email" if x >= optimal_threshold else "Safe Email")

print("\n[Classification Report] (Combined Score):")
print(classification_report(df["email_type"], df["predicted_label"]))

cm = confusion_matrix(df["email_type"], df["predicted_label"], labels=["Phishing Email", "Safe Email"])
print("[Confusion Matrix]:\n", cm)

TP = cm[0, 0]
FN = cm[0, 1]
FP = cm[1, 0]
TN = cm[1, 1]

print(f"\nTrue Positives : {TP}")
print(f"True Negatives : {TN}")
print(f"False Positives: {FP}")
print(f"False Negatives: {FN}\n")

plt.figure(figsize=(8, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Phishing Email", "Safe Email"],
            yticklabels=["Phishing Email", "Safe Email"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.2f})", color="darkorange")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

precision, recall, _ = precision_recall_curve(labels, df["combined_score"])
avg_precision = average_precision_score(labels, df["combined_score"])

plt.figure(figsize=(10, 8))
plt.plot(recall, precision, color='green', label=f"PR (AP = {avg_precision:.2f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

df.to_csv("email_scores.csv", index=False)
print("\n[Info] Scores saved to 'email_scores.csv'")

eval_end = time.time()
eval_time = eval_end - eval_start
total_time = eval_end - total_start

print(f"\nTime to load model: {load_time:.2f} sec")
print(f"Time to evaluate emails: {eval_time:.2f} sec")
print(f"Total time: {total_time:.2f} sec")
