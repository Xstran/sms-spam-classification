from pathlib import Path
import sys

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import load_sms_data, prepare_text_columns
from src.feature_engineering import add_handcrafted_features

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "sms_spam.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Load and prepare dataset
df = load_sms_data(DATA_PATH)
df = prepare_text_columns(df)
df = add_handcrafted_features(df)

print("Dataset shape:", df.shape)
print("\nClass distribution:")
print(df["label"].value_counts())
print("\nClass percentages:")
print((df["label"].value_counts(normalize=True) * 100).round(2))

# Save processed dataset
df.to_csv(PROCESSED_DIR / "sms_spam_processed.csv", index=False)

# Class distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="label")
plt.title("SMS Class Distribution")
plt.xlabel("Message Type")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "class_distribution.png", dpi=300)
plt.close()

# Message-length distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="msg_len", hue="label", bins=50, element="step")
plt.title("Message Length by Class")
plt.xlabel("Message Length")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "message_length_distribution.png", dpi=300)
plt.close()

# Spam word cloud
spam_text = " ".join(df.loc[df["target"] == 1, "message_ml"])
wordcloud = WordCloud(width=1000, height=500, background_color="white").generate(spam_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Frequent Words in Spam Messages")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "spam_wordcloud.png", dpi=300)
plt.close()

print("\nProcessed dataset saved to:", PROCESSED_DIR / "sms_spam_processed.csv")
print("EDA figures saved to:", RESULTS_DIR)
