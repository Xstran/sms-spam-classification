from pathlib import Path
import sys

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engineering import FEATURE_COLUMNS
from src.machine_learning import build_ml_pipeline, evaluate_classifier

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "sms_spam_processed.csv"

df = pd.read_csv(DATA_PATH)
X = df[["message_ml"] + FEATURE_COLUMNS]
y = df["target"]

# Stratified 80/20 split used for model evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

models = {
    "Baseline Naive Bayes": build_ml_pipeline(MultinomialNB(), FEATURE_COLUMNS, use_smote=False),
    "Baseline Decision Tree": build_ml_pipeline(DecisionTreeClassifier(random_state=42), FEATURE_COLUMNS, use_smote=False),
    "Tuned Naive Bayes": build_ml_pipeline(MultinomialNB(alpha=0.1), FEATURE_COLUMNS, max_features=3000, use_smote=True),
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    metrics, matrix, _ = evaluate_classifier(model, X_test, y_test)
    metrics["model"] = name
    results.append(metrics)

    print(f"\n{name}")
    print(pd.Series(metrics).drop("model"))
    print("Confusion matrix:")
    print(matrix)

results_df = pd.DataFrame(results)[["model", "accuracy", "precision", "recall", "f1"]]
print("\nModel comparison:")
print(results_df.round(3))
