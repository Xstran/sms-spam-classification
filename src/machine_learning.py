
# TRADITIONAL MACHINE-LEARNING UTILITIES


from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV


def build_preprocessor(feature_columns, max_features=3000):
    """
    Combine TF-IDF text features with handcrafted numerical features.
    """

    return ColumnTransformer(
        transformers=[
            (
                "text",
                TfidfVectorizer(
                    ngram_range=(1, 3),
                    max_features=max_features,
                ),
                "message_ml",
            ),
            (
                "numeric",
                "passthrough",
                feature_columns,
            ),
        ]
    )


def build_ml_pipeline(
    model,
    feature_columns,
    max_features=3000,
    use_smote=True,
):
    """
    Build the machine-learning pipeline.

    Processing order:
    TF-IDF + handcrafted features -> SMOTE -> classifier
    """

    preprocessor = build_preprocessor(
        feature_columns=feature_columns,
        max_features=max_features,
    )

    steps = [
        ("preprocessor", preprocessor),
    ]

    if use_smote:
        steps.append(
            (
                "smote",
                SMOTE(random_state=42),
            )
        )

    steps.append(
        ("model", model)
    )

    return Pipeline(steps)


def tune_classifier(
    pipeline,
    param_grid,
    X_train,
    y_train,
    cv=5,
):
    """
    Tune a classifier using GridSearchCV with F1 as the
    optimisation metric.
    """

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        return_train_score=True,
    )

    search.fit(
        X_train,
        y_train,
    )

    return search


def evaluate_classifier(
    model,
    X_test,
    y_test,
):
    """Evaluate spam-classification performance."""

    predictions = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(
            y_test,
            predictions,
        ),
        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),
    }

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    )

    return metrics, matrix, predictions
