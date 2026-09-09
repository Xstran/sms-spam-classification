import re
from pathlib import Path

import pandas as pd


def load_sms_data(filepath):
    """Load and prepare the SMS Spam Collection dataset."""

    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    # The source CSV used in the project contains v1/v2 plus
    # irrelevant unnamed columns.
    df = pd.read_csv(filepath, encoding="latin-1")

    required_columns = {"v1", "v2"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            "Expected dataset columns 'v1' and 'v2' were not found."
        )

    # Keep only the two useful columns
    df = df[["v1", "v2"]].copy()

    # Rename columns for clarity
    df = df.rename(
        columns={
            "v1": "label",
            "v2": "message",
        }
    )

    # Remove duplicated messages
    df = df.drop_duplicates().reset_index(drop=True)

    # Convert labels to binary values
    df["target"] = df["label"].map(
        {
            "ham": 0,
            "spam": 1,
        }
    )

    if df["target"].isna().any():
        raise ValueError("Unexpected class label found in dataset.")

    return df


def clean_ml_text(text):
    """
    Clean text for TF-IDF machine-learning models.

    URLs are removed while digits and selected punctuation
    associated with spam patterns are retained.
    """

    text = str(text).lower()

    # Remove URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text,
    )

    # Keep letters, digits and ! ? £ $ @
    text = re.sub(
        r"[^a-z0-9!?£$@\s]",
        " ",
        text,
    )

    # Normalise whitespace
    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text


def clean_dl_text(text):
    """
    Clean text for the RNN and LSTM models.

    Only lowercase alphanumeric text and spaces are retained.
    """

    text = str(text).lower()

    # Remove URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text,
    )

    # Keep only letters, digits and spaces
    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text,
    )

    # Normalise whitespace
    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text


def prepare_text_columns(df):
    """Create separate cleaned text columns for ML and DL models."""

    df = df.copy()

    df["message_ml"] = df["message"].apply(clean_ml_text)
    df["message_dl"] = df["message"].apply(clean_dl_text)

    return df
