
# DEEP-LEARNING UTILITIES


from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


VOCAB_SIZE = 5000
MAX_SEQUENCE_LENGTH = 100
GLOVE_DIMENSION = 100


# TOKENISATION AND SEQUENCE PREPARATION


def prepare_sequences(
    train_text,
    test_text,
    vocab_size=VOCAB_SIZE,
    max_length=MAX_SEQUENCE_LENGTH,
):
    """
    Fit a tokenizer on training text and convert train/test
    messages into padded integer sequences.
    """

    tokenizer = Tokenizer(
        num_words=vocab_size,
        oov_token="<OOV>",
    )

    tokenizer.fit_on_texts(train_text)

    train_sequences = tokenizer.texts_to_sequences(train_text)
    test_sequences = tokenizer.texts_to_sequences(test_text)

    X_train = pad_sequences(
        train_sequences,
        maxlen=max_length,
        padding="post",
        truncating="post",
    )

    X_test = pad_sequences(
        test_sequences,
        maxlen=max_length,
        padding="post",
        truncating="post",
    )

    return tokenizer, X_train, X_test



# CLASS WEIGHTS


def calculate_class_weights(y_train):
    """
    Calculate balanced class weights for the ham/spam target.
    """

    classes = np.unique(y_train)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )

    return {
        int(class_label): float(weight)
        for class_label, weight in zip(classes, weights)
    }



# GLOVE EMBEDDINGS


def load_glove_embeddings(filepath):
    """
    Load pretrained GloVe word vectors from a text file.
    """

    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(
            f"GloVe embedding file not found: {filepath}"
        )

    embeddings = {}

    with filepath.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:
            values = line.rstrip().split()

            word = values[0]
            vector = np.asarray(
                values[1:],
                dtype="float32",
            )

            embeddings[word] = vector

    return embeddings


def create_embedding_matrix(
    tokenizer,
    embeddings,
    vocab_size=VOCAB_SIZE,
    embedding_dimension=GLOVE_DIMENSION,
):
    """
    Create an embedding matrix matching the fitted tokenizer.
    """

    matrix = np.zeros(
        (
            vocab_size,
            embedding_dimension,
        ),
        dtype="float32",
    )

    for word, index in tokenizer.word_index.items():

        if index >= vocab_size:
            continue

        vector = embeddings.get(word)

        if vector is not None:
            matrix[index] = vector

    return matrix



# MODEL EVALUATION


def evaluate_dl_classifier(
    model,
    X_test,
    y_test,
    threshold=0.5,
):
    """
    Evaluate a binary RNN/LSTM spam classifier.
    """

    probabilities = model.predict(
        X_test,
        verbose=0,
    ).ravel()

    predictions = (
        probabilities >= threshold
    ).astype(int)

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

    return metrics, matrix, predictions, probabilities
