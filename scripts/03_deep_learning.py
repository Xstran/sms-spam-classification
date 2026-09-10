from pathlib import Path
import sys
import pandas as pd

from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, Dense, Dropout, SpatialDropout1D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.deep_learning import prepare_sequences, calculate_class_weights, load_glove_embeddings, create_embedding_matrix, evaluate_dl_classifier

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "sms_spam_processed.csv"
GLOVE_PATH = PROJECT_ROOT / "data" / "embeddings" / "glove.6B.100d.txt"

df = pd.read_csv(DATA_PATH)
X_train, X_test, y_train, y_test = train_test_split(df["message_dl"], df["target"], test_size=0.20, random_state=42, stratify=df["target"])

tokenizer, X_train_seq, X_test_seq = prepare_sequences(X_train, X_test)
class_weights = calculate_class_weights(y_train)

# Tuned SimpleRNN
rnn = Sequential([
    Embedding(5000, 100),
    SimpleRNN(64, dropout=0.3, recurrent_dropout=0.2),
    BatchNormalization(),
    Dense(1, activation="sigmoid")
])
rnn.compile(optimizer=Adam(learning_rate=0.0005, clipnorm=1.0), loss="binary_crossentropy", metrics=["accuracy"])

rnn_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
rnn.fit(X_train_seq, y_train, epochs=30, batch_size=32, validation_split=0.20, class_weight=class_weights, callbacks=[rnn_stop], verbose=1)

rnn_metrics, _, _, _ = evaluate_dl_classifier(rnn, X_test_seq, y_test)
print("\nTuned RNN:")
print(pd.Series(rnn_metrics).round(3))

# LSTM with pretrained GloVe embeddings
if GLOVE_PATH.exists():
    embeddings = load_glove_embeddings(GLOVE_PATH)
    embedding_matrix = create_embedding_matrix(tokenizer, embeddings)

    lstm = Sequential([
        Embedding(5000, 100, weights=[embedding_matrix], trainable=False),
        SpatialDropout1D(0.2),
        LSTM(128, dropout=0.3, kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dense(1, activation="sigmoid")
    ])
    lstm.compile(optimizer=Adam(learning_rate=0.0002), loss="binary_crossentropy", metrics=["accuracy"])

    lstm_stop = EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
    lstm.fit(X_train_seq, y_train, epochs=30, batch_size=32, validation_split=0.20, class_weight=class_weights, callbacks=[lstm_stop], verbose=1)

    lstm_metrics, _, _, _ = evaluate_dl_classifier(lstm, X_test_seq, y_test)
    print("\nLSTM + GloVe:")
    print(pd.Series(lstm_metrics).round(3))
else:
    print("\nGloVe file not found. RNN completed; LSTM + GloVe skipped.")
