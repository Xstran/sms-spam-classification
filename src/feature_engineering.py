
# HANDCRAFTED SMS FEATURES


import pandas as pd


FEATURE_COLUMNS = [
    "msg_len",
    "num_digits",
    "has_url",
    "has_call",
    "word_count",
    "len_le40",
    "len_le60",
    "len_le80",
    "len_le120",
    "len_le160",
]


def extract_features(text):
    """Extract structural and behavioural features from an SMS message."""

    text = str(text)

    # Basic message characteristics
    msg_len = len(text)
    num_digits = sum(char.isdigit() for char in text)
    has_url = int(
        "http" in text.lower()
        or "www" in text.lower()
        or ".com" in text.lower()
    )
    has_call = int("call" in text.lower())
    word_count = len(text.split())

    # Message-length threshold features
    len_le40 = int(msg_len <= 40)
    len_le60 = int(msg_len <= 60)
    len_le80 = int(msg_len <= 80)
    len_le120 = int(msg_len <= 120)
    len_le160 = int(msg_len <= 160)

    return [
        msg_len,
        num_digits,
        has_url,
        has_call,
        word_count,
        len_le40,
        len_le60,
        len_le80,
        len_le120,
        len_le160,
    ]


def add_handcrafted_features(df):
    """Add handcrafted numerical features to the SMS dataset."""

    df = df.copy()

    features = df["message"].apply(extract_features).tolist()

    feature_df = pd.DataFrame(
        features,
        columns=FEATURE_COLUMNS,
        index=df.index,
    )

    return pd.concat([df, feature_df], axis=1)
