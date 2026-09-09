# Data

This project uses the **UCI SMS Spam Collection**, a labelled dataset for binary SMS spam classification.

## Dataset

The original dataset contains **5,572 SMS messages**:

- 4,825 ham messages
- 747 spam messages
- Spam proportion: 13.41%

The original file contains two relevant columns:

- `v1` — message label (`ham` or `spam`)
- `v2` — SMS message text

Additional unnamed columns containing mostly missing values are removed during preprocessing.

## Cleaning

The preprocessing workflow:

- renames `v1` to `label`
- renames `v2` to `message`
- removes irrelevant unnamed columns
- removes duplicate messages
- converts labels to binary values
- prepares separate text representations for machine learning and deep learning

After removing **403 duplicate rows**, **5,169 messages** remain for modelling.

## Class Imbalance

The dataset is substantially imbalanced, with spam representing only a small proportion of messages.

Different imbalance-handling strategies are therefore compared:

- **SMOTE** for the TF-IDF machine-learning pipeline
- **class weighting** for the neural-network models

## Source

UCI SMS Spam Collection  
Almeida et al. (2011)
