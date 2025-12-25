# Arabic Misogyny Detection: A Hybrid CNN-BiGRU Approach

This repository contains the source code, preprocessing pipeline, and research findings for an automated **Arabic Misogyny Detection** system. By leveraging a Hybrid CNN-BiGRU architecture and specialized FastText sub-word embeddings, this model achieves a stable test accuracy of **83.53%** on a consolidated corpus of ~13,000 dialectal tweets.

---

## Key Results

- **Test Accuracy**: 83.53%
- **F1-Score**: 0.8599
- **Recall (Misogyny Class)**: 84.35%
- **OOV Handling**: Reduced Out-of-Vocabulary rates from 37.99% to 0% using sub-word info.

---

## Project Architecture

The core innovation of this project is a dual-pathway feature extraction model:

- **Local Context (CNN)**: A 1D Convolutional layer captures misogynistic keywords and local n-gram patterns.
- **Global Context (BiGRU)**: A Bidirectional Gated Recurrent Unit (GRU) learns sequential dependencies and identifies sarcasm or mockery across the entire tweet.
- **Dual Pooling**: Simultaneous Global Average and Global Max pooling layers capture both the overall tone and the most salient offensive triggers.

---

## Specialized Preprocessing

Dialectal Arabic social media is highly "noisy." This project implements a **Semantic Preservation** strategy (found in `src/preprocess.py`):

- **Emoji-to-Text Conversion**: Translates visual icons (e.g., 🤮) into descriptive text tokens to retain emotional sentiment.
- **Orthographic Normalization**: Standardizes Alif, Hamza, and Ya variations to unify lexical forms.
- **Hashtag Normalization**: Converts segmented hashtags (e.g., #يا_واطية) into plain text for better feature extraction.
- **Punctuation Weighting**: Replaces `!` and `?` with specific tokens to help the model identify aggressive or mocking tones.

---

## Getting Started

### 1. Prerequisites

Install the required dependencies by running the following command:

```bash
pip install -r requirements.txt
```

### 2. Basic Usage

The model can be easily integrated into any Python workflow using the unified `prepare_input` function:

```python
import tensorflow as tf
from src.preprocess import prepare_input

# 1. Load the model (Download from Hugging Face if not local)
model = tf.keras.models.load_model("models/arabic_misogyny_hybrid_model.keras")

# 2. End-to-end classification
raw_tweet = "إنتِ جميلة جدا؟ 🤮"
input_data = prepare_input(raw_tweet)  # Handles cleaning, tokenizing, and padding
prediction = model.predict(input_data)

print("Misogyny Detected" if prediction > 0.5 else "Neutral Content")
```

---

## Dataset Attribution

This research consolidated two major benchmarks in Arabic NLP:

- **LeT-Mi**: Levantine Twitter Misogyny Dataset.
- **ArMI**: Arabic Misogyny Identification (FIRE 2021).

**Note**: Datasets are not included in this repository due to licensing restrictions. Please refer to the original authors for data access.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
