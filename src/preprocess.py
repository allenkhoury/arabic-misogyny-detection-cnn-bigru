import re
import json
import emoji
import pyarabic.araby as araby
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_tokenizer(path='tokenizer.json'):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return tokenizer_from_json(data)

TOKENIZER = load_tokenizer()

def data_cleaning(text):

  # For any foreign language or for links
  text = text.lower()

  # Removing all mentions with @
  text = re.sub(r"@\w+", '', text)

  # Remove all links
  text = re.sub(r'https?:\/\/.*[\r\n]*', "", text, flags=re.MULTILINE)

  # These are some issues that couldnt be removed so I had to manually force remove them
  for char in ["مستخدم@", "#", "…", "RT", "\ufffd"]:
    text = text.replace(char, "")

  # Convert exclamation point and question mark into words, to give them more weight in the context
  text = re.sub(r'!+', ' [EXCLAMATION] ', text)
  text = re.sub(r'\?+', ' [QUESTION] ', text)

  # Removed some useless characters
  chars_to_remove = r'[!"#$%&\'()*+,-./:;<=>?@\[\\\]^_`{|}~،؛؟ـ٪٫٬«»“”•·…﴾﴿〈〉°±÷×©®™€£¥¢]'
  text = re.sub(chars_to_remove, ' ', text)

  # I am keeping everything on one line
  text = re.sub(r'[\r\n]+', ' ', text)

  # Normalizing some letters to unify some words and match more words
  text = re.sub("[إأآا]", "ا", text)
  text = re.sub("ى", "ي", text)
  text = re.sub("ؤ", "ء", text)
  text = re.sub("ئ", "ء", text)
  text = re.sub("ة", "ه", text)
  text = re.sub("گ", "ك", text)

  # Remove extra spaces
  text = re.sub(r'\s+', ' ', text)

  # This is for underscores inside hashtags, so I am converting the hashtag into words
  text = text.replace("_", " ")

  # Since I dont have much data, I am converting emojis into text to also give more context to sentences
  text = emoji.demojize(text, delimiters=(" ", " "))

  # strip tashkeel and tatweel
  text = araby.strip_tashkeel(text)
  text = araby.strip_tatweel(text)

  text = text.strip()
  return text

def prepare_input(text, max_len=48):
    # Step 1: Clean
    cleaned_text = data_cleaning(text)

    # Step 2: Tokenize (Convert text to sequence of integers)
    # Wrap cleaned_text in a list because texts_to_sequences expects a list of strings
    sequence = TOKENIZER.texts_to_sequences([cleaned_text])

    # Step 3: Pad (Ensure fixed length for the CNN-BiGRU architecture)
    padded = pad_sequences(sequence, maxlen=max_len)

    return padded
