"""
Generative AI with LSTM — Word-Level Text Generation
=====================================================

Trains an LSTM language model on a corpus of text (default: the "tiny
Shakespeare" corpus — Shakespeare's plays, public domain) and generates new
text from a seed phrase.

Pipeline
--------
1. Load & clean text (lowercase, strip punctuation).
2. Tokenize into words, build a vocabulary, encode as integers.
3. Build fixed-length input windows -> "next word" targets.
4. Train an Embedding -> LSTM -> Dense(softmax) model.
5. Generate new text by iteratively sampling the next word.

Usage
-----
    python text_generation_lstm.py --data data/shakespeare.txt --epochs 30

Dataset
-------
Default dataset: "tiny Shakespeare" (~1.1MB of Shakespeare's plays),
originally popularized by Andrej Karpathy's char-rnn project, itself
sourced from Project Gutenberg's public-domain Shakespeare texts:
    https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

You may substitute ANY plain-text (.txt) corpus — e.g. the full Shakespeare
works from Project Gutenberg (https://www.gutenberg.org/ebooks/100), or any
book from https://www.gutenberg.org/. Just point --data at the file.
"""

import argparse
import os
import re
import string
import pickle
import random

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# --------------------------------------------------------------------------- #
# 1. DATA LOADING & PREPROCESSING
# --------------------------------------------------------------------------- #

def load_text(path: str, max_chars: int | None = None) -> str:
    """Read a plain-text file. `max_chars` optionally caps the corpus size
    (handy for fast experimentation on a laptop/CPU)."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if max_chars:
        text = text[:max_chars]
    return text


def clean_text(text: str) -> str:
    """Lowercase and strip punctuation, collapse whitespace."""
    text = text.lower()
    # Keep letters, digits and spaces only; drop punctuation entirely.
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    """Word-level tokenization (whitespace split, already cleaned)."""
    return text.split(" ")


class Vocabulary:
    """Maps words <-> integer ids. Rare words are collapsed into <unk>."""

    def __init__(self, tokens: list[str], min_freq: int = 2):
        from collections import Counter

        counts = Counter(tokens)
        # Keep words seen at least `min_freq` times -> keeps vocab (and the
        # final Dense/softmax layer) a manageable size.
        vocab_words = [w for w, c in counts.items() if c >= min_freq]
        vocab_words = sorted(vocab_words)

        self.word2idx = {"<pad>": 0, "<unk>": 1}
        for w in vocab_words:
            self.word2idx[w] = len(self.word2idx)
        self.idx2word = {i: w for w, i in self.word2idx.items()}

    def __len__(self):
        return len(self.word2idx)

    def encode(self, tokens: list[str]) -> list[int]:
        unk = self.word2idx["<unk>"]
        return [self.word2idx.get(t, unk) for t in tokens]

    def decode(self, ids: list[int]) -> list[str]:
        return [self.idx2word.get(i, "<unk>") for i in ids]


def build_sequences(token_ids: list[int], seq_len: int, step: int = 1):
    """Slide a window of length `seq_len` over the token stream. Each
    window's next token is the target label (many-to-one next-word
    prediction)."""
    X, y = [], []
    for i in range(0, len(token_ids) - seq_len, step):
        X.append(token_ids[i:i + seq_len])
        y.append(token_ids[i + seq_len])
    return np.array(X, dtype=np.int32), np.array(y, dtype=np.int32)


# --------------------------------------------------------------------------- #
# 2. MODEL
# --------------------------------------------------------------------------- #

def build_model(vocab_size: int, seq_len: int, embedding_dim: int = 100,
                 lstm_units: int = 150) -> keras.Model:
    """Embedding -> stacked LSTM -> Dense(softmax) next-word classifier."""
    model = keras.Sequential([
        layers.Input(shape=(seq_len,)),
        layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim),
        layers.LSTM(lstm_units, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(lstm_units),
        layers.Dropout(0.2),
        layers.Dense(vocab_size, activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        # sparse_categorical_crossentropy = categorical_crossentropy but
        # takes integer labels directly instead of one-hot vectors, which
        # avoids materializing a (batch, vocab_size) one-hot tensor for
        # every target -- much lighter on memory for large vocabularies.
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# --------------------------------------------------------------------------- #
# 3. TEXT GENERATION
# --------------------------------------------------------------------------- #

def sample_with_temperature(probs: np.ndarray, temperature: float = 1.0) -> int:
    """Sample an index from a probability distribution, reshaped by
    `temperature`. temperature < 1 -> more conservative/repetitive,
    temperature > 1 -> more random/creative."""
    probs = np.asarray(probs).astype("float64")
    probs = np.log(probs + 1e-9) / temperature
    probs = np.exp(probs)
    probs = probs / np.sum(probs)
    return np.random.choice(len(probs), p=probs)


def generate_text(model: keras.Model, vocab: Vocabulary, seed_text: str,
                   seq_len: int, n_words: int = 50,
                   temperature: float = 0.8) -> str:
    """Iteratively predict and append the next word, feeding the model's
    own output back in as context (autoregressive generation)."""
    seed_tokens = tokenize(clean_text(seed_text))
    ids = vocab.encode(seed_tokens)

    # Left-pad or trim so the seed matches the model's expected input length.
    if len(ids) < seq_len:
        ids = [vocab.word2idx["<pad>"]] * (seq_len - len(ids)) + ids
    else:
        ids = ids[-seq_len:]

    generated = list(seed_tokens)
    window = list(ids)

    for _ in range(n_words):
        x = np.array(window[-seq_len:]).reshape(1, seq_len)
        preds = model.predict(x, verbose=0)[0]
        next_id = sample_with_temperature(preds, temperature)
        next_word = vocab.idx2word.get(next_id, "<unk>")
        generated.append(next_word)
        window.append(next_id)

    return " ".join(generated)


# --------------------------------------------------------------------------- #
# 4. MAIN — glue it all together
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Train an LSTM text generator.")
    parser.add_argument("--data", type=str, default="data/shakespeare.txt")
    parser.add_argument("--max_chars", type=int, default=None,
                         help="Optionally cap corpus size for a quick run.")
    parser.add_argument("--seq_len", type=int, default=15)
    parser.add_argument("--min_freq", type=int, default=2)
    parser.add_argument("--embedding_dim", type=int, default=100)
    parser.add_argument("--lstm_units", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--seed_texts", type=str, nargs="*",
                         default=["to be or not to be",
                                  "the king said unto",
                                  "love is a"])
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ---- Preprocess -------------------------------------------------------
    print("Loading & cleaning text...")
    raw_text = load_text(args.data, max_chars=args.max_chars)
    cleaned = clean_text(raw_text)
    tokens = tokenize(cleaned)
    print(f"Total tokens: {len(tokens):,}")

    vocab = Vocabulary(tokens, min_freq=args.min_freq)
    print(f"Vocabulary size: {len(vocab):,}")

    token_ids = vocab.encode(tokens)
    X, y = build_sequences(token_ids, seq_len=args.seq_len)
    print(f"Training sequences: {X.shape[0]:,}")

    # Save vocab for later reuse (e.g. loading the trained model separately).
    with open(os.path.join(args.checkpoint_dir, "vocab.pkl"), "wb") as f:
        pickle.dump(vocab, f)

    # ---- Train/val split ---------------------------------------------------
    n_val = int(len(X) * args.val_split)
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    # ---- Model ---------------------------------------------------------
    model = build_model(len(vocab), args.seq_len,
                         embedding_dim=args.embedding_dim,
                         lstm_units=args.lstm_units)
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,
                                       restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(
            os.path.join(args.checkpoint_dir, "best_model.keras"),
            monitor="val_loss", save_best_only=True),
    ]

    # ---- Train ---------------------------------------------------------
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    # ---- Generate sample text ---------------------------------------------
    print("\n" + "=" * 70)
    print("SAMPLE GENERATED TEXT")
    print("=" * 70)
    for seed in args.seed_texts:
        for temp in (0.5, 1.0):
            out = generate_text(model, vocab, seed, args.seq_len,
                                 n_words=40, temperature=temp)
            print(f"\nSeed: {seed!r}  (temperature={temp})\n-> {out}")

    return model, vocab, history


if __name__ == "__main__":
    main()
