"""Load the trained checkpoint and generate sample text for a few seeds."""
import pickle
import numpy as np
from tensorflow import keras
from text_generation_lstm import generate_text, Vocabulary, clean_text, tokenize

SEQ_LEN = 12

with open("checkpoints/vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

model = keras.models.load_model("checkpoints/best_model.keras")

seeds = ["to be or not to be", "the king said unto", "love is a", "romeo where art thou"]

lines = []
lines.append("SAMPLE GENERATED TEXT (from checkpoints/best_model.keras)")
lines.append("=" * 70)
for seed in seeds:
    for temp in (0.5, 1.0):
        out = generate_text(model, vocab, seed, SEQ_LEN, n_words=40, temperature=temp)
        block = f"\nSeed: {seed!r}  (temperature={temp})\n-> {out}"
        print(block)
        lines.append(block)

with open("sample_output.txt", "w") as f:
    f.write("\n".join(lines) + "\n")
print("\nSaved to sample_output.txt")
