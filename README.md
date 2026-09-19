# Generative AI with LSTM — Text Generation

Word-level LSTM language model trained on Shakespeare's plays, with
autoregressive text generation from a seed phrase.

## Contents

| File                        | Description                                              |
|------------------------------|------------------------------------------------------------|
| `text_generation_lstm.py`    | Main script: preprocessing, model, training, generation   |
| `generate_samples.py`        | Loads a saved checkpoint and prints/saves sample text     |
| `data/shakespeare.txt`       | Dataset used (see "Dataset" below)                         |
| `checkpoints/best_model.keras`| Best checkpoint saved so far (lowest validation loss)     |
| `checkpoints/vocab.pkl`      | Pickled `Vocabulary` object (word<->id mappings) matching the checkpoint |
| `sample_output.txt`          | Sample generated text at several seeds/temperatures        |
| `run.log`                    | Full training console log                                   |

## Dataset

Default: the "tiny Shakespeare" corpus (~1.1MB of Shakespeare's plays),
public domain, originally popularized by Andrej Karpathy's char-rnn project
and sourced from Project Gutenberg:

```
https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

A copy is included at `data/shakespeare.txt`. You can substitute any
plain-text (`.txt`) corpus, e.g. the complete works of Shakespeare from
Project Gutenberg (https://www.gutenberg.org/ebooks/100) or any other book
from https://www.gutenberg.org/ — just point `--data` at the file.

## Pipeline

1. **Preprocessing** — lowercase, strip punctuation, whitespace-tokenize;
   rare words (freq < `min_freq`) are treated as `<unk>` to keep the
   vocabulary (and softmax output layer) a manageable size.
2. **Windowing** — a sliding window of `seq_len` tokens is used to predict
   the next token (many-to-one next-word prediction).
3. **Model** — `Embedding -> LSTM(return_sequences=True) -> Dropout ->
   LSTM -> Dropout -> Dense(softmax)`, trained with
   `sparse_categorical_crossentropy` + Adam.
4. **Training** — 90/10 train/val split, `EarlyStopping` (patience=3,
   restores best weights) + `ModelCheckpoint` (saves only when
   `val_loss` improves).
5. **Generation** — autoregressive sampling: the model predicts a
   distribution over the vocabulary for the next word, a word is sampled
   from that distribution (temperature-scaled), appended, and fed back in.

## Usage

```bash
pip install tensorflow numpy

# Train (full run, ~30 epochs on the full corpus)
python text_generation_lstm.py --data data/shakespeare.txt --epochs 30

# Quick experimentation run (smaller corpus slice, fewer epochs)
python text_generation_lstm.py --data data/shakespeare.txt \
    --max_chars 400000 --epochs 20 --seq_len 12 --batch_size 256

# Generate more samples from a saved checkpoint
python generate_samples.py
```

Key hyperparameters (all overridable via CLI flags — see `--help`):
`--seq_len` (context window length), `--embedding_dim`, `--lstm_units`,
`--batch_size`, `--epochs`, `--min_freq` (vocabulary pruning threshold).

## Status / notes on this run

The checkpoint included here (`checkpoints/best_model.keras`) is from an
in-progress training run (`--max_chars 400000 --epochs 20 --seq_len 12
--batch_size 256`) that had completed only a few epochs at the time this
package was zipped — see `run.log` for the full console output and
`sample_output.txt` for generated samples from that checkpoint. Loss was
still decreasing steadily and had not converged, so the sample outputs are
grammatically rough (lots of `<unk>` tokens and disjointed phrasing) rather
than coherent Shakespearean prose. Letting training run the full 20 epochs
(or more, on the full ~1.1MB corpus) with early stopping will substantially
improve coherence — training curves (loss/accuracy per epoch) are visible
in `run.log`.

## Bonus: architecture/hyperparameter experiments

Suggested experiments (all controllable via CLI flags without touching the
code):
- **Sequence length** (`--seq_len`): shorter windows (e.g. 8) train faster
  and see more examples per epoch but have less context; longer windows
  (e.g. 20-30) capture more context but need more data/compute to fit well.
- **Depth** (`--lstm_units`, or edit `build_model` to stack a 3rd LSTM
  layer): deeper/wider models have more capacity to model long-range
  structure but are slower to train and more prone to overfitting on a
  small corpus — watch the train/val loss gap.
- **Vocabulary size** (`--min_freq`): lowering it keeps more rare words
  (bigger softmax, sparser gradient signal per word); raising it collapses
  more words to `<unk>`, which trains faster but hurts fluency.
- **Sampling temperature**: at generation time, lower temperature (e.g.
  0.5) gives safer, more repetitive text; higher temperature (e.g. 1.0+)
  gives more novel but less grammatical text — see the temperature=0.5 vs.
  temperature=1.0 pairs in `sample_output.txt`.
