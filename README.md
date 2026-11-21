# Stylistic Text Generator Comparator

[![Built with askThili](https://msrit.ask.thili.ai/api/v1/public/projects/7722ernyur/badge.svg)](https://msrit.ask.thili.ai/b/7722ernyur)

A command-line tool for sports journalists: it learns a commentary style (football or cricket) from a small corpus
in two ways, a word n-gram model and distilgpt2 fine-tuned with LoRA, then shows their output side by side on the
same prompt with numbers that say how repetitive and how copied each one is.

## Milestones

- [x] **1. Corpus:** football and cricket commentary, cleaned and tokenized the same way for both models.
- [x] **2. N-gram model** with backoff, generating text word by word.
- [x] **3. LoRA injection** into the attention layers of a pre-trained distilgpt2, written from scratch.
- [x] **4. Fine-tuning** of the LoRA adapter only (0.18% of the weights) on one style.
- [x] **5. Generation** from the LoRA model with a hand-written sampling loop.
- [x] **6. Comparison:** side-by-side output, scores, a base-model baseline, and a held-out evaluation.

## Run it

You need Python 3.12 and git.

Mac / Linux:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m app.cli hello
pytest
```

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1      # if scripts are blocked: Set-ExecutionPolicy -Scope Process Bypass
pip install -r requirements.txt
python -m app.cli hello
pytest
```

No API key is needed: both models run on your own machine. The first `train` downloads distilgpt2 (about 350 MB)
from Hugging Face once.

## Using it

```bash
python -m app.cli corpus                                         # styles and corpus sizes
python -m app.cli ngram --style football --prompt "He picks it"   # n-gram model
python -m app.cli train --style football                         # LoRA fine-tune (a few minutes on a CPU)
python -m app.cli generate --style football --prompt "He picks it"
python -m app.cli compare --style football --prompt "He picks it" --with-base --rate
python -m app.cli ratings                                        # which model won your ratings
python -m app.cli evaluate --style football                      # held-out scores, writes results/football.md
```

Train each style once (`--style cricket` too). Adapters are saved in `adapters/<style>.pt`, a few hundred KB each
because only the LoRA weights are stored.

## How it works

| Milestone | Where | What |
|---|---|---|
| 1. Corpus | `corpus/*.txt`, `load_corpus`, `tokenize` | One file per commentary style, one passage per line, written for this project. Text is cleaned and tokenized the same way for both models. |
| 2. N-gram | `NGramModel` | Counts word n-grams (default n=3) with sentence start/end markers. Generation samples the next word from the longest context seen, backing off to shorter contexts. |
| 3. LoRA injection | `LoRALayer`, `inject_lora` | Freezes distilgpt2 and wraps each attention projection (`c_attn`) as `W x + (alpha/r) · B A x`. `B` starts at zero, so the model is unchanged until training. About 0.2% of parameters are trainable. |
| 4. Fine-tuning | `train_lora` | Next-token cross-entropy on blocks of the corpus, AdamW on the LoRA weights only. `train` prints the corpus loss before and after. |
| 5. Generation | `generate_llm` | A hand-written sampling loop: temperature, top-k, stop at end-of-text. |
| 6. Comparison | `compare`, `render_comparison`, `compare --rate` | Same prompt and seeds for every model, shown side by side with simple numbers: `distinct_2` (repetitiveness) and `copied_4` (share of 4-word runs copied from the corpus). `--with-base` adds distilgpt2 with the adapter switched off, to show what LoRA changed. `--rate` saves your judgement to `data/app.json`. |

What to look for: the n-gram model sounds right but copies long runs of the corpus (high `copied_4`) and drifts off
topic. The base model writes fluent English that is not commentary. The LoRA model picks up the commentary style
while writing new sentences.

## Results

`python -m app.cli evaluate --style <style>` holds out every 6th passage, trains all four models on the rest, and
scores them on the passages none of them saw. The unigram model (random words by frequency) is the simple baseline.
Full output with examples: [results/football.md](results/football.md), [results/cricket.md](results/cricket.md).

| Model | Football: held-out perplexity per word | Cricket: held-out perplexity per word | copied_4 (football / cricket) |
|---|---|---|---|
| unigram (simple baseline) | 420.4 | 492.1 | 0.00 / 0.00 |
| 3-gram | 289.9 | 400.7 | **0.44 / 0.42** |
| base distilgpt2 (no LoRA) | 249.3 | 896.5 | 0.00 / 0.00 |
| **LoRA distilgpt2** | **105.9** | **359.0** | 0.01 / 0.01 |

Lower perplexity means the model finds real, unseen commentary more likely. It is measured per word (total
negative log-likelihood divided by the number of words), so the word-level n-grams and the subword LLM are on the
same scale. The n-grams score lower-cased text, which slightly favours them.

What this shows:

- **LoRA works.** Training 0.18% of the weights on 40 passages cuts held-out perplexity 2.4x on football and 2.5x
  on cricket compared with the same model before fine-tuning.
- **The 3-gram beats the baseline but copies.** It is better than the unigram baseline, yet 42-44% of its 4-word runs
  are lifted straight from the training text. The LoRA model writes new sentences (1% copied).
- **The base model does not know the style.** It is fluent but drifts into unrelated text, and on cricket it does
  worse than even the unigram baseline, because cricket vocabulary (leg before wicket, maiden, yorker) is rare in
  general English.

Example, prompt "That was plumb, it" (cricket, not in training):

- **3-gram:** That was plumb, it is a beautiful cover drive, timed to perfection, and the pressure is building on the field, and it is a no ball, deceives him completely,
- **base distilgpt2:** That was plumb, it was a little over a quarter of the way back in March the day before the election and that was when I looked over my arm and I saw a lot of it in a large size.
- **LoRA distilgpt2:** That was plumb, it was a little over the line...

Limits: the corpus is small (48 passages per style) and the test set is 8 passages, so the numbers are a guide, not
a benchmark.

## Tests

`pytest` runs one test per milestone. The LLM tests use a tiny randomly initialised GPT-2 and a word-level tokenizer,
so they need no download and run in seconds (also on GitHub Actions).

## Stretch ideas

- Ask a hosted model to judge which output reads more like commentary, next to your own ratings.
- Train on a bigger corpus, or add a third style.
