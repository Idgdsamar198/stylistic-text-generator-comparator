"""The comparator: an n-gram model and a LoRA-adapted LLM, trained on the same sports commentary corpus.

Each section below is one milestone from README.md.
"""
from __future__ import annotations

import math
import random
import re
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

CORPUS_DIR = Path("corpus")
ADAPTER_DIR = Path("adapters")
BASE_MODEL = "distilbert/distilgpt2"


# --- Milestone 1: Prepare a small corpus of sports commentary text. -------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[.,!?;:]")


def available_styles(corpus_dir: Path | str = CORPUS_DIR) -> list[str]:
    """Each `corpus/<style>.txt` file is one commentary style."""
    return sorted(p.stem for p in Path(corpus_dir).glob("*.txt"))


def clean_text(text: str) -> str:
    """Normalise quotes, dashes and whitespace so both models see the same text."""
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = text.replace("—", ", ").replace("–", "-").replace('"', "")
    return re.sub(r"\s+", " ", text).strip()


def load_corpus(style: str, corpus_dir: Path | str = CORPUS_DIR) -> list[str]:
    """Return the cleaned passages (one per line) of one style; blank lines and `#` comments are skipped."""
    path = Path(corpus_dir) / f"{style}.txt"
    if not path.exists():
        raise FileNotFoundError(f"No corpus for style {style!r}. Styles: {', '.join(available_styles(corpus_dir))}")
    passages = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = clean_text(line)
        if line and not line.startswith("#"):
            passages.append(line)
    return passages


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens; punctuation marks are tokens of their own."""
    return _TOKEN_RE.findall(clean_text(text).lower())


def detokenize(tokens: list[str]) -> str:
    """Join tokens back into readable text: no space before punctuation, capital letter after . ! ?"""
    out: list[str] = []
    capital = True
    for tok in tokens:
        if tok in ".,!?;:":
            if out:
                out[-1] += tok
            capital = tok in ".!?"
            continue
        out.append(tok.capitalize() if capital else tok)
        capital = False
    return " ".join(out)


def corpus_stats(passages: list[str]) -> dict:
    tokens = [t for p in passages for t in tokenize(p)]
    return {"passages": len(passages), "tokens": len(tokens), "vocabulary": len(set(tokens))}


# --- Milestone 2: Implement the n-gram model and generate text from it. ----------------------------------------

BOS, EOS = "<s>", "</s>"


class NGramModel:
    """A word n-gram model with backoff: if a context was never seen, it falls back to a shorter one."""

    def __init__(self, n: int = 3):
        if n < 1:
            raise ValueError("n must be at least 1")
        self.n = n
        # counts[k][context of length k] -> Counter of next words, for k = 0 .. n-1
        self.counts: list[dict[tuple, Counter]] = [defaultdict(Counter) for _ in range(n)]

    def fit(self, passages: list[str]) -> "NGramModel":
        for passage in passages:
            tokens = [BOS] * (self.n - 1) + tokenize(passage) + [EOS]
            for i in range(self.n - 1, len(tokens)):
                for k in range(self.n):
                    context = tuple(tokens[i - k:i]) if k else ()
                    self.counts[k][context][tokens[i]] += 1
        return self

    def next_word_probs(self, context: list[str]) -> dict[str, float]:
        """Probabilities of the next word, from the longest context the model has seen."""
        padded = [BOS] * (self.n - 1) + list(context)
        for k in range(self.n - 1, -1, -1):
            ctx = tuple(padded[len(padded) - k:]) if k else ()
            counter = self.counts[k].get(ctx)
            if counter:
                total = sum(counter.values())
                return {word: c / total for word, c in counter.items()}
        return {}

    def generate(self, prompt: str = "", max_words: int = 30, temperature: float = 1.0,
                 seed: int | None = None) -> str:
        """Continue `prompt` word by word until the end of a passage or `max_words`."""
        rng = random.Random(seed)
        tokens = tokenize(prompt)
        for _ in range(max_words):
            probs = self.next_word_probs(tokens)
            if not probs:
                break
            words = list(probs)
            weights = [p ** (1.0 / max(temperature, 1e-6)) for p in probs.values()]
            word = rng.choices(words, weights=weights)[0]
            if word == EOS:
                break
            tokens.append(word)
        return detokenize(tokens)


# --- Milestone 3: Load a small pre-trained LLM and implement LoRA adapter injection. --------------------------

class LoRALayer(nn.Module):
    """Wraps a frozen layer W and adds a trainable low-rank update:  y = W x + (alpha / r) * B A x.

    A starts random and B starts at zero, so before training the layer behaves exactly like the original.
    """

    def __init__(self, base: nn.Module, in_features: int, out_features: int,
                 r: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.r = r
        self.scale = alpha / r
        self.enabled = True
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.enabled:
            out = out + (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scale
        return out


def _layer_shape(module: nn.Module) -> tuple[int, int]:
    """(in_features, out_features) of a Linear layer, or of GPT-2's Conv1D (whose weight is stored transposed)."""
    if isinstance(module, nn.Linear):
        return module.in_features, module.out_features
    if hasattr(module, "nf") and hasattr(module, "weight"):          # transformers' Conv1D
        return module.weight.shape[0], module.weight.shape[1]
    raise TypeError(f"Cannot add LoRA to {type(module).__name__}")


def inject_lora(model: nn.Module, targets: tuple[str, ...] = ("c_attn",), r: int = 8,
                alpha: float = 16.0, dropout: float = 0.05) -> int:
    """Freeze every weight of `model`, then wrap each layer whose name is in `targets` with a LoRALayer.

    Returns how many layers were wrapped.
    """
    for p in model.parameters():
        p.requires_grad = False
    found = [(parent, name, child) for parent in model.modules()
             for name, child in parent.named_children() if name in targets and not isinstance(child, LoRALayer)]
    for parent, name, child in found:
        in_f, out_f = _layer_shape(child)
        setattr(parent, name, LoRALayer(child, in_f, out_f, r=r, alpha=alpha, dropout=dropout))
    if not found:
        raise ValueError(f"No layers named {targets} in the model")
    return len(found)


def set_lora_enabled(model: nn.Module, enabled: bool) -> None:
    """Switch the adapters off to get the original model back (used to show the base model in comparisons)."""
    for m in model.modules():
        if isinstance(m, LoRALayer):
            m.enabled = enabled


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """(trainable, total) parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, sum(p.numel() for p in model.parameters())


def load_base_model(name: str = BASE_MODEL):
    """Download (once) and load a small pre-trained causal LM and its tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name)
    model.eval()
    return model, tokenizer


# --- Milestone 4: Fine-tune the LoRA adapter on the sports commentary corpus. ---------------------------------

def make_blocks(tokenizer, passages: list[str], block_size: int = 64) -> torch.Tensor:
    """Encode all passages (each followed by end-of-text) into one stream and cut it into equal blocks."""
    ids: list[int] = []
    for passage in passages:
        ids += tokenizer.encode(passage) + [tokenizer.eos_token_id]
    if len(ids) < block_size + 1:
        return torch.tensor([ids])
    n_blocks = len(ids) // block_size
    return torch.tensor(ids[: n_blocks * block_size]).view(n_blocks, block_size)


def lm_loss(model: nn.Module, batch: torch.Tensor) -> torch.Tensor:
    """Next-token cross-entropy: each position predicts the token after it."""
    logits = model(batch).logits
    return F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1))


@torch.no_grad()
def evaluate_loss(model: nn.Module, tokenizer, passages: list[str], block_size: int = 64) -> float:
    model.eval()
    return lm_loss(model, make_blocks(tokenizer, passages, block_size)).item()


def train_lora(model: nn.Module, tokenizer, passages: list[str], epochs: int = 4, lr: float = 2e-3,
               batch_size: int = 4, block_size: int = 64, seed: int = 0, log=None) -> list[float]:
    """Train only the LoRA parameters. Returns the mean training loss of each epoch."""
    torch.manual_seed(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("Nothing to train: call inject_lora first")
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    blocks = make_blocks(tokenizer, passages, block_size)
    history = []
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(blocks))
        losses = []
        for start in range(0, len(blocks), batch_size):
            loss = lm_loss(model, blocks[order[start:start + batch_size]])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        history.append(sum(losses) / len(losses))
        if log:
            log(f"epoch {epoch + 1}/{epochs}  loss {history[-1]:.3f}")
    model.eval()
    return history


def save_lora(model: nn.Module, path: Path | str, config: dict) -> None:
    """Save only the adapter weights (a few hundred KB), plus the settings needed to rebuild them."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    weights = {k: v for k, v in model.state_dict().items() if "lora_" in k}
    torch.save({"config": config, "weights": weights}, path)


def load_lora_model(path: Path | str):
    """Load the base model, inject LoRA with the saved settings and load the trained adapter weights."""
    saved = torch.load(Path(path), map_location="cpu")
    cfg = saved["config"]
    model, tokenizer = load_base_model(cfg["base_model"])
    inject_lora(model, tuple(cfg["targets"]), r=cfg["r"], alpha=cfg["alpha"], dropout=0.0)
    missing = [k for k in saved["weights"] if k not in model.state_dict()]
    if missing:
        raise ValueError(f"Adapter does not match the model: {missing[:3]}")
    model.load_state_dict(saved["weights"], strict=False)
    model.eval()
    return model, tokenizer


def adapter_path(style: str) -> Path:
    return ADAPTER_DIR / f"{style}.pt"


# --- Milestone 5: Generate text from the LoRA-adapted LLM. ----------------------------------------------------

@torch.no_grad()
def generate_llm(model: nn.Module, tokenizer, prompt: str, max_new_tokens: int = 40, temperature: float = 0.8,
                 top_k: int = 40, seed: int | None = None) -> str:
    """Sample a continuation of `prompt` token by token (temperature + top-k), stopping at end-of-text."""
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    model.eval()
    ids = tokenizer.encode(clean_text(prompt)) or [tokenizer.eos_token_id]
    prompt_len = len(ids)
    for _ in range(max_new_tokens):
        logits = model(torch.tensor([ids])).logits[0, -1] / max(temperature, 1e-6)
        if top_k:
            kth = torch.topk(logits, min(top_k, logits.size(-1))).values[-1]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        next_id = torch.multinomial(F.softmax(logits, dim=-1), 1, generator=generator).item()
        if next_id == tokenizer.eos_token_id:
            break
        ids.append(next_id)
    if len(ids) == prompt_len:
        return clean_text(prompt)
    return clean_text(tokenizer.decode(ids).split("\n")[0])   # the prompt never contains a newline


# --- Milestone 6: Provide a qualitative comparison interface for generated texts from both models. -----------

def text_stats(text: str, corpus: list[str]) -> dict:
    """Simple numbers to read next to the text.

    - words: length of the output
    - distinct_2: share of word pairs that are different (low = repetitive)
    - copied_4: share of 4-word sequences that appear word for word in the corpus (high = memorised)
    """
    tokens = [t for t in tokenize(text) if t not in ".,!?;:"]
    pairs = list(zip(tokens, tokens[1:]))
    grams = [tuple(tokens[i:i + 4]) for i in range(len(tokens) - 3)]
    corpus_grams = set()
    for passage in corpus:
        words = [t for t in tokenize(passage) if t not in ".,!?;:"]
        corpus_grams.update(tuple(words[i:i + 4]) for i in range(len(words) - 3))
    return {
        "words": len(tokens),
        "distinct_2": round(len(set(pairs)) / len(pairs), 2) if pairs else 0.0,
        "copied_4": round(sum(g in corpus_grams for g in grams) / len(grams), 2) if grams else 0.0,
    }


def compare(prompt: str, generators: dict[str, callable], corpus: list[str], samples: int = 2,
            seed: int = 0) -> dict:
    """Run every generator on the same prompt. `generators` maps a label to `fn(prompt, seed) -> text`."""
    results = {}
    for label, fn in generators.items():
        outputs = [fn(prompt, seed + i) for i in range(samples)]
        results[label] = [{"text": t, **text_stats(t, corpus)} for t in outputs]
    return {"prompt": prompt, "results": results}


def render_comparison(comparison: dict, width: int = 88) -> str:
    """A plain-text report: each model's samples under its own heading, with the numbers beside them."""
    lines = [f"Prompt: {comparison['prompt']}", "=" * width]
    for label, samples in comparison["results"].items():
        lines.append(f"[{label}]")
        for i, s in enumerate(samples, 1):
            wrapped = textwrap.fill(s["text"], width=width - 6, initial_indent=f"  {i}. ",
                                    subsequent_indent="     ")
            lines.append(wrapped)
            lines.append(f"     words={s['words']}  distinct_2={s['distinct_2']}  copied_4={s['copied_4']}")
        lines.append("-" * width)
    lines.append("distinct_2: higher = less repetitive.  copied_4: higher = more text copied from the corpus.")
    return "\n".join(lines)
