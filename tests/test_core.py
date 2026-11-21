"""One test per milestone in README.md.

The LLM tests use a tiny randomly initialised GPT-2 and a word-level tokenizer built from the corpus, so they run
in seconds and need no download. `python -m app.cli train` uses the real pre-trained distilgpt2.
"""
import math

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from app import core


@pytest.fixture(scope="module")
def passages():
    return core.load_corpus("football")


class WordTokenizer:
    """Just enough of a Hugging Face tokenizer (encode / decode / eos_token_id) for the tests."""

    def __init__(self, passages):
        words = sorted({t for p in passages for t in core.tokenize(p)})
        self.vocab = ["<eos>", "<unk>"] + words
        self.index = {w: i for i, w in enumerate(self.vocab)}
        self.eos_token_id = 0

    def encode(self, text):
        return [self.index.get(t, 1) for t in core.tokenize(text)]

    def decode(self, ids):
        return core.detokenize([self.vocab[i] for i in ids if i > 1])


@pytest.fixture()
def tiny(passages):
    torch.manual_seed(0)
    tok = WordTokenizer(passages)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=len(tok.vocab), n_positions=64, n_embd=32, n_layer=2, n_head=2))
    model.eval()
    return model, tok


# Milestone 1: Prepare a small corpus of sports commentary text.
def test_milestone_1(passages):
    assert {"football", "cricket"} <= set(core.available_styles())
    assert len(passages) >= 40
    assert not any(p.startswith("#") for p in passages)            # comments are skipped
    assert core.clean_text("  It’s   in—the net ") == "It's in, the net"
    assert core.tokenize("What a save! He's down.") == ["what", "a", "save", "!", "he's", "down", "."]
    assert core.detokenize(["what", "a", "save", "!", "he", "is", "down", "."]) == "What a save! He is down."
    stats = core.corpus_stats(passages)
    assert stats["tokens"] > 500 and 0 < stats["vocabulary"] < stats["tokens"]


# Milestone 2: Implement the n-gram model and generate text from it.
def test_milestone_2(passages):
    model = core.NGramModel(3).fit(["the keeper saves it .", "the keeper drops it ."])
    assert model.next_word_probs(["the", "keeper"]) == {"saves": 0.5, "drops": 0.5}
    assert model.next_word_probs(["unknown", "keeper"])                 # backs off to a shorter context
    assert abs(sum(model.next_word_probs(["the"]).values()) - 1) < 1e-9

    model = core.NGramModel(3).fit(passages)
    text = model.generate("He picks", max_words=25, seed=1)
    assert text.startswith("He picks") and len(text.split()) > 3
    assert text == model.generate("He picks", max_words=25, seed=1)     # same seed, same text
    with pytest.raises(ValueError):
        core.NGramModel(0)


# Milestone 3: Load a small pre-trained LLM and implement LoRA adapter injection.
def test_milestone_3(tiny):
    model, tok = tiny
    ids = torch.tensor([tok.encode("he picks it up in midfield")])
    before = model(ids).logits
    wrapped = core.inject_lora(model, ("c_attn",), r=4, alpha=8)
    assert wrapped == 2                                                  # one per transformer block
    assert torch.allclose(before, model(ids).logits, atol=1e-6)         # B starts at zero: nothing changes yet
    trainable, total = core.count_parameters(model)
    assert 0 < trainable < total * 0.2
    assert all(("lora_" in n) == p.requires_grad for n, p in model.named_parameters())
    with pytest.raises(ValueError):
        core.inject_lora(model, ("no_such_layer",))


# Milestone 4: Fine-tune the LoRA adapter on the sports commentary corpus.
def test_milestone_4(tiny, passages, tmp_path):
    model, tok = tiny
    core.inject_lora(model, ("c_attn",), r=4, alpha=8, dropout=0.0)
    frozen = {n: p.clone() for n, p in model.named_parameters() if not p.requires_grad}
    before = core.evaluate_loss(model, tok, passages, block_size=32)
    history = core.train_lora(model, tok, passages, epochs=3, lr=1e-2, block_size=32)
    after = core.evaluate_loss(model, tok, passages, block_size=32)
    assert len(history) == 3 and after < before                         # it learned the corpus
    assert all(torch.equal(p, frozen[n]) for n, p in model.named_parameters() if n in frozen)

    core.save_lora(model, tmp_path / "a.pt", {"r": 4})
    saved = torch.load(tmp_path / "a.pt")
    assert saved["config"] == {"r": 4} and saved["weights"]
    assert all("lora_" in k for k in saved["weights"])                  # only the adapter is saved


# Milestone 5: Generate text from the LoRA-adapted LLM.
def test_milestone_5(tiny):
    model, tok = tiny
    core.inject_lora(model, ("c_attn",), r=4, alpha=8)
    text = core.generate_llm(model, tok, "the ball is", max_new_tokens=10, seed=3)
    assert text.lower().startswith("the ball is")
    assert text == core.generate_llm(model, tok, "the ball is", max_new_tokens=10, seed=3)
    core.set_lora_enabled(model, False)
    assert all(not m.enabled for m in model.modules() if isinstance(m, core.LoRALayer))


# Milestone 6: Provide a qualitative comparison interface for generated texts from both models.
def test_milestone_6(passages):
    copied = core.text_stats(passages[0], passages)
    assert copied["copied_4"] == 1.0 and copied["words"] > 5
    fresh = core.text_stats("purple elephants dance quietly under silver moons", passages)
    assert fresh["copied_4"] == 0.0 and fresh["distinct_2"] == 1.0
    assert core.text_stats("goal goal goal goal goal", passages)["distinct_2"] == 0.25

    gens = {"echo": lambda p, s: f"{p} number {s}", "shout": lambda p, s: p.upper()}
    result = core.compare("He shoots", gens, passages, samples=2)
    assert [r["text"] for r in result["results"]["echo"]] == ["He shoots number 0", "He shoots number 1"]
    report = core.render_comparison(result)
    assert "Prompt: He shoots" in report and "[echo]" in report and "[shout]" in report
    assert "distinct_2=" in report


# Held-out evaluation (part of milestone 6): perplexity per word on passages no model trained on.
def test_held_out_evaluation(tiny, passages):
    train, test = core.split_corpus(passages)
    assert len(test) == len(passages) // 6 and not set(train) & set(test)

    trigram = core.NGramModel(3).fit(train)
    unigram = core.NGramModel(1).fit(train)
    # an unknown word still gets a finite score, and real commentary scores better than shuffled words
    assert math.isfinite(trigram.passage_nll("the zorblax shoots ."))
    shuffled = " ".join(reversed(test[0].split()))
    assert trigram.passage_nll(test[0]) < trigram.passage_nll(shuffled)
    # word order matters to the trigram, so it beats the unigram baseline on held-out text
    assert core.perplexity_per_word(trigram.passage_nll, test) < core.perplexity_per_word(unigram.passage_nll, test)

    model, tok = tiny
    nll = core.llm_passage_nll(model, tok, test[0])
    assert nll > 0 and math.isfinite(core.perplexity_per_word(lambda p: core.llm_passage_nll(model, tok, p), test))

    gens = {"a": lambda p, s: p + " one two", "b": lambda p, s: p}
    summary = core.summarise(core.compare("he shoots", gens, passages, samples=2))
    assert summary["a"]["words"] == 4 and summary["b"]["words"] == 2
