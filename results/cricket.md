## cricket

Trained on 40 passages, scored on 8 held-out passages. Generation numbers average 2 samples for each of 8 held-out openings.

| Model | Held-out perplexity per word (lower is better) | copied_4 (lower = less copying) | distinct_2 | words |
|---|---|---|---|---|
| unigram (simple baseline) | 492.1 | 0.00 | 1.00 | 26 |
| 3-gram | 400.7 | 0.42 | 1.00 | 13 |
| base distilgpt2 (no LoRA) | 896.5 | 0.00 | 0.90 | 25 |
| LoRA distilgpt2 | 359.0 | 0.01 | 0.92 | 18 |

Example, prompt "That was plumb, it":

- **unigram (simple baseline):** That was plumb, it believe takes four. The for pressure. The leg clears the. Keeper out off with will eleven free
- **3-gram:** That was plumb, it is a beautiful cover drive, timed to perfection, and the pressure is building on the field, and it is a no ball, deceives him completely,
- **base distilgpt2 (no LoRA):** That was plumb, it was a little over a quarter of the way back in March the day before the election and that was when I looked over my arm and I saw a lot of it in a large size. I was
- **LoRA distilgpt2:** That was plumb, it was a little over the line...
