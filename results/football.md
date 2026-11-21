## football

Trained on 40 passages, scored on 8 held-out passages. Generation numbers average 2 samples for each of 8 held-out openings.

| Model | Held-out perplexity per word (lower is better) | copied_4 (lower = less copying) | distinct_2 | words |
|---|---|---|---|---|
| unigram (simple baseline) | 420.4 | 0.00 | 1.00 | 28 |
| 3-gram | 289.9 | 0.44 | 1.00 | 11 |
| base distilgpt2 (no LoRA) | 249.3 | 0.00 | 0.94 | 22 |
| LoRA distilgpt2 | 105.9 | 0.01 | 0.98 | 21 |

Example, prompt "The corner is swung":

- **unigram (simple baseline):** The corner is swung everything outside is the cross is around. Acres! Throw flank the just big the catches score forcing not. Long had net has, to that clean tackle
- **3-gram:** The corner is swung everything forward in search of a winner.
- **base distilgpt2 (no LoRA):** The corner is swung a bit, with a bit of the A on the left, the D on the right and the D on the right. Then it's a big, big swing.
- **LoRA distilgpt2:** The corner is swung a bit wide.
