### contamination-control

The *documented training cutoff* (not the release date) is what matters. Let me verify specifics rather than rely on memory — cutoffs are often poorly documented and differ from release dates.Key caveat already: Mistral never officially documents cutoffs — the "December 2023" figure for Mistral-7B-Instruct-v0.2 is community-inferred from LMSYS, not official, and Dec 2023 is right on your boundary. Let me check the models with *officially documented* earlier cutoffs.Best fit, and it's not particularly close: **Llama-3-8B-Instruct** (`meta-llama/Meta-Llama-3-8B-Instruct`). Its cutoff is March 2023 per Meta's documentation — comfortably inside your boundary — while reasoning is far ahead of anything else in the pre-2024 small-model class. The 70B variant has a Dec 2023 cutoff, but the **8B specifically is March 2023**, which is the lucky detail that makes it ideal here. Only caveat: Llama community license (restrictions only bite above 700M MAU), so it's open-*weight*, not OSI open-source.

Ranked alternatives depending on what you weight:

- **Llama-2-7B/13B** — most conservative margin. Model card states pretraining cutoff of September 2022, with some tuning data more recent, up to July 2023. For contamination control the number that matters is the *finetune* cutoff (~July 2023), not the pretrain one — the model saw whatever was in the SFT/RLHF data. Reasoning is weak by 2026 standards; use only if you want maximal safety margin.

- **Mistral-7B-v0.1** — Apache 2.0 (truly permissive), good reasoning, released Sept 2023. But Mistral *refuses to document cutoffs* on competitive grounds, and the only "December 2023" figure floating around (for Instruct-v0.2) is community-inferred from LMSYS Chat Arena, not official. You cannot certify the cutoff, which is a real problem for a leakage-control component. Prefer base v0.1 over Instruct v0.2 if you go this route.

- **Phi-2** (2.7B, MIT) — the clever pick for *your specific* use case. It's trained largely on synthetic "textbook" + filtered web rather than a raw news/social crawl, so it has structurally **less memorized real-world event knowledge** — which is exactly the lookahead-by-memorization channel you're trying to kill for crypto-event extraction. Smallest and most permissively licensed. Trade-off: thinner world knowledge if your extraction needs entity grounding.

- **Yi-6B/9B** (Apache 2.0) — cutoff ~June 2023, solid reasoning, fully open license; a middle ground between Mistral and Llama-3-8B.

Two caveats that matter more than the model choice, both tying back to your contamination concern:

1. **Nominal cutoff ≠ no leakage.** Research on "effective cutoff" (Pęzik et al., and the CommonCrawl temporal-misalignment work) shows a model's knowledge isn't uniformly cut off at one date — newer crawl dumps contain large amounts of older information, and semantic duplicates leak across the boundary. So a March-2023 nominal cutoff doesn't guarantee the model never saw a later event.

2. **Therefore the cutoff only gets you a prior, not proof.** You still run the empirical audit from before — entity redaction (`LUNA` → `Asset_X`) plus the pre/post-cutoff temporal holdout — *regardless* of which model you pick. The cutoff buys margin; the audit is what actually certifies a given backtest window is clean.

Net: **Llama-3-8B-Instruct for the reasoning/cutoff sweet spot; Phi-2 if you want to minimize event memorization by construction; Mistral-7B-v0.1 only if Apache licensing outweighs the unverifiable cutoff.**

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("microsoft/phi-2")
CTX = 2048

def build_prompt(few_shot: str, item: str, reserve_out: int = 256) -> str:
    """Fit few_shot + item into CTX - reserve_out tokens; truncate item if needed."""
    head = f"{few_shot}\nInstruct: Extract the event as JSON.\nText: "
    tail = "\nOutput:"
    fixed = len(tok(head + tail)["input_ids"])
    budget = max(CTX - reserve_out - fixed, 0)
    item_ids = tok(item)["input_ids"][:budget]
    return head + tok.decode(item_ids) + tail

# minimal test: input+reserve must never exceed the window
assert len(tok(build_prompt("", "token " * 5000))["input_ids"]) <= CTX - 256
```

