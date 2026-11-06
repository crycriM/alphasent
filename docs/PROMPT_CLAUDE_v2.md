# Task: write `docs/WHITEPAPER_v2.md`

You are writing the second version of a quantitative research note. The first
version is `docs/WHITEPAPER.md` (read it in full before writing anything). The
corpus it was built on has since more than tripled, and the number in its
headline cell has moved in a way that looks like progress and is not.

Your job is to write the version that a competent quant reader — someone who
knows statistics but nothing about LLMs, and nothing about this project — can
read and trust.

## Deliverable

Exactly one file: `docs/WHITEPAPER_v2.md`. Do not modify any other file. Do not
write a changelog, a plan, or notes; write the paper.

## Inputs

- `docs/WHITEPAPER.md` — v1. Read every section. v2 must be readable *against* it.
- `docs/EVIDENCE_v2.md` — the evidence pack: every number you may use, its
  provenance, the v1-claim status table, and an explicit list of things you must
  not claim. Read it fully before drafting.
- `data/results/` (refreshed CSVs) and `data/results_baseline_20260729/` (the v1
  vintage). Spot-check at least five numbers you intend to use, by running one of
  the verification commands in §7 of the evidence pack or by reading the CSV
  directly. If a number you want is in neither the pack nor those files, it was
  not measured — write "not measured" instead of a plausible value.

## What v2 is for

v1's three findings were: no LLM signal clears significance; polarity fails
everywhere; a free non-LLM tone baseline beats both models. After the refresh,
the pooled headline IC for phi-4's composite roughly quintuples (+0.016 → +0.082
at 120 h) — which reads like "the signal showed up once we had data". The
evidence pack shows that reading is wrong: split the sample into regimes and the
correlation is negative or zero in every one of them, and removing the regime
means collapses the pooled value from +0.087 to −0.034. The regimes differ in
both news density and mean return, so the pooled number is measuring the
coincidence between them.

The paper's spine is therefore: *what changed, what that appears to mean, and
what it actually means.* Then, honestly, what survives: the overlap-corrected
inference lesson, the flag family as the only place worth further work, and the
narrowed sense in which the tone baseline is still the bar to clear.

## Structure

Keep v1's section numbering and titles wherever the content is the same, so a
reader can diff the two documents; add new sections rather than renumbering.
v1 runs: Abstract, 1 System overview, 2 Data, 3 Methodology (3.1 why IC not
Sharpe, 3.2 overlap-robust inference, 3.3 event study with placebo), 4 Results,
5 timestep, 6 model choice, 7 LLM vs tone, 8 knowledge contamination, 9 event
study, 10 what did not work, 11 limitations.

Requirements:

1. **Abstract** — rewrite completely; it must state the regime-contrast finding,
   because that is the paper's contribution.
2. **A new section titled "Regime contrast: why the headline number moved"** —
   the §4 material of the evidence pack. Explain the arithmetic plainly: two
   regimes, each with its own mean feature and mean return; a pooled correlation
   over both absorbs the difference between them. Give the numbers. State that
   the between-regime component is estimated on three groups, and say what that
   means for anyone tempted to trade it.
3. **A table of v1 claims and their status** — supported / qualified / weakened /
   overturned — with the evidence beside each. Take it from §3 of the pack, but
   write it as prose-plus-table for a reader who has v1 open next to them, and
   name *which new data* moved each one (more RSS weeks, extended OHLCV, or
   nothing).
4. **A section separating three sources of change:** (a) more data, (b) the
   perimeter pre-filter fix, (c) nothing — things that look changed but are not
   (the tone baseline, the redaction audit, the llama-3 arm). This distinction is
   what the user asked for: which new data supports which new conclusion.
5. **Keep v1's exposition of overlap-robust inference and of the naive-vs-corrected
   t-statistic**, with the refreshed numbers (§5 of the pack). It is the part of
   v1 that held up and it is the reason not to trust the 8.80.
6. **Limitations and future work** — carry v1's forward, updated: 5 assets,
   1.2 % extraction failures, ticker false positives, the frozen llama-3 arm,
   three regimes, one prompt version.

## Style rules

- Write for a human. Explain a term the first time it is used (IC, point-in-time,
  Newey–West, fixed effects, CAR) in one clause, not a paragraph.
- No filler, no hype, no "it is worth noting", "delve", "landscape", "robust"
  used loosely, no three-adjective lists, no bullet-per-sentence padding. If a
  sentence carries no information, delete it.
- State magnitudes and samples with every claim: an IC without its n and t is
  not evidence.
- Where v1 was wrong or lucky, say so in the plainest available words. Do not
  soften it and do not apologise at length.
- Prefer a table when comparing versions, prose when explaining a mechanism.
- Length: comparable to v1 (~220 lines) plus the new sections. Longer is fine if
  the extra lines carry numbers or reasoning; padding is not.
- Attribute numbers to their source file in the text where it helps a reader
  verify (e.g. "`data/results_.../car_adjusted.csv`"), but do not let citations
  break the prose.
- Language: English. Audience: the author of v1 (a quant) and anyone reviewing
  the work cold.

## Before you finish

- Re-read your own draft against §6 of the evidence pack ("what v2 must not
  claim") — one pass dedicated to that checklist.
- Confirm every number in the paper appears in the evidence pack or in the two
  results directories.
- Regenerate nothing; the analyses are already run.
