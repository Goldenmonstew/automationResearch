# Pre-registration: 48h Sprint Evaluation Protocol (frozen 2026-06-10 ~23:00 CST)

This document freezes the selection rules, claim hierarchy, and disclosure
commitments for the in-progress batch replication of the AI Scientist v2
pipeline, BEFORE the remaining ~8-12 runs complete. Committed for timestamp;
rules below are final and apply regardless of how the remaining data turns out.

Honesty disclosure (freeze provenance): at the time this file was committed
(2026-06-10 22:40 CST), approximately 15 of the 28 runs feeding the analysis
already had scores visible - the six rescue papers (timestamped 06-06 / 06-09)
plus nine sprint runs that finished before 22:40. Strictly, this file is
therefore a mid-analysis plan snapshot (part of the data already existed), not
a pure pre-data pre-registration. The specific numeric ranges in sections 2
and 3 are descriptive ranges fitted to the already-visible data, not blind
predictions; the genuinely blind commitment covers only the remaining ~8-12
runs.

## 1. Rulers (two-ruler discipline)

- The ONLY absolute ruler used for cross-system comparison is the 5-vote
  ensemble (gpt-5.5 ×3 + gpt-4o ×2, NeurIPS conference form, single pass,
  1 few-shot example, gpt-4o meta-review), applied identically to: our papers
  (raw and post-gate versions), the three official showcase papers (clean
  PDFs), and the calibration set (Attention; two marginal real ICLR papers).
- The official papers' historical human workshop scores (e.g. 6.33 for the
  accepted compositional-regularization paper) are a DIFFERENT ruler (human,
  workshop form) and are not compared against machine scores anywhere.
- Known instrument limits, disclosed up front: vote granularity 0.2 on the
  ensemble mean; gpt-5.5 voters exhibit mode-collapse (uniform 2s) in the 2-3
  band. Therefore absolute-score claims are limited to "same band" (parity).
  Any superiority claim is made ONLY via blind pairwise preference voting
  (both presentation orders, >=3 model families as judges, win-rate with
  binomial confidence interval). Caveat on the interval: the pairings are NOT
  iid - each paper faces only n=3 opponents and each pair is repeated 6 times
  (3 judge families x 2 presentation orders), so the trials are clustered and a
  naive binomial/Wilson interval understates the true uncertainty.

## 2. Claim hierarchy (in order; no claim above its evidence)

1. Headline measurements (novel quantities, reported before any score
   comparison): claim-level grounding rate of raw machine-written papers
   (measured 14-40%, independent of experimental completeness), and the
   honesty tax (score change after forcing every claim to be log-grounded;
   measured mean -0.3, range 0 to -0.6).
2. Grounding-gate pass rate: ours 100% of papers converge to >=95% grounded
   within <=3 rewrite rounds (every claim traceable to run artifacts, with a
   per-paper provenance certificate), versus 0/3 for the official showcase
   papers (which cannot undergo log-level audit at all - no public logs - and
   whose fabrications are documented by the vendor's own annotated PDFs).
3. Score claims: raw-vs-raw parity at the top of the band (ours 2.4-2.6,
   official 2.2-2.4); post-gate (honest) versions remain within the official
   raw band. Superiority statements only via the pairwise instrument above.

## 3. Selection rules (showcase)

- Showcase = top 5 papers by post-gate ensemble mean, with ties broken by
  pre-run idea-curation consensus score. Rescue-pipeline papers (rebuilt from
  killed runs) and cap-truncated trees are eligible but flagged as such in
  every table and figure; they are never silently pooled with native runs.
- The FULL score distribution of ALL completed papers (every run started in
  this sprint plus the six rescue papers and the four earlier full-budget
  papers) is published - nothing is dropped. Comparison framing: we publish
  all N and select 5 (~12-20% selection ratio); the official showcase
  published 3 of 43 (7%) with the other 40 undisclosed.
- Runs whose tree finishes too late for the full gate chain (less than ~6h
  before the deadline) enter the distribution as score-only entries and are
  marked as such; they are not showcase-eligible.

## 4. Audit instrument commitments

- The grounding auditor is calibrated before final certificates are issued:
  fault-injection benchmark (known-fabricated claims planted into grounded
  papers, plus known-true claims) reporting precision/recall/false-positive
  rate; cross-family double-audit (deepseek-v3.2) with agreement statistics.
- Final certificates are signed by a held-out auditor configuration (different
  model family and prompt from the one used inside the rewrite loop); claim
  denominators are frozen after a single extraction pass per final text.
  Outcome note (this commitment was not met as stated): in practice all 31/31
  final certificates were signed by the in-loop gpt-5.5 auditor, and the
  held-out final audit (different family/prompt) was downgraded to future work.
- The audit-tier asymmetry is disclosed explicitly: our papers receive
  claim-vs-log strong audit; official papers can only receive text-level
  consistency audit (no public logs), and the two tiers are never reported in
  the same column.

## 5. Honest-rewrite stage (writeup-v2)

Papers may receive one full rewrite from grounded materials only (run
artifacts: journals, data files, stage summaries, figures, audited claim
list). Constraints, frozen now: machine artifacts as sole input; one global
rewrite prompt frozen before batch application (no per-paper tuning after
seeing scores); every rewrite must re-pass the same grounding gate; the
rewrite may delete, reorganize and restate but never add experimental claims;
all three versions (raw / surgically-grounded / rewritten-grounded) are
scored and reported for every paper where the stage is applied.
