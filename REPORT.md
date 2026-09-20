# Speech Evaluation Report — Prisma ASR & Timbre TTS

## Executive Summary

This report covers observations from evaluating **Gnani Prisma ASR** and **Timbre TTS** as black-box services. The evaluation tested both systems across accent variation, numerical expressions, code-switching, domain-specific vocabulary, and conversational speech. No model training or fine-tuning was performed.

---

## Part 1 — Prisma ASR

### Overall Results by Category

| Category | Clips | Accuracy | WER | CER | Subs | Dels | Ins | Avg Latency |
|---|---|---|---|---|---|---|---|---|
| Numeric | 25 | **95.0%** | 5.0% | 2.4% | 3.8% | 1.2% | 0.0% | 0.88 s |
| Domain — Medical | 25 | 86.1% | 13.9% | 4.2% | 8.2% | 1.7% | 4.0% | 0.89 s |
| Accent — Bengali | 20 | 84.0% | 16.9% | 8.1% | 12.8% | 0.0% | 4.1% | 0.55 s |
| Accent — Malayalam | 20 | 84.6% | 20.4% | 7.5% | 14.5% | 0.6% | 5.3% | 0.68 s |
| Baseline — Clean | 25 | 74.5% | 25.5% | 19.8% | 5.5% | **19.9%** | 0.0% | 1.06 s |
| Code-switching | 2 ⚠ | 47.4% | **52.6%** | **51.4%** | 51.7% | 0.9% | 0.0% | 1.93 s |

> ⚠ Only 2 code-switching clips were scored (23 skipped — Devanagari-script output for English audio). Results are indicative only.

---

### Numeric Recognition — Strongest Category

Prisma handled numerical expressions with the highest accuracy (WER 5%, accuracy 95%). Sub-type breakdown:

| Sub-type | n | Accuracy | WER | CER |
|---|---|---|---|---|
| Cardinal | 5 | 98.8% | 1.2% | 0.5% |
| Date | 5 | 96.1% | 3.9% | 0.8% |
| Currency | 5 | 93.8% | 6.2% | 2.0% |
| Ordinal | 10 | 93.1% | 6.9% | 4.3% |

Cardinals and dates are near-perfect. Ordinals and currency show slightly higher error, still well within acceptable range.

---

### Accent — Similar Accuracy, Different Error Profiles

Both Bengali and Malayalam accents achieved ~84% accuracy, but the error distributions differ:

- **Bengali errors are substitution-driven (12.8% Subs, 0% Dels)** — the model transcribes a different word rather than missing speech.
- **Malayalam has higher insertions (5.3% Ins)** alongside substitutions — the model adds extra words or splits tokens.

Representative failures:

| Accent | Reference | Hypothesis | WER |
|---|---|---|---|
| Bengali | `book a hp gas cylinder using my registered contact number 79083174897` | `bouquet hp gas cylinder using my registered contact number 7 9 0 8 3 1 7 4 8 9 7` | 118% |
| Bengali | `moby dick` | `movie t` | 100% |
| Bengali | `house` | `haan` | 100% |
| Malayalam | `39` | `30 9` | 200% |
| Malayalam | `visual` | `vishva` | 100% |

Short single-word clips inflate WER disproportionately. Longer accent clips perform significantly better.

---

### Medical / Domain Vocabulary — Better Than Baseline

Medical terminology (drug names, clinical phrasing) outperformed the baseline clean speech category (WER 13.9% vs 25.5%). The model handled most vocabulary correctly; failures are localised:

| Reference | Hypothesis | WER |
|---|---|---|
| `zerodol and calpol` | `0 doll and calpol` | 66.7% |
| `500 mg also because you're feeling weak take zincovit once in a day` | `5 100 mg also because you are feeling weak take zincovit once in a day` | 30.8% |
| `not having adequate rest okay okay so that continues for better part of the day hmm hmm hmm then` | `not having adequate rest so that continues for better part of the day` | 31.6% |

The dominant failure modes are **number formatting mismatches** and **deletion of fillers** ("okay", "hmm"). Filler omission is often not operationally significant.

---

### Baseline Clean Speech — Highest Deletion Rate

AMI meeting-corpus clips produced the worst non-code-switch results (WER 25.5%, CER 19.8%), driven almost entirely by **deletions (19.9%)**. The model consistently produced a shorter, cleaner transcript — capturing semantic content while omitting disfluencies and filler tokens counted in the reference.

| Reference (truncated) | Hypothesis | WER |
|---|---|---|
| `uh uh ive do ive uh done a little uh research on the internet and not much information about it um about uh interface but uh uh` | `doing a little research on the internet` | 77.8% |
| `yeah i started i started using your um um the speak i dont know if youve seen that in the yeah` | `try that a few times i started using your i dont know` | 76.2% |
| *(non-empty reference)* | *(empty output)* | 100% |

One clip returned an empty hypothesis, likely a VAD or minimum-duration edge case.

---

### Code-Switching — Critical Script Mismatch

The most significant finding: when English audio was submitted, Prisma returned **Devanagari-script transliteration** instead of English text, making the output unusable for the speaker's likely intent.

| Reference (English) | Hypothesis |
|---|---|
| `is worth the read i think put it like that…` | `इस वरथ द रड लइक दट आई ऍम जसट थकग अलउड…` |

The single Hindi→Hindi clip performed acceptably (WER 5.3%, one substitution). The service handles single-language Hindi correctly but defaults to Devanagari script for English phonemes — a real-world gap for mixed-language use cases.

---

### Latency

All categories responded within sub-2-second wall-clock times. Code-switching showed the highest variance (P50 1.19 s, P95 2.68 s). Accent clips were fastest (Bengali P50 0.50 s).

---

## Part 2 — Timbre TTS

### Results

| Category | Clips | Avg WER | Avg CER |
|---|---|---|---|
| General English | 15 | 1.8% | 0.9% |
| Accent Diversity | 10 | 3.2% | 1.5% |
| Technical Vocabulary | 10 | 5.1% | 2.4% |
| Medical | 25 | 7.1% | 3.8% |
| **Overall** | **60** | **4.3%** | **2.1%** |

*WER/CER measured by round-tripping Timbre output through a local Whisper large-v3 server.*

### Observations

- **Overall intelligibility is high** — avg WER of 4.3% across 60 clips; Whisper recovers the intended text with very few errors.
- **General English is near-perfect** — WER 1.8% indicates common English sentences are synthesised without intelligibility issues.
- **Medical terminology introduces the most variation** — 7.1% WER is still low in absolute terms; drug names and clinical terms are the likely source.
- **No major pronunciation or intelligibility failures** were identified during human listening review.

---

## Summary

| | Observation |
|---|---|
| **ASR — best** | Numeric (WER 5.0%, accuracy 95%) |
| **ASR — weakest** | Code-switching (WER 52.6%, Devanagari output for English audio) |
| **ASR — accents** | ~84% accuracy; substitution-heavy, deletion rate near zero |
| **ASR — clean conversational** | 25.5% WER driven by disfluency deletion, not recognition failure |
| **TTS — intelligibility** | High across all categories (avg WER 4.3%) |
| **TTS — medical** | Slightly elevated WER (7.1%), still within acceptable range |

The most actionable ASR finding is the **code-switching script mismatch**: English audio returned as Devanagari is unusable in bilingual contexts. Accent and medical categories perform solidly with predictable, localised error patterns.

---

## Repository

**GitHub:** https://github.com/hemanth-nagesh/AI_Solutions_Engineer
