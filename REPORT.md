# Speech Evaluation Report — Prisma ASR & Timbre TTS

## 1. Executive Summary

This evaluation assesses **Gnani Prisma Automatic Speech Recognition (ASR)** and **Timbre Text-to-Speech (TTS)** as black-box speech services from a customer perspective. No model training or fine-tuning was performed.

The test design focuses on practical conditions that can affect speech quality, including **accent variation, background noise, numerical expressions, code-switching, domain-specific vocabulary, and utterance length**.

For Prisma ASR, audio clips are submitted through the REST API and compared with reference transcripts using **Word Error Rate (WER)** and **Character Error Rate (CER)**. Both metrics are implemented from scratch using **Levenshtein alignment**, enabling detailed analysis of **substitutions (S), deletions (D), and insertions (I)** in addition to aggregate scores.

For Timbre TTS, generated speech is assessed through human listening review, with emphasis on **intelligibility and pronunciation**.

---

## 2. Test Design & Methodology

### Prisma ASR

The ASR evaluation is structured around multiple dimensions so that performance can be analyzed beyond a single overall score:

| Test dimension | Purpose |
|---|---|
| Clean speech | Establish a baseline for recognition quality |
| Accent variation | Assess robustness to pronunciation and speaker variation |
| Background noise | Evaluate recognition under real-world acoustic conditions |
| Numbers | Test recognition of numerical expressions and formatting differences |
| Code-switching | Evaluate mixed-language speech recognition |
| Domain-specific vocabulary | Assess recognition of uncommon and specialized terminology |
| Utterance length | Identify effects related to short and long speech segments |

A single clip may contribute to multiple dimensions. For example, an Indian-accented utterance containing a number and recorded with background noise can be evaluated simultaneously for accent, numerical recognition, and noise robustness.

The evaluation pipeline is:

```text
Audio clip
    ↓
Prisma REST API
    ↓
Predicted transcript
    ↓
Text normalization
    ↓
Levenshtein alignment
    ↓
Substitutions / Deletions / Insertions
    ↓
WER + CER
    ↓
Category- and duration-level analysis
```

### WER and CER

WER is calculated as:

```text
WER = (S + D + I) / N
```

where `N` is the number of words in the reference transcript.

CER uses the same edit-distance principle at the character level. Reporting both metrics helps distinguish word-level recognition failures from smaller character-level differences.

Two normalization levels are considered:

- **Raw normalization:** Unicode normalization, case normalization, and punctuation handling.
- **Normalized scoring:** additional normalization for cases such as number formatting and fillers.

This is particularly useful when evaluating outputs such as **“twenty five”** versus **“25”**, where the difference may be formatting rather than a true recognition failure.

### Error Analysis

The S/D/I breakdown provides the main diagnostic layer:

- **Substitution:** the recognizer outputs a different word or character.
- **Deletion:** expected content is omitted.
- **Insertion:** additional content is produced.

These error types can help identify patterns such as vocabulary/accent-related substitutions, missed short or quiet speech, and spurious words in noisy or silent regions.

Utterances are also grouped by duration (`<5s`, `5–15s`, `15–30s`, `>30s`) so that short-utterance effects and possible long-utterance/chunking effects are not confused with category-specific behavior.

API failures are tracked separately from recognition errors, while latency and Real-Time Factor (RTF) are treated as supplementary operational metrics.

---

## 3. ASR Findings

The evaluation framework is designed to identify **where Prisma breaks down**, rather than relying only on a single aggregate WER.

The main areas of investigation are:

- **Accent variation:** recurring substitutions can indicate difficulty with pronunciation or speaker variation.
- **Background noise:** increased deletions or insertions may indicate reduced robustness when speech is mixed with environmental noise.
- **Numbers and formatting:** differences between spoken and formatted numerical forms can be separated through raw versus normalized scoring.
- **Code-switching:** mixed-language utterances may increase recognition errors because vocabulary and language usage change within the same clip.
- **Domain-specific vocabulary:** uncommon medical or technical terminology is expected to expose vocabulary-related substitutions more clearly than ordinary conversational speech.
- **Short utterances:** a single error can produce a disproportionately high WER, so these clips are analyzed separately.
- **Long or chunked utterances:** longer clips are reviewed for possible segmentation or endpointing effects.

The most useful interpretation should come from the **category-level WER/CER together with the S/D/I distribution and representative error examples**. This avoids treating the headline WER as sufficient evidence of root cause.

> **Results note:** Numerical WER/CER findings should be taken directly from the generated evaluation outputs (`clips.csv`, `report_tables.md`, and `summary.json`). No numerical result is stated here unless it is supported by the recorded run output.

---

## 4. TTS Evaluation — Timbre

Timbre TTS was evaluated through human listening review using general and medical/technical pronunciation cases.

### Human Review Result

**No identifiable pronunciation or intelligibility issues were found in the reviewed samples.** The generated speech was understandable, and no obvious pronunciation failures were identified during manual inspection.

This result should be interpreted as an observation from the reviewed sample set rather than as a claim of universally error-free pronunciation. A larger study with more speakers, accents, and terminology would provide stronger evidence.

A useful next step would be to add an independent objective intelligibility check by passing generated audio through a separate ASR system and comparing the recovered transcript with the original reference text.

---

## 5. Recommendations

### Prisma ASR

1. Prioritize the categories with the highest **category-level WER/CER** once the live results are available.
2. Review recurring **substitutions** to identify systematic accent, numerical, or domain-vocabulary issues.
3. Examine **deletion-heavy short clips** separately for endpointing or minimum-duration effects.
4. Compare **raw versus normalized WER** before treating formatting differences as recognition failures.
5. Use the worst individual clips and top word confusions for targeted error analysis rather than relying only on aggregate metrics.

### Timbre TTS

1. The current human review indicates a good baseline for intelligibility and pronunciation in the tested samples.
2. Expand the evaluation across more accents, speakers, and domain-specific terminology.
3. Add an independent automated intelligibility measure such as WER/CER from a separate ASR system.
4. Where possible, include structured human ratings for **pronunciation, intelligibility, naturalness, and prosody**.

---

## 6. Overall Assessment

The evaluation is designed as a **reproducible, customer-oriented speech testing framework** rather than a single benchmark number.

For ASR, the combination of **WER, CER, S/D/I alignment, normalization-aware scoring, duration analysis, and category-level breakdowns** makes it possible to identify both the frequency and the likely nature of recognition failures.

For TTS, the human review provides a direct assessment of whether the generated speech is understandable and whether any obvious pronunciation problems are audible in the tested samples.

The overall approach is therefore suitable for comparing speech quality under realistic conditions while keeping the evaluation transparent, reproducible, and focused on actionable failure analysis.

---

## Repository

**GitHub:** https://github.com/hemanth-nagesh/AI_Solutions_Engineer

