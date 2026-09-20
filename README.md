Please find the report at [REPORT.md](REPORT.md).

# AI Solutions Engineer – Speech Evaluation

This repository contains the evaluation work for:

- **Part 1 – ASR:** Gnani Prisma Speech-to-Text evaluation
- **Part 2 – TTS:** Timbre Text-to-Speech evaluation

The project is designed as a customer-side, black-box evaluation. No model training or fine-tuning is performed.

## Repository Structure

```text
AI_Solutions_Engineer/
│
├── REPORT.md
├── README.md
├── requirements.txt
├── .example.env
│
├── Part 1 - ASR - Prisma/
│   ├── evaluate_prisma.py
│   └── ...
│
└── Part 2 - TTS - Timbre/
    └── evaluate_timbre.py
    |__...
```

## 1. Prerequisites

Recommended:

- Python 3.10+
- Git
- A virtual environment
- Gnani Prisma API access for Part 1
- Timbre API/access required by the Part 2 script

Clone the repository:

```bash
git clone https://github.com/hemanth-nagesh/AI_Solutions_Engineer.git
cd AI_Solutions_Engineer
```

## 2. Create and Activate a Virtual Environment

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### Windows

```powershell
python -m venv venv
venv\Scripts\activate
```

## 3. Install Dependencies

From the repository root:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Environment Variables

Copy the example environment file:

```bash
cp .example.env .env
```

Open `.env` and provide the required credentials.

For Prisma, set:

```bash
GNANI_API_KEY=your_gnani_api_key
```

**Do not commit `.env` or any real API keys to GitHub.**

The code should read secrets from environment variables rather than hard-coding them.

---

# Part 1 – Prisma ASR Evaluation

## Purpose

Part 1 evaluates Gnani Prisma through its REST Speech-to-Text API using a locally maintained test set.

The evaluation calculates:

- Word Error Rate (WER)
- Character Error Rate (CER)
- Substitutions (S)
- Deletions (D)
- Insertions (I)
- Per-clip results
- Category-level summaries
- Duration-level summaries
- Latency / Real-Time Factor where available

WER and CER are implemented from scratch using Levenshtein alignment.

## Dataset Layout

The ASR evaluator expects a top-level folder containing category folders. For example:

```text
data/
├── accent/
├── noise/
├── numbers/
├── code_switching/
└── medical/
```

Each audio clip must have a corresponding reference transcript, using one of the supported manifest/file formats documented in `evaluate_prisma.py`.

Example using a sibling text file:

```text
data/
└── medical/
    ├── clip01.wav
    └── clip01.txt
```

The transcript file contains the reference text spoken in the audio clip.

## Prisma API Setup

Set the API key before running the evaluator:

```bash
export GNANI_API_KEY="your-key"
```

On Windows PowerShell:

```powershell
$env:GNANI_API_KEY="your-key"
```

## Run the Prisma Self-Test

The self-test does not call the API. It validates the metric, alignment, and normalization logic.

```bash
cd "Part 1 - ASR - Prisma"
python evaluate_prisma.py --self-test
```

## Dry Run

Use the mock backend to validate the pipeline without consuming API credits:

```bash
python evaluate_prisma.py --data-dir data --backend mock
```

The mock backend is only for testing the evaluation pipeline. Its WER/CER values are intentionally not meaningful model-performance results.

## Small Live Test

Run a small test before the complete evaluation:

```bash
python evaluate_prisma.py \
    --data-dir data \
    --max-per-category 3
```

This is recommended first to verify:

- API key
- API endpoint
- request format
- audio format
- language configuration
- response parsing

## Full Evaluation

Run the complete test set:

```bash
python evaluate_prisma.py --data-dir data
```

## Medical Glossary Evaluation

For domain-specific terminology, a glossary can be supplied:

```bash
python evaluate_prisma.py \
    --data-dir data \
    --glossary medical_terms.txt
```

## Number Formatting Experiment

To compare raw/verbatim output with formatted transcription output:

```bash
python evaluate_prisma.py \
    --data-dir data \
    --format verbatim \
    --out results_verbatim
```

Run the corresponding formatted mode separately and compare the resulting WER/CER.

## Cached Transcripts

Prisma responses are cached so that metric/normalization experiments can be repeated without making the same API calls again.

Typical cache location:

```text
results/transcripts_cache_prisma.jsonl
```

Delete the cache only when fresh API requests are required.

## Prisma Outputs

The evaluation produces results such as:

```text
results/
├── clips.csv
├── report_tables.md
├── summary.json
├── wer_by_category.png
└── transcripts_cache_prisma.jsonl
```

### Important result files

- `clips.csv` – per-clip reference, hypothesis, normalized text, S/D/I, WER, CER, duration, latency, and status.
- `report_tables.md` – summary tables, error analysis, worst clips, and glossary results.
- `summary.json` – machine-readable aggregate results.
- `wer_by_category.png` – visual comparison of WER across categories.

---

# Part 2 – Timbre TTS Evaluation

## Purpose

Part 2 evaluates the Timbre Text-to-Speech output using human listening review.

The review focuses on:

- Pronunciation
- Intelligibility
- Naturalness
- Technical/medical terminology
- Overall speech quality

The current human review found **no identifiable pronunciation or intelligibility issues in the reviewed samples**.

Because this is a limited human review, the observation should be interpreted as a result for the evaluated samples rather than a claim of perfect TTS quality across all possible inputs.

## Running the TTS Evaluation

Move into the TTS directory:

```bash
cd "Part 2 - TTS - Timbre"
```

Run the TTS evaluation script provided in this directory.

For example, if the script is named `evaluate_timbre.py`:

```bash
python evaluate_timbre.py
```

If the script uses command-line arguments, view the available options with:

```bash
python evaluate_timbre.py --help
```

> **Note:** Use the exact Python filename present in `Part 2 - TTS - Timbre/` if it differs from `evaluate_timbre.py`.

## TTS Dataset

The TTS evaluation includes general and medical/technical pronunciation cases. Medical speech samples are used to check terminology that can be harder to pronounce correctly than common English words.

A typical TTS evaluation flow is:

```text
Reference text
      ↓
Timbre TTS
      ↓
Generated audio
      ↓
Human listening review
      ↓
Pronunciation / Intelligibility / Naturalness
```

## Optional Objective Intelligibility Check

A future extension can pass generated TTS audio through an independent ASR system and compare the resulting transcript with the source text:

```text
Reference text
      ↓
Timbre TTS
      ↓
Generated audio
      ↓
Independent ASR
      ↓
WER / CER
```

This complements human review but does not replace human assessment of naturalness and prosody.

---

# Reproducibility and Safety

- Do not commit API keys or `.env` files.
- Do not commit large audio datasets unless the dataset license explicitly permits redistribution.
- Keep the original reference transcripts unchanged; document any normalization separately.
- Record the dataset version, model/API configuration, and evaluation date when producing final results.
- Run the Prisma self-test and a small live test before a full API run.

## Submission

The requested deliverables are:

1. **Code** – Python evaluation code covering Parts 1 and 2.
2. **Report** – [`REPORT.md`](REPORT.md) containing the test design, methodology, findings, and recommendations.
3. **GitHub repository** – this repository contains the code and documentation.

Repository: https://github.com/hemanth-nagesh/AI_Solutions_Engineer
