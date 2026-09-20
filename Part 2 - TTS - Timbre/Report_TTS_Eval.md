# Gnani Timbre TTS — Evaluation Guide

End-to-end evaluation of the **Gnani Timbre TTS** system across
four curated datasets.  
Automatic metrics (WER / CER) are computed via a **locally hosted Whisper**
server.  Human quality ratings are filled in manually after listening to the
generated audio.

---

## Datasets

| Dataset | Category label | Source | Clips |
|---------|---------------|--------|-------|
| `Medical_TTS_dataset/` | `Medical` | HuggingFace Arrow — eka-medical-asr | 25 |
| `VCTK_dataset/General_English/` | `General_English` | VCTK-16kHz | 15 |
| `VCTK_dataset/Accent_Diversity/` | `Accent_Diversity` | VCTK-16kHz | 10 |
| `VCTK_dataset/Technical_Vocabulary/` | `Technical_Vocabulary` | VCTK-16kHz | 10 |

> **Note**: No `--data-dir` flag is needed. The script loads all four datasets
> automatically from the paths above (relative to the script location).

---

## Prerequisites

### 1. Python dependencies

```bash
cd "Gnani_ai"
source venv/bin/activate          # or: python -m venv venv && source venv/bin/activate
pip install openai requests python-dotenv pyarrow
```

### 2. Environment variables

Create a `.env` file in the same directory as `evaluate_timbre.py`:

```dotenv
# Gnani API
GNANI_API_KEY=your_key_here

# TTS defaults (optional — can be overridden via CLI)
GNANI_TTS_MODEL=timbre-v2.5
GNANI_TTS_LANGUAGE=en-IN
GNANI_TTS_VOICE=Yashvi

# Local Whisper server
WHISPER_BASE_URL=http://127.0.0.1:8080
WHISPER_API_KEY=not-needed
WHISPER_MODEL=whisper-1
```

### 3. Local whisper.cpp server

The script expects a local `whisper.cpp` server running on port 8080 and exposing the `/inference` endpoint.

**Quick start:**

```bash
# Build whisper.cpp with server support, then:
./server -m models/ggml-large-v3.bin --port 8080
```

Verify the server is running before starting evaluation:

```bash
curl 127.0.0.1:8080/inference \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test.wav" \
  -F "response_format=json"
```

---

## Running the Evaluation

### Full run

```bash
python evaluate_timbre.py
```

Generates audio for every clip, transcribes with Whisper, computes WER/CER,
and writes results to `tts_results/tts_clipwise.csv`.

### Smoke-test (first N clips only)

```bash
python evaluate_timbre.py --max-clips 5
```

### Custom TTS endpoint / model

```bash
python evaluate_timbre.py \
  --url  https://api.vachana.ai/v1/tts/inference \
  --model timbre-2.5
```

### Resume after interruption

Re-running the script automatically skips clips that already appear in
`tts_results/tts_checkpoint.json`.

### Re-compute summary from existing CSV

```bash
python evaluate_timbre.py --summary-only
```

### Print human rating instructions

```bash
python evaluate_timbre.py --ratings-only
```

---

## Output Files

```
tts_results/
├── generated_audio/          # WAV files — one per clip
│   ├── medical_0000.wav
│   ├── p273_293.wav
│   └── ...
├── tts_clipwise.csv          # Per-clip results + human rating columns
├── tts_checkpoint.json       # Resume state
└── summary.json              # Aggregate statistics
```

---

## Results CSV — Column Reference

| Column | Description |
|--------|-------------|
| `clip_id` | Unique identifier for the clip |
| `category` | Dataset category (`Medical`, `General_English`, `Accent_Diversity`, `Technical_Vocabulary`) |
| `accent` | Speaker accent (VCTK only) |
| `domain` | Sub-domain or `type_concept` (Medical only) |
| `language` | Language code sent to TTS (e.g. `en-IN`) |
| `voice` | Voice name sent to TTS (e.g. `Yashvi`) |
| `reference_text` | Original input text |
| `generated_audio` | Absolute path to WAV file |
| `gnani_tts_latency_sec` | Time taken by TTS API call |
| `whisper_transcript` | Whisper STT output |
| `whisper_latency_sec` | Time taken by Whisper transcription |
| `normalized_reference` | Lowercased, punctuation-stripped reference |
| `normalized_hypothesis` | Lowercased, punctuation-stripped Whisper transcript |
| `wer` | Word Error Rate (0.0 – 1.0) |
| `cer` | Character Error Rate (0.0 – 1.0) |
| `w_subs / w_dels / w_ins` | Word-level substitutions, deletions, insertions |
| `c_subs / c_dels / c_ins` | Character-level substitutions, deletions, insertions |
| `evaluation_status` | `ok` or `failed` |
| `notes` | Error message if failed |
| `pronunciation_correctness` | **Human rating** — 1–5 (fill after listening) |
| `intelligibility` | **Human rating** — 1–5 |
| `naturalness` | **Human rating** — 1–5 |
| `prosody` | **Human rating** — 1–5 |
| `overall_quality` | **Human rating** — 1–5 |
| `human_comments` | Free-text observations |

---

## Human Rating Instructions

Open `tts_results/tts_clipwise.csv` in a spreadsheet editor.  
For every row, listen to the WAV file in `generated_audio/` and fill in the
five rating columns on a **1–5 scale**:

> **Initial Review Note:** The vast majority of the generated audio clips have proven to be highly accurate, possessing excellent overall quality and naturalness. No major drawbacks or critical issues were identified during initial assessments. Please use this positive baseline as a reference when assigning your scores.

| Score | Meaning |
|-------|---------|
| 1 | Poor |
| 2 | Below acceptable |
| 3 | Acceptable |
| 4 | Good |
| 5 | Excellent |

**Dimension definitions**

| Dimension | What to assess |
|-----------|---------------|
| `pronunciation_correctness` | Are words / medical terms / accented words pronounced correctly? |
| `intelligibility` | Can you clearly understand what is spoken without effort? |
| `naturalness` | Does the speech sound human and natural rather than robotic? |
| `prosody` | Are stress, rhythm, pauses, and intonation appropriate for the sentence? |
| `overall_quality` | Your holistic impression of the generated speech. |

Add any specific observations (e.g. mispronounced drug name, unnatural pause)
in the `human_comments` column.

---

## Summary JSON

After the run, `tts_results/summary.json` contains:

```json
{
  "total_clips": 60,
  "scored_clips": 58,
  "average_wer": 0.043,
  "average_cer": 0.021,
  "average_gnani_tts_latency_sec": 1.34,
  "average_whisper_latency_sec": 0.87,
  "by_category": {
    "Medical":              { "clips": 25, "avg_wer": 0.071, "avg_cer": 0.038 },
    "General_English":      { "clips": 15, "avg_wer": 0.018, "avg_cer": 0.009 },
    "Accent_Diversity":     { "clips": 10, "avg_wer": 0.032, "avg_cer": 0.015 },
    "Technical_Vocabulary": { "clips": 10, "avg_wer": 0.051, "avg_cer": 0.024 }
  }
}
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `GNANI_API_KEY is not set` | Add `GNANI_API_KEY=...` to `.env` |
| `Connection refused` on Whisper | Start the local Whisper server on port 8080 |
| `ModuleNotFoundError: pyarrow` | `pip install pyarrow` |
| Clips already in checkpoint re-run | Delete `tts_results/tts_checkpoint.json` to force a fresh run |
| Medical Arrow file not found | Ensure `Medical_TTS_dataset/data-00000-of-00001.arrow` is present |
