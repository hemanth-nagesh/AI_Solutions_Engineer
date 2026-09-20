"""
TTS evaluation pipeline for Gnani Timbre / TTS.

Datasets evaluated
------------------
  Medical_TTS_dataset/                    (HuggingFace Arrow file — medical dictation)
  VCTK_dataset/General_English/           manifest.jsonl
  VCTK_dataset/Accent_Diversity/          manifest.jsonl
  VCTK_dataset/Technical_Vocabulary/      manifest.jsonl

Pipeline
--------
  Reference text
      -> Gnani TTS
      -> generated audio
      -> local Whisper-compatible server  (STT for WER / CER)
      -> transcript
      -> WER / CER computed from scratch
      -> human rating columns left blank for manual fill-in

Automatic metrics (Whisper-based):   WER, CER
Human metrics (fill in CSV after listening):
    pronunciation_correctness, intelligibility, naturalness,
    prosody, overall_quality

Expected local Whisper-compatible server
----------------------------------------
The script expects a local whisper.cpp server running on port 8080 exposing the `/inference` endpoint.
It sends a POST request with `multipart/form-data` containing the audio file.
"""

import argparse
import csv
import json
import os
import re
import time
import random
from pathlib import Path

import pyarrow.ipc as ipc
import requests
from dotenv import load_dotenv


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv(override=True)

BASE_DIR = Path(__file__).parent

GNANI_API_KEY = os.getenv("GNANI_API_KEY")

# Gnani TTS REST endpoint.
DEFAULT_TTS_URL = "https://api.vachana.ai/api/v1/tts/inference"

# Local Whisper-compatible OpenAI API server (STT for WER/CER).
WHISPER_BASE_URL = os.getenv("WHISPER_BASE_URL", "http://127.0.0.1:8080")
WHISPER_API_KEY  = os.getenv("WHISPER_API_KEY",  "not-needed")
WHISPER_MODEL    = os.getenv("WHISPER_MODEL",     "whisper-1")

DEFAULT_MODEL    = os.getenv("GNANI_TTS_MODEL",    "timbre-v2.5")
DEFAULT_LANGUAGE = os.getenv("GNANI_TTS_LANGUAGE", "en-IN")
DEFAULT_VOICE    = os.getenv("GNANI_TTS_VOICE",    "Yashvi")

OUTPUT_DIR      = BASE_DIR / "tts_results"
AUDIO_DIR       = OUTPUT_DIR / "generated_audio"
RESULTS_CSV     = OUTPUT_DIR / "tts_clipwise.csv"
CHECKPOINT_PATH = OUTPUT_DIR / "tts_checkpoint.json"
SUMMARY_PATH    = OUTPUT_DIR / "summary.json"

REQUEST_TIMEOUT = 120
MAX_RETRIES     = 2

# ---- fixed dataset paths (relative to BASE_DIR) ------------

MEDICAL_ARROW_PATH = BASE_DIR / "Medical_TTS_dataset" / "data-00000-of-00001.arrow"

VCTK_MANIFESTS = {
    "General_English":      BASE_DIR / "VCTK_dataset" / "General_English"      / "manifest.jsonl",
    "Accent_Diversity":     BASE_DIR / "VCTK_dataset" / "Accent_Diversity"     / "manifest.jsonl",
    "Technical_Vocabulary": BASE_DIR / "VCTK_dataset" / "Technical_Vocabulary" / "manifest.jsonl",
}


# ============================================================
# HUMAN RATING SCALE
# ============================================================

HUMAN_RATING_COLUMNS = [
    "pronunciation_correctness",
    "intelligibility",
    "naturalness",
    "prosody",
    "overall_quality",
]


# ============================================================
# NORMALISATION
# ============================================================

_PUNCT_RE = re.compile(r"[^\w\s]")


def normalize(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = _PUNCT_RE.sub("", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ============================================================
# LEVENSHTEIN
# ============================================================

def levenshtein(seq_a, seq_b):

    n = len(seq_a)
    m = len(seq_b)

    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if seq_a[i - 1] == seq_b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j - 1],
                    dp[i - 1][j],
                    dp[i][j - 1],
                )

    substitutions = deletions = insertions = 0
    i, j = n, m

    while i > 0 or j > 0:
        if (
            i > 0
            and j > 0
            and seq_a[i - 1] == seq_b[j - 1]
            and dp[i][j] == dp[i - 1][j - 1]
        ):
            i -= 1
            j -= 1
        elif (
            i > 0
            and j > 0
            and dp[i][j] == dp[i - 1][j - 1] + 1
        ):
            substitutions += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            deletions += 1
            i -= 1
        else:
            insertions += 1
            j -= 1

    return dp[n][m], substitutions, deletions, insertions


def calculate_metrics(reference, hypothesis):

    ref_words = reference.split()
    hyp_words = hypothesis.split()
    _, ws, wd, wi = levenshtein(ref_words, hyp_words)

    ref_chars = list(reference.replace(" ", ""))
    hyp_chars = list(hypothesis.replace(" ", ""))
    _, cs, cd, ci = levenshtein(ref_chars, hyp_chars)

    return {
        "wer":   (ws + wd + wi) / max(len(ref_words), 1),
        "cer":   (cs + cd + ci) / max(len(ref_chars), 1),
        "w_subs": ws,
        "w_dels": wd,
        "w_ins":  wi,
        "c_subs": cs,
        "c_dels": cd,
        "c_ins":  ci,
        "reference_word_count": len(ref_words),
    }


# ============================================================
# CHECKPOINT
# ============================================================

def load_checkpoint():
    if not CHECKPOINT_PATH.exists():
        return {}
    try:
        return json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_checkpoint(data):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CHECKPOINT_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(CHECKPOINT_PATH)


# ============================================================
# DATASET LOADING
# ============================================================

def _make_record(clip_id, category, text, accent="", domain="", language=None, voice=None):
    """Return a normalised record dict shared by all loaders."""
    return {
        "clip_id":        str(clip_id),
        "category":       category,
        "reference_text": text,
        "language":       language or DEFAULT_LANGUAGE,
        "voice":          voice    or DEFAULT_VOICE,
        "accent":         accent,
        "domain":         domain or category,
    }


def load_medical_dataset():
    """
    Load texts from Medical_TTS_dataset Arrow file.
    Uses the `text` column; category = 'Medical'.
    """
    if not MEDICAL_ARROW_PATH.exists():
        print(f"[WARN] Medical Arrow file not found: {MEDICAL_ARROW_PATH}")
        return []

    with open(MEDICAL_ARROW_PATH, "rb") as f:
        table = ipc.open_stream(f).read_all()

    records = []
    names = table.schema.names

    for idx in range(len(table)):
        text = str(table["text"][idx]).strip()
        if not text or text.lower() == "none":
            continue

        type_concept = (
            str(table["type_concept"][idx])
            if "type_concept" in names
            else "medical"
        )

        records.append(
            _make_record(
                clip_id  = f"medical_{idx:04d}",
                category = "Medical",
                text     = text,
                domain   = type_concept,
            )
        )

    print(f"Loaded {len(records)} Medical records.")
    return records


def load_vctk_manifest(category, path):
    """Load a VCTK manifest.jsonl file."""
    if not path.exists():
        print(f"[WARN] VCTK manifest not found: {path}")
        return []

    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get("reference_text") or row.get("text", "")
            if not text:
                continue
            records.append(
                _make_record(
                    clip_id  = row.get("clip_id", f"{category}_{len(records):04d}"),
                    category = category,
                    text     = text,
                    accent   = row.get("accent", ""),
                    domain   = category,
                )
            )

    print(f"Loaded {len(records)} VCTK '{category}' records.")
    return records


def load_all_datasets():
    """Load all four evaluation datasets and return a combined list."""
    records = []
    records.extend(load_medical_dataset())
    for category, path in VCTK_MANIFESTS.items():
        records.extend(load_vctk_manifest(category, path))
    print(f"\nTotal: {len(records)} TTS test clips.\n")
    return records


# ============================================================
# GNANI TTS
# ============================================================

def call_gnani_tts(text, language, voice, model, url):

    if not GNANI_API_KEY:
        raise RuntimeError("GNANI_API_KEY is not set.")

    payload = {
        "text":        text,
        "voice":       voice,
        "model":       model,
        "language":    language,
        "speed":       1.0,
        "audio_config": {
            "sample_rate": 48000,
            "num_channels": 1,
            "sample_width": 2,
            "encoding": "linear_pcm",
            "container": "wav"
        }
    }

    headers = {
        "X-API-Key-ID": GNANI_API_KEY,
        "Content-Type":  "application/json",
    }

    last_error = None

    for attempt in range(MAX_RETRIES + 1):
        try:
            start    = time.perf_counter()
            response = requests.post(url, json=payload, headers=headers, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            latency  = time.perf_counter() - start
            return response.content, latency
        except Exception as exc:
            last_error = exc
            if attempt >= MAX_RETRIES:
                break
            wait = 2 ** attempt + random.uniform(0, 0.5)
            print(f"    Retry {attempt + 1}: {exc}; waiting {wait:.1f}s")
            time.sleep(wait)

    raise RuntimeError(f"TTS failed: {last_error}")


# ============================================================
# LOCAL WHISPER (STT for WER / CER)
# ============================================================

def whisper_transcribe(audio_path):
    """
    Send audio to the locally hosted whisper.cpp server.
    Returns (transcript_text, latency_seconds).
    """
    url = WHISPER_BASE_URL.rstrip('/')
    if url.endswith('/v1'):
        url = url[:-3]
    url = f"{url}/inference"
    
    start = time.perf_counter()
    
    with open(audio_path, "rb") as f:
        files = {"file": f}
        data = {"response_format": "json"}
        resp = requests.post(url, files=files, data=data)
        
    resp.raise_for_status()
    result = resp.json()

    latency = time.perf_counter() - start
    return result.get("text", "").strip(), latency


# ============================================================
# CSV
# ============================================================

CSV_COLUMNS = [
    "clip_id",
    "category",
    "accent",
    "domain",
    "language",
    "voice",
    "reference_text",
    "generated_audio",
    "gnani_tts_latency_sec",
    "whisper_transcript",
    "whisper_latency_sec",
    "normalized_reference",
    "normalized_hypothesis",
    "wer",
    "cer",
    "w_subs",
    "w_dels",
    "w_ins",
    "c_subs",
    "c_dels",
    "c_ins",
    "evaluation_status",
    "notes",

    # Human evaluation — fill these in after listening
    "pronunciation_correctness",
    "intelligibility",
    "naturalness",
    "prosody",
    "overall_quality",
    "human_comments",
]


def initialize_csv():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if RESULTS_CSV.exists():
        return
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=CSV_COLUMNS).writeheader()


def append_result(result):
    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore").writerow(result)


# ============================================================
# MAIN EVALUATION
# ============================================================

def evaluate(args):

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    initialize_csv()

    records    = load_all_datasets()
    checkpoint = load_checkpoint()

    if args.max_clips:
        records = records[: args.max_clips]

    print(f"Evaluating {len(records)} clips...\n")

    for index, record in enumerate(records, start=1):

        clip_id = record["clip_id"]

        if clip_id in checkpoint:
            print(f"[{index}/{len(records)}] {clip_id}  (already done — skip)")
            continue

        print(f"[{index}/{len(records)}] {clip_id}")
        print(f"  Category : {record['category']}")
        print(f"  Text     : {record['reference_text'][:80]}")

        try:

            # --------------------------------------------------
            # 1. Gnani Timbre TTS
            # --------------------------------------------------

            audio_bytes, tts_latency = call_gnani_tts(
                text     = record["reference_text"],
                language = record["language"],
                voice    = record["voice"],
                model    = args.model,
                url      = args.url,
            )

            audio_path = AUDIO_DIR / f"{clip_id}.wav"
            audio_path.write_bytes(audio_bytes)

            # --------------------------------------------------
            # 2. Local Whisper STT
            # --------------------------------------------------

            whisper_text, whisper_latency = whisper_transcribe(audio_path)

            # --------------------------------------------------
            # 3. WER / CER
            # --------------------------------------------------

            ref_norm = normalize(record["reference_text"])
            hyp_norm = normalize(whisper_text)
            metrics  = calculate_metrics(ref_norm, hyp_norm)

            # --------------------------------------------------
            # 4. Save
            # --------------------------------------------------

            result = {
                "clip_id":               clip_id,
                "category":              record["category"],
                "accent":                record["accent"],
                "domain":                record["domain"],
                "language":              record["language"],
                "voice":                 record["voice"],
                "reference_text":        record["reference_text"],
                "generated_audio":       str(audio_path),
                "gnani_tts_latency_sec": round(tts_latency,    4),
                "whisper_transcript":    whisper_text,
                "whisper_latency_sec":   round(whisper_latency, 4),
                "normalized_reference":  ref_norm,
                "normalized_hypothesis": hyp_norm,
                **metrics,
                "evaluation_status": "ok",
                "notes":             "",

                # Intentionally blank — fill after listening
                "pronunciation_correctness": "",
                "intelligibility":           "",
                "naturalness":               "",
                "prosody":                   "",
                "overall_quality":           "",
                "human_comments":            "",
            }

            append_result(result)

            checkpoint[clip_id] = {
                "audio_path":       str(audio_path),
                "whisper_transcript": whisper_text,
                "tts_latency":      tts_latency,
                "whisper_latency":  whisper_latency,
            }

            save_checkpoint(checkpoint)

            print(f"  WER  : {metrics['wer'] * 100:.2f}%")
            print(f"  CER  : {metrics['cer'] * 100:.2f}%")
            print(f"  Audio: {audio_path}")

        except Exception as exc:
            print(f"  FAILED: {exc}")
            append_result({
                "clip_id":           clip_id,
                "category":          record["category"],
                "reference_text":    record["reference_text"],
                "evaluation_status": "failed",
                "notes":             str(exc),
            })


# ============================================================
# HUMAN RATING INSTRUCTIONS
# ============================================================

def print_rating_instructions():
    print("""
============================================================
HUMAN RATING — fill in tts_results/tts_clipwise.csv
============================================================

For every generated audio file, listen once or twice, then
enter a score (1–5) for each dimension:

  1 = Poor
  2 = Below acceptable
  3 = Acceptable
  4 = Good
  5 = Excellent

Dimensions
----------
  pronunciation_correctness — words / terms pronounced correctly?
  intelligibility           — clearly understandable?
  naturalness               — natural rather than robotic?
  prosody                   — stress, rhythm, pauses appropriate?
  overall_quality           — overall quality.

Add observations in human_comments.

Example
-------
  pronunciation_correctness = 4
  intelligibility           = 5
  naturalness               = 4
  prosody                   = 4
  overall_quality           = 4
  human_comments: "Medical term 'metformin' was clear."

============================================================
""")


# ============================================================
# SUMMARY
# ============================================================

def generate_summary():

    if not RESULTS_CSV.exists():
        return

    with open(RESULTS_CSV, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    valid = [r for r in rows if r.get("wer")]

    if not valid:
        return

    def average(field):
        vals = []
        for row in valid:
            try:
                vals.append(float(row[field]))
            except Exception:
                pass
        return sum(vals) / len(vals) if vals else None

    # Per-category breakdown
    categories = {}
    for row in valid:
        cat = row.get("category", "unknown")
        categories.setdefault(cat, []).append(row)

    cat_summary = {}
    for cat, cat_rows in categories.items():
        wers = [float(r["wer"]) for r in cat_rows if r.get("wer")]
        cers = [float(r["cer"]) for r in cat_rows if r.get("cer")]
        cat_summary[cat] = {
            "clips":   len(cat_rows),
            "avg_wer": round(sum(wers) / len(wers), 4) if wers else None,
            "avg_cer": round(sum(cers) / len(cers), 4) if cers else None,
        }

    summary = {
        "total_clips":                len(rows),
        "scored_clips":               len(valid),
        "average_wer":                average("wer"),
        "average_cer":                average("cer"),
        "average_gnani_tts_latency_sec":  average("gnani_tts_latency_sec"),
        "average_whisper_latency_sec":    average("whisper_latency_sec"),
        "by_category":                cat_summary,
    }

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nSUMMARY")
    print("=" * 50)
    print(f"Total clips  : {summary['total_clips']}")
    print(f"Scored clips : {summary['scored_clips']}")
    print(f"Average WER  : {summary['average_wer']}")
    print(f"Average CER  : {summary['average_cer']}")
    print(f"Avg TTS latency : {summary['average_gnani_tts_latency_sec']} s")
    print("\nBy category:")
    for cat, s in cat_summary.items():
        print(f"  {cat:25s}  clips={s['clips']}  WER={s['avg_wer']}  CER={s['avg_cer']}")


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Gnani Timbre TTS across Medical and VCTK datasets "
            "using locally hosted Whisper for WER/CER, plus manual human ratings."
        )
    )

    parser.add_argument(
        "--url",
        default=DEFAULT_TTS_URL,
        help="Gnani TTS endpoint URL.",
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Gnani TTS model name (default: timbre-2.5).",
    )

    parser.add_argument(
        "--max-clips",
        type=int,
        default=None,
        help="Cap total clips (useful for smoke-testing).",
    )

    parser.add_argument(
        "--ratings-only",
        action="store_true",
        help="Print human rating instructions without running TTS.",
    )

    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Re-compute and print the summary from an existing results CSV.",
    )

    args = parser.parse_args()

    if args.ratings_only:
        print_rating_instructions()
        return

    if args.summary_only:
        generate_summary()
        return

    evaluate(args)
    generate_summary()
    print_rating_instructions()


if __name__ == "__main__":
    main()