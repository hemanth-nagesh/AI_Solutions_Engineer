import csv
import io
import json
import os
import re
import sys
import time
from pathlib import Path

import requests
from datasets import load_from_disk, Audio
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# 0.  CONFIGURATION
# ---------------------------------------------------------------------------

load_dotenv()

GNANI_API_KEY = os.environ.get("GNANI_API_KEY", "")   # X-API-Key-ID header

# Gnani Vachana API base
GNANI_BASE_URL = "https://api.vachana.ai"
REST_URL       = f"{GNANI_BASE_URL}/stt/v3"
BATCH_JOBS_URL = f"{GNANI_BASE_URL}/stt/v3/batch/jobs"

BASE_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Dataset map:
#   category -> (folder, transcript_col, id_col, duration_col,
#                audio_col, audio_col_needs_recast,
#                uses_audio_url, subcategory_col, language_code)
#
# audio_col_needs_recast=True  → cast_column(audio_col, Audio(decode=False))
#                                 before iterating (Svarah datasets)
# uses_audio_url=True          → no local bytes; download from row["audio_url"]
#                                 (small/large librispeech duration datasets)
# ---------------------------------------------------------------------------
DATASET_MAP = {
    #  category          folder                           transcript_col   id_col          dur_col         audio_col        recast  url_dl  subcat_col     lang_code
    "baseline_clean":   ("clean_audio_dataset",           "text",          "id",           "audio_length_s","audio",        False,  False,  None,          "en-IN"),
    "duration_short":   ("small_audio_dataset",           "text",          "id",           "audio_length_s","audio_url",    False,  True,   None,          "en-IN"),
    "duration_long":    ("large_audio_dataset",           "text",          "id",           "audio_length_s","audio_url",    False,  True,   None,          "en-IN"),
    "accent_malayalam": ("Malayalam_accent_India_dataset","text",          None,           "duration",      "audio_filepath",True, False,  None,          "ml-IN"),
    "accent_bengali":   ("Bengali_accent_India_dataset",  "text",          None,           "duration",      "audio_filepath",True, False,  None,          "bn-IN"),
    "domain_medical":   ("Medical_ASR_dataset",           "text",          "file_name",    "duration",      "audio",        False,  False,  None,          "en-IN"),
    "numeric":          ("numbers_dataset",               "transcript",    "clip_id",      "audio_length_s","audio",        False,  False,  "number_type", "en-IN"),
    "code_switch":      ("Code_Switch_dataset",           "transcription", "audio_file_name",None,          None,           False,  False,  None,          "en-IN"),
    "noise":            ("Noise_English_dataset",         None,            "sample_id",    None,            "audio",        False,  False,  "noise_type",  "en-IN"),
}

REPORT_PATH      = BASE_DIR / "prisma_eval_report.md"
CSV_PATH         = BASE_DIR / "prisma_eval_clipwise.csv"
CHECKPOINT_PATH  = BASE_DIR / "eval_checkpoint.json"   # clip_id -> transcript

LOW_SAMPLE_THRESHOLD = 10   # categories with < N clips get asterisk in report
POLL_INTERVAL_S      = 15   # seconds between batch job status polls
BATCH_MAX_FILES      = 100  # Gnani Batch hard limit per job
TERMINAL_STATUSES    = {"COMPLETED", "PARTIAL_FAILURE", "FAILED", "START_FAILED", "CANCELLED"}

# ---------------------------------------------------------------------------
# 1.  TEXT NORMALISATION
# ---------------------------------------------------------------------------

_NUM_WORD_MAP = {
    "zero":"0","one":"1","two":"2","three":"3","four":"4","five":"5",
    "six":"6","seven":"7","eight":"8","nine":"9","ten":"10",
    "eleven":"11","twelve":"12","thirteen":"13","fourteen":"14",
    "fifteen":"15","sixteen":"16","seventeen":"17","eighteen":"18",
    "nineteen":"19","twenty":"20","thirty":"30","forty":"40",
    "fifty":"50","sixty":"60","seventy":"70","eighty":"80","ninety":"90",
    "hundred":"100","thousand":"1000","million":"1000000","billion":"1000000000",
}
_NUM_WORD_RE = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in _NUM_WORD_MAP) + r")\b", re.IGNORECASE
)
_PUNCT_RE = re.compile(r"[^\w\s]")


def normalize(text: str) -> str:
    """Lowercase -> number-word digits -> strip punctuation -> collapse whitespace."""
    text = text.lower()
    text = _NUM_WORD_RE.sub(lambda m: _NUM_WORD_MAP[m.group(0).lower()], text)
    text = _PUNCT_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# 2.  WER / CER (from-scratch Levenshtein, no external library)
# ---------------------------------------------------------------------------

def _levenshtein(seq_a, seq_b):
    """
    Levenshtein DP on two sequences (word list or char list).
    Returns (distance, substitutions, deletions, insertions).
    Back-tracks the DP matrix to count S/D/I separately.
    """
    n, m = len(seq_a), len(seq_b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): dp[i][0] = i
    for j in range(m + 1): dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if seq_a[i-1] == seq_b[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1])
    subs = dels = ins = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1]:
            i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            subs += 1; i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            dels += 1; i -= 1
        else:
            ins += 1; j -= 1
    return dp[n][m], subs, dels, ins


def edit_distance_metrics(ref: str, hyp: str) -> dict:
    """
    Compute WER and CER from pre-normalised strings.
    Returns wer, cer, accuracy, and per-type counts.
    """
    rw, hw = ref.split(), hyp.split()
    rc, hc = list(ref.replace(" ", "")), list(hyp.replace(" ", ""))
    wl = max(len(rw), 1)
    _, ws, wd, wi = _levenshtein(rw, hw)
    wer = (ws + wd + wi) / wl
    cl = max(len(rc), 1)
    _, cs, cd, ci = _levenshtein(rc, hc)
    cer = (cs + cd + ci) / cl
    return {
        "wer": wer, "cer": cer, "accuracy": max(0.0, 1.0 - wer),
        "w_subs": ws, "w_dels": wd, "w_ins": wi,
        "c_subs": cs, "c_dels": cd, "c_ins": ci,
        "ref_word_count": len(rw),
    }


# ---------------------------------------------------------------------------
# 3.  CHECKPOINT (save / load transcriptions so restarts skip done clips)
# ---------------------------------------------------------------------------

def load_checkpoint() -> dict:
    """Return {clip_id: transcript_str} from the checkpoint file if it exists."""
    if CHECKPOINT_PATH.exists():
        try:
            return json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_checkpoint(ckpt: dict) -> None:
    """Persist the current checkpoint dict atomically."""
    tmp = CHECKPOINT_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(ckpt, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(CHECKPOINT_PATH)


# ---------------------------------------------------------------------------
# 4.  AUDIO EXTRACTION
# ---------------------------------------------------------------------------

def _audio_bytes_from_value(raw) -> bytes | None:
    """
    Pull raw WAV/audio bytes out of whatever shape the HF column gives us.
    Handles: bytes, dict{bytes, path}, AudioDecoder (after no-decode cast).
    """
    if raw is None:
        return None
    if isinstance(raw, bytes):
        return raw
    if isinstance(raw, dict):
        b = raw.get("bytes")
        if b:
            return b
        p = raw.get("path")
        if p and os.path.isfile(p):
            return open(p, "rb").read()
    return None


def _download_url(url: str) -> bytes | None:
    """Download audio from a public URL. Returns bytes or None on failure."""
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        return r.content
    except Exception as e:
        print(f"      [warn] download failed for {url}: {e}")
        return None


# ---------------------------------------------------------------------------
# 5.  DATA LOADING
# ---------------------------------------------------------------------------

def load_records() -> list[dict]:
    """Load every dataset; return flat list of record dicts."""
    all_records = []

    for category, cfg in DATASET_MAP.items():
        (folder, transcript_col, id_col, dur_col,
         audio_col, needs_recast, uses_url_dl, subcat_col, lang_code) = cfg

        folder_path = BASE_DIR / folder
        print(f"  Loading '{folder}'  ->  '{category}' ...")

        if not folder_path.is_dir():
            print(f"    [WARN] folder not found — skipping.")
            continue

        try:
            ds = load_from_disk(str(folder_path))
        except Exception as e:
            print(f"    [WARN] could not load: {e} — skipping.")
            continue

        # Cast audio column to no-decode so we get raw bytes dict
        if audio_col and audio_col in ds.column_names and needs_recast:
            ds = ds.cast_column(audio_col, Audio(decode=False))

        if transcript_col is None:
            print(f"    [WARN] no transcript column — clips will be skipped during scoring.")

        for idx, row in enumerate(ds):
            # clip_id
            clip_id = str(row[id_col]) if (id_col and id_col in row and row[id_col]) else f"{category}_{idx:04d}"

            # reference text
            ref_text = None
            if transcript_col and transcript_col in row:
                v = row[transcript_col]
                if isinstance(v, str) and v.strip():
                    ref_text = v

            # subcategory
            subcategory = None
            if subcat_col and subcat_col in row and row[subcat_col] is not None:
                subcategory = str(row[subcat_col])

            # duration
            duration_sec = None
            if dur_col and dur_col in row:
                try: duration_sec = float(row[dur_col])
                except (TypeError, ValueError): pass

            # audio bytes
            audio_bytes = None
            if audio_col and audio_col in row:
                if uses_url_dl:
                    # store the URL string; will download lazily per-clip
                    audio_bytes = row[audio_col]   # a URL string
                else:
                    audio_bytes = _audio_bytes_from_value(row[audio_col])

            all_records.append({
                "clip_id":        clip_id,
                "category":       category,
                "subcategory":    subcategory,
                "reference_text": ref_text,
                "audio_bytes":    audio_bytes,   # bytes | URL str | None
                "duration_sec":   duration_sec,
                "lang_code":      lang_code,
                "uses_url_dl":    uses_url_dl,
            })

        print(f"    OK  {len(ds)} records.")

    return all_records


# ---------------------------------------------------------------------------
# 6.  GNANI BATCH API
# ---------------------------------------------------------------------------

def _gnani_headers() -> dict:
    return {"X-API-Key-ID": GNANI_API_KEY}


def _create_batch_job(clip_batch: list[dict]) -> str | None:
    """
    Upload up to BATCH_MAX_FILES clips to a new batch job.
    Returns job_id or None on failure.
    clip_batch items: {clip_id, audio_bytes (bytes), lang_code}
    """
    # All clips in a single job must share a language_code.
    # (We group by lang_code before calling this function.)
    lang = clip_batch[0]["lang_code"]
    config = json.dumps({
        "model": "gnani-prisma-v2.5",
        "language_code": lang,
        "mode": "transcribe",
        "with_diarization": False,
        "is_multi_channel": False,
    })

    files = [("config", (None, config, "application/json"))]
    for item in clip_batch:
        files.append(("files", (f"{item['clip_id']}.wav", item["audio_bytes"], "audio/wav")))

    try:
        r = requests.post(BATCH_JOBS_URL, headers=_gnani_headers(), files=files, timeout=120)
        r.raise_for_status()
        job_id = r.json()["job_id"]
        print(f"    [batch] job created: {job_id}  ({len(clip_batch)} files, lang={lang})")
        return job_id
    except Exception as e:
        print(f"    [ERROR] create batch job failed: {e}")
        return None


def _start_batch_job(job_id: str) -> bool:
    """POST .../start. Returns True on success."""
    try:
        r = requests.post(
            f"{BATCH_JOBS_URL}/{job_id}/start",
            headers=_gnani_headers(), timeout=30
        )
        r.raise_for_status()
        print(f"    [batch] job {job_id} started.")
        return True
    except Exception as e:
        print(f"    [ERROR] start job {job_id} failed: {e}")
        return False


def _poll_until_done(job_id: str) -> str:
    """Poll GET .../jobs/{job_id} until terminal. Returns final status string."""
    while True:
        try:
            r = requests.get(
                f"{BATCH_JOBS_URL}/{job_id}",
                headers=_gnani_headers(), timeout=30
            )
            r.raise_for_status()
            status = r.json().get("status", "UNKNOWN")
            print(f"    [batch] {job_id} status: {status}")
            if status in TERMINAL_STATUSES:
                return status
        except Exception as e:
            print(f"    [WARN] poll error for {job_id}: {e}")
        time.sleep(POLL_INTERVAL_S)


def _collect_transcripts_from_job(job_id: str, clip_batch: list[dict]) -> dict:
    """
    GET .../jobs/{job_id}/files, then download each transcript_url.
    Returns {clip_id: transcript_text}.
    clip_batch is the ordered list we sent so we can match by filename.
    """
    results = {}

    # Build mapping: uploaded filename base (without .wav) -> clip_id
    fname_to_id = {f"{item['clip_id']}.wav": item["clip_id"] for item in clip_batch}

    try:
        r = requests.get(
            f"{BATCH_JOBS_URL}/{job_id}/files",
            headers=_gnani_headers(), timeout=30
        )
        r.raise_for_status()
        files_data = r.json()
    except Exception as e:
        print(f"    [ERROR] get files for {job_id}: {e}")
        return results

    file_list = files_data if isinstance(files_data, list) else files_data.get("files", [])

    for f in file_list:
        original_path = f.get("original_path", "")
        fname = Path(original_path).name
        clip_id = fname_to_id.get(fname)
        if clip_id is None:
            continue

        transcript_url = f.get("transcript_url")
        if not transcript_url:
            print(f"      [warn] no transcript_url for {fname} (status={f.get('status')})")
            continue

        try:
            dl = requests.get(transcript_url, timeout=60)
            dl.raise_for_status()
            transcript = dl.json().get("full_transcript", "")
            results[clip_id] = transcript
        except Exception as e:
            print(f"      [warn] transcript download failed for {fname}: {e}")

    return results


def run_batch_for_group(clip_batch: list[dict], ckpt: dict) -> dict:
    """
    Submit one batch job for a group of clips, poll until done, collect transcripts.
    Updates ckpt in-place and saves checkpoint after completion.
    Returns {clip_id: transcript}.
    """
    job_id = _create_batch_job(clip_batch)
    if not job_id:
        return {}

    if not _start_batch_job(job_id):
        return {}

    final_status = _poll_until_done(job_id)
    print(f"    [batch] job {job_id} finished with status: {final_status}")

    if final_status in ("FAILED", "START_FAILED", "CANCELLED"):
        print(f"    [ERROR] job {job_id} did not complete — no transcripts.")
        return {}

    transcripts = _collect_transcripts_from_job(job_id, clip_batch)
    ckpt.update(transcripts)
    save_checkpoint(ckpt)
    print(f"    [batch] {len(transcripts)}/{len(clip_batch)} clips transcribed. Checkpoint saved.")
    return transcripts


# ---------------------------------------------------------------------------
# 7.  GNANI REST API (fallback for single clips or when batch unavailable)
# ---------------------------------------------------------------------------

def transcribe_rest(audio_bytes: bytes, lang_code: str) -> str:
    """
    Call the Gnani REST STT endpoint for a single clip.
    Raises RuntimeError after 2 retries.
    """
    MAX_RETRIES = 2
    last_err = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            r = requests.post(
                REST_URL,
                headers=_gnani_headers(),
                data={"language_code": lang_code, "format": "verbatim"},
                files={"audio_file": ("clip.wav", audio_bytes, "audio/wav")},
                timeout=60,
            )
            r.raise_for_status()
            return r.json().get("transcript", "").strip()
        except Exception as exc:
            last_err = exc
            if attempt < MAX_RETRIES:
                wait = 2 ** attempt
                print(f"      [retry {attempt+1}/{MAX_RETRIES}] {exc}. Waiting {wait}s...")
                time.sleep(wait)
    raise RuntimeError(f"REST failed after {MAX_RETRIES+1} attempts: {last_err}")


# ---------------------------------------------------------------------------
# 8.  TRANSCRIPTION ORCHESTRATION
# ---------------------------------------------------------------------------

def transcribe_all(records: list[dict], ckpt: dict) -> dict:
    """
    Transcribe all scorable clips using batch API (grouped by language).
    Skips clips already in ckpt.
    Returns updated ckpt {clip_id: transcript}.
    """
    # Separate scorable clips (have audio + reference)
    to_transcribe = []
    for rec in records:
        clip_id = rec["clip_id"]
        if clip_id in ckpt:
            continue  # already done
        if rec["reference_text"] is None:
            continue  # no ref — will be skipped in scoring anyway
        # resolve audio bytes
        ab = rec["audio_bytes"]
        if rec["uses_url_dl"] and isinstance(ab, str):
            print(f"  Downloading audio for {clip_id} ...")
            ab = _download_url(ab)
            rec["audio_bytes"] = ab  # cache in-place
        if not ab:
            continue  # no audio
        to_transcribe.append(rec)

    if not to_transcribe:
        print("  All clips already in checkpoint or have no audio — nothing to transcribe.")
        return ckpt

    print(f"\n  {len(to_transcribe)} clips to transcribe ({len(ckpt)} already in checkpoint).")

    # Group by language code, then chunk to BATCH_MAX_FILES
    from collections import defaultdict
    by_lang = defaultdict(list)
    for rec in to_transcribe:
        by_lang[rec["lang_code"]].append(rec)

    for lang, group in by_lang.items():
        print(f"\n  Language '{lang}': {len(group)} clips")
        # Chunk into batches of BATCH_MAX_FILES
        for start in range(0, len(group), BATCH_MAX_FILES):
            chunk = group[start:start + BATCH_MAX_FILES]
            # Build the minimal dict the batch function needs
            batch_items = [
                {"clip_id": r["clip_id"], "audio_bytes": r["audio_bytes"], "lang_code": lang}
                for r in chunk
            ]
            run_batch_for_group(batch_items, ckpt)

    return ckpt


# ---------------------------------------------------------------------------
# 9.  SCORING
# ---------------------------------------------------------------------------

def score_records(records: list[dict], ckpt: dict) -> list[dict]:
    """Combine loaded records with transcriptions from ckpt and compute metrics."""
    results = []

    def _skip(rec, error):
        results.append({**rec, "hypothesis_text": None, "ref_norm": None,
                         "hyp_norm": None, "wer": None, "cer": None,
                         "accuracy": None, "w_subs": None, "w_dels": None,
                         "w_ins": None, "c_subs": None, "c_dels": None,
                         "c_ins": None, "ref_word_count": None, "error": error})

    for rec in records:
        clip_id = rec["clip_id"]
        ref_raw = rec["reference_text"]

        if ref_raw is None:
            _skip(rec, "no_reference")
            continue

        hyp_raw = ckpt.get(clip_id)
        if hyp_raw is None:
            _skip(rec, "no_transcript")
            continue

        ref_norm = normalize(ref_raw)
        hyp_norm = normalize(hyp_raw)
        metrics  = edit_distance_metrics(ref_norm, hyp_norm)

        results.append({
            **rec,
            "hypothesis_text": hyp_raw,
            "ref_norm": ref_norm, "hyp_norm": hyp_norm,
            **metrics, "error": None,
        })

    scored = sum(1 for r in results if r["wer"] is not None)
    print(f"\n  Scored {scored}/{len(records)} clips.")
    return results


# ---------------------------------------------------------------------------
# 10. AGGREGATION
# ---------------------------------------------------------------------------

def _agg_group(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return {"n": 0, "accuracy": None, "wer": None, "cer": None,
                "w_subs_rate": None, "w_dels_rate": None, "w_ins_rate": None}

    def _rate(key):
        vals = [(r.get(key) or 0) / r["ref_word_count"]
                for r in rows if (r.get("ref_word_count") or 0) > 0]
        return sum(vals) / len(vals) if vals else None

    return {
        "n":           n,
        "accuracy":    sum(r["accuracy"] for r in rows) / n,
        "wer":         sum(r["wer"]      for r in rows) / n,
        "cer":         sum(r["cer"]      for r in rows) / n,
        "w_subs_rate": _rate("w_subs"),
        "w_dels_rate": _rate("w_dels"),
        "w_ins_rate":  _rate("w_ins"),
    }


def aggregate(results: list[dict]) -> dict:
    scored = [r for r in results if r.get("wer") is not None]
    cats   = sorted(set(r["category"] for r in results))

    by_cat  = {}
    by_sub  = {}
    worst   = {}

    for cat in cats:
        cs = [r for r in scored if r["category"] == cat]
        g  = _agg_group(cs)
        g["total_clips"] = sum(1 for r in results if r["category"] == cat)
        g["skipped"]     = g["total_clips"] - g["n"]
        by_cat[cat] = g

        worst[cat] = sorted(cs, key=lambda r: r["wer"], reverse=True)[:3]

        if any(r.get("subcategory") for r in cs):
            subs = sorted(set(r.get("subcategory") or "unknown" for r in cs))
            by_sub[cat] = {
                sc: _agg_group([r for r in cs if (r.get("subcategory") or "unknown") == sc])
                for sc in subs
            }

    return {"by_category": by_cat, "by_subcategory": by_sub, "worst_clips": worst}


# ---------------------------------------------------------------------------
# 11. REPORT WRITING
# ---------------------------------------------------------------------------

def _p(v, d=1):
    return f"{v*100:.{d}f}%" if v is not None else "--"


def write_report(agg: dict, path: Path) -> None:
    L = []
    L.append("# Gnani Prisma ASR Evaluation Report\n")
    L.append(f"_Generated: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}_\n")

    L.append("\n---\n\n## Summary — Results by Category\n")
    L.append("| Category | n | Accuracy % | WER % | CER % | Subs% | Dels% | Ins% | Notes |\n")
    L.append("|---|---|---|---|---|---|---|---|---|\n")

    for cat, g in sorted(agg["by_category"].items()):
        flag = " *" if g["n"] < LOW_SAMPLE_THRESHOLD else ""
        note = f"{g['skipped']} skipped" if g.get("skipped") else ""
        L.append(
            f"| {cat}{flag} | {g['n']} | {_p(g['accuracy'])} | {_p(g['wer'])} | {_p(g['cer'])} "
            f"| {_p(g['w_subs_rate'])} | {_p(g['w_dels_rate'])} | {_p(g['w_ins_rate'])} | {note} |\n"
        )
    L.append("\n_\\* fewer than 10 scored clips — average less reliable._\n")

    if agg["by_subcategory"]:
        L.append("\n---\n\n## Subcategory Breakdowns\n")
        for cat, sub_dict in sorted(agg["by_subcategory"].items()):
            L.append(f"\n### {cat}\n")
            L.append("| Subcategory | n | Accuracy % | WER % | CER % |\n")
            L.append("|---|---|---|---|---|\n")
            for sc, g in sorted(sub_dict.items()):
                flag = " *" if g["n"] < LOW_SAMPLE_THRESHOLD else ""
                L.append(f"| {sc}{flag} | {g['n']} | {_p(g['accuracy'])} | {_p(g['wer'])} | {_p(g['cer'])} |\n")

    L.append("\n---\n\n## Worst 3 Clips per Category\n")
    for cat, clips in sorted(agg["worst_clips"].items()):
        L.append(f"\n### {cat}\n")
        if not clips:
            L.append("_No scored clips._\n")
            continue
        for c in clips:
            L.append(
                f"- **{c['clip_id']}** — WER {_p(c['wer'])}\n"
                f"  - **Ref:** {c['ref_norm'] or '--'}\n"
                f"  - **Hyp:** {c['hyp_norm'] or '--'}\n"
            )

    path.write_text("".join(L), encoding="utf-8")
    print(f"\n[OK] Report: {path}")


def write_csv(results: list[dict], path: Path) -> None:
    fields = [
        "clip_id","category","subcategory","duration_sec",
        "reference_text","hypothesis_text","ref_norm","hyp_norm",
        "wer","cer","accuracy",
        "w_subs","w_dels","w_ins","c_subs","c_dels","c_ins",
        "ref_word_count","error",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in results:
            row = {k: r.get(k) for k in fields}
            for k in ("wer","cer","accuracy"):
                if row[k] is not None:
                    row[k] = f"{row[k]*100:.4f}"
            w.writerow(row)
    print(f"[OK] CSV: {path}")


# ---------------------------------------------------------------------------
# 12. MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  Gnani Prisma ASR Evaluation")
    print("=" * 70)

    if not GNANI_API_KEY:
        print(
            "\n[ERROR] GNANI_API_KEY is not set.\n"
            "  Export it before running:\n"
            "    export GNANI_API_KEY=<your_key>\n"
        )
        sys.exit(1)

    # Step 1 — load
    print("\n-- Step 1: Loading datasets --")
    records = load_records()
    print(f"\nTotal records: {len(records)}")

    # Step 2 — transcribe (batch, with checkpoint resume)
    print("\n-- Step 2: Transcribing (batch API, checkpoint-resume) --")
    ckpt = load_checkpoint()
    if ckpt:
        print(f"  Checkpoint found: {len(ckpt)} clips already transcribed.")
    ckpt = transcribe_all(records, ckpt)

    # Step 3 — score
    print("\n-- Step 3: Scoring --")
    results = score_records(records, ckpt)

    # Step 4 — aggregate
    print("\n-- Step 4: Aggregating --")
    agg = aggregate(results)

    # Step 5 — write outputs
    print("\n-- Step 5: Writing outputs --")
    write_csv(results, CSV_PATH)
    write_report(agg, REPORT_PATH)

    print("\n[DONE] Evaluation complete.\n")


if __name__ == "__main__":
    main()
