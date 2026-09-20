import csv
import io
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import requests
from datasets import load_from_disk, Audio
from dotenv import load_dotenv

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

    def tqdm(iterable=None, total=None, desc=None, unit=None, **kwargs):
        """Minimal drop-in fallback so the script still runs without tqdm installed."""
        class _Bar:
            def __init__(self):
                self.n = 0
                self.total = total
                self.desc = desc or ""

            def _render(self):
                pct = f"{(self.n / self.total * 100):5.1f}%" if self.total else ""
                print(f"\r  {self.desc} {pct} ({self.n}/{self.total or '?'})", end="", flush=True)

            def update(self, n=1):
                self.n += n
                self._render()

            def set_postfix(self, **kw):
                extra = " ".join(f"{k}={v}" for k, v in kw.items())
                print(f"\r  {self.desc} ({self.n}/{self.total or '?'}) {extra}", end="", flush=True)

            def close(self):
                print()

            def __enter__(self):
                return self

            def __exit__(self, *a):
                self.close()

        bar = _Bar()
        if iterable is not None:
            for item in iterable:
                yield item
                bar.update(1)
            bar.close()
        else:
            return bar

# ---------------------------------------------------------------------------
# 0.  CONFIGURATION
# ---------------------------------------------------------------------------

load_dotenv(override=True)

GNANI_API_KEY = os.getenv("GNANI_API_KEY")
HF_TOKEN       = os.getenv("HF_TOKEN", "")   # for downloading signed HF audio URLs

# Gnani Vachana REST STT endpoint
REST_URL = "https://api.vachana.ai/stt/v3"

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
    # "duration_long":    ("large_audio_dataset",           "text",          "id",           "audio_length_s","audio_url",    False,  True,   None,          "en-IN"),
    "accent_malayalam": ("Malayalam_accent_India_dataset","text",          None,           "duration",      "audio_filepath",True, False,  None,          "en-IN"),
    "accent_bengali":   ("Bengali_accent_India_dataset",  "text",          None,           "duration",      "audio_filepath",True, False,  None,          "en-IN"),
    "domain_medical":   ("Medical_ASR_dataset",           "text",          "file_name",    "duration",      "audio",        False,  False,  None,          "en-IN"),
    "numeric":          ("numbers_dataset",               "transcript",    "clip_id",      "audio_length_s","audio",        False,  False,  "number_type", "en-IN"),
    "code_switch":      ("Code_Switch_dataset",           "transcription", "audio_file_name",None,          "audio",        False,  False,  None,          "en-IN"),
    # "noise":            ("Noise_English_dataset",         "text",          "sample_id",    None,            "audio",        False,  False,  "noise_type",  "en-IN"),
}

REPORT_PATH      = BASE_DIR / "prisma_eval_report.md"
CSV_PATH         = BASE_DIR / "prisma_eval_clipwise.csv"
CHECKPOINT_PATH  = BASE_DIR / "eval_checkpoint.json"   # clip_id -> transcript (saved after every clip)

LOW_SAMPLE_THRESHOLD = 10   # categories with < N clips get asterisk in report
MAX_RETRIES          = 2    # retry attempts per clip on transient API error

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


def ckpt_text(entry) -> "str | None":
    """
    Extract the transcript string from a checkpoint entry.
    Supports both the legacy format (entry is a plain str) and the new
    format (entry is {"text": ..., "latency_sec": ...}) so old checkpoint
    files keep working without needing clips to be re-transcribed.
    """
    if entry is None:
        return None
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        return entry.get("text")
    return None


def ckpt_latency(entry) -> "float | None":
    """Extract the exact per-clip latency (seconds) from a checkpoint entry, if recorded."""
    if isinstance(entry, dict):
        return entry.get("latency_sec")
    return None


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
    """Download audio from a public URL. Passes HF_TOKEN if set (needed for signed HF URLs)."""
    headers = {}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
    try:
        r = requests.get(url, headers=headers, timeout=60)
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
# 6.  GNANI REST STT  (one clip at a time, checkpoint saved after each clip)
# ---------------------------------------------------------------------------

def _gnani_headers() -> dict:
    return {"X-API-Key-ID": GNANI_API_KEY}


REQUEST_TIMEOUT_SEC = 60


def _is_retryable(exc: Exception) -> bool:
    """
    5xx, timeouts, and connection errors are transient -> worth retrying.
    4xx (bad request, auth, etc.) will fail again identically -> retrying wastes
    time/quota and hides the real problem, so fail fast on those instead.
    """
    if isinstance(exc, requests.exceptions.HTTPError) and exc.response is not None:
        return exc.response.status_code >= 500 or exc.response.status_code == 429
    if isinstance(exc, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)):
        return True
    return False


def _transcribe_chunk(audio_bytes: bytes, lang_code: str) -> tuple[str, float]:
    """
    POST one clip to the Gnani REST STT endpoint and return (transcript, latency_sec).
    latency_sec is the exact wall-clock time of the *successful* HTTP call only
    (retry backoff sleeps are excluded so the number reflects real API latency).
    Retries up to MAX_RETRIES times on transient failures (5xx / 429 / network),
    but fails immediately on non-transient errors (e.g. 400/401/403).
    Raises RuntimeError after all attempts are exhausted.
    """
    # Guess format from magic bytes to prevent 400 Bad Request from Vachana API
    header = audio_bytes[:10]
    filename = "clip.wav"
    mime_type = "audio/wav"
    if header.startswith(b"fLaC"):
        filename, mime_type = "clip.flac", "audio/flac"
    elif header.startswith(b"OggS"):
        filename, mime_type = "clip.ogg", "audio/ogg"
    elif header.startswith(b"\xff\xfb") or header.startswith(b"\xff\xf3") or header.startswith(b"ID3"):
        filename, mime_type = "clip.mp3", "audio/mpeg"
    elif header.startswith(b"RIFF"):
        filename, mime_type = "clip.wav", "audio/wav"

    last_err = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            t0 = time.perf_counter()
            r = requests.post(
                REST_URL,
                headers=_gnani_headers(),
                data={"language_code": lang_code, "format": "verbatim"},
                files={"audio_file": (filename, audio_bytes, mime_type)},
                timeout=REQUEST_TIMEOUT_SEC,
            )
            r.raise_for_status()
            latency_sec = time.perf_counter() - t0
            return r.json().get("transcript", "").strip(), latency_sec
        except Exception as exc:
            last_err = exc
            if not _is_retryable(exc):
                raise RuntimeError(f"Non-retryable error: {exc}") from exc
            if attempt < MAX_RETRIES:
                if isinstance(exc, requests.exceptions.HTTPError) and exc.response is not None and exc.response.status_code == 429:
                    wait = 60.0
                else:
                    wait = (2 ** attempt) + random.uniform(0, 0.5)  # 1s, 2s... + jitter
                print(f"      [retry {attempt+1}/{MAX_RETRIES}] {exc}. Waiting {wait:.1f}s...")
                time.sleep(wait)
    raise RuntimeError(f"All {MAX_RETRIES+1} attempts failed: {last_err}")


def transcribe(audio_bytes: bytes, lang_code: str) -> tuple[str, float]:
    import soundfile as sf
    import io
    try:
        data, sr = sf.read(io.BytesIO(audio_bytes))
        total_sec = len(data) / sr
    except Exception:
        return _transcribe_chunk(audio_bytes, lang_code)
        
    MAX_DURATION = 29.0
    if total_sec <= MAX_DURATION:
        return _transcribe_chunk(audio_bytes, lang_code)
        
    transcripts = []
    total_latency = 0.0
    chunk_samples = int(MAX_DURATION * sr)
    
    for i in range(0, len(data), chunk_samples):
        chunk_data = data[i:i+chunk_samples]
        buf = io.BytesIO()
        sf.write(buf, chunk_data, sr, format='WAV')
        chunk_bytes = buf.getvalue()
        
        text, lat = _transcribe_chunk(chunk_bytes, lang_code)
        if text:
            transcripts.append(text)
        total_latency += lat
        
    return " ".join(transcripts), total_latency



# ---------------------------------------------------------------------------
# 7.  TRANSCRIPTION ORCHESTRATION
# ---------------------------------------------------------------------------

def transcribe_all(records: list[dict], ckpt: dict) -> dict:
    """
    Transcribe every scorable clip one-by-one via the REST API, grouped by
    category so each category gets its own progress bar.
    Skips clips already present in the checkpoint.
    Checkpoint is saved to disk immediately after each successful call
    (transcript + exact latency) so that re-running after a crash, or a
    Ctrl-C, only processes the remaining clips.
    Returns the updated checkpoint dict.
    """
    cats_order = list(DATASET_MAP.keys())   # preserve defined category order
    ordered = sorted(records, key=lambda r: cats_order.index(r["category"]))

    need = [r for r in ordered
            if r["clip_id"] not in ckpt and r["reference_text"] is not None]

    # Pre-resolve audio for URL-download categories
    scorable = []
    for rec in need:
        ab = rec["audio_bytes"]
        if rec["uses_url_dl"] and isinstance(ab, str):
            ab = _download_url(ab)
            rec["audio_bytes"] = ab
        if ab:
            scorable.append(rec)
        else:
            print(f"  [skip] {rec['clip_id']} — no audio")

    already_done = sum(1 for r in records if r["clip_id"] in ckpt)
    total_todo   = len(scorable)
    print(f"  {already_done} already done (checkpoint). {total_todo} to transcribe.")

    # Group remaining clips by category, preserving DATASET_MAP order,
    # so we can show one progress bar per category.
    by_cat: dict[str, list[dict]] = {}
    for rec in scorable:
        by_cat.setdefault(rec["category"], []).append(rec)

    try:
        for cat in [c for c in cats_order if c in by_cat]:
            cat_recs = by_cat[cat]
            bar = tqdm(total=len(cat_recs), desc=f"{cat:<18}", unit="clip")
            wers, latencies, fails = [], [], 0

            for rec in cat_recs:
                clip_id, lang_code = rec["clip_id"], rec["lang_code"]
                try:
                    hyp, latency_sec = transcribe(rec["audio_bytes"], lang_code)
                    ckpt[clip_id] = {"text": hyp, "latency_sec": latency_sec}
                    save_checkpoint(ckpt)          # persist after every clip

                    # quick inline score for progress display
                    ref_n = normalize(rec["reference_text"])
                    hyp_n = normalize(hyp)
                    m = edit_distance_metrics(ref_n, hyp_n)
                    wers.append(m["wer"])
                    latencies.append(latency_sec)
                    bar.set_postfix(
                        wer=f"{m['wer']*100:.1f}%",
                        acc=f"{m['accuracy']*100:.1f}%",
                        lat=f"{latency_sec:.2f}s",
                        avg_wer=f"{(sum(wers)/len(wers))*100:.1f}%",
                        avg_lat=f"{(sum(latencies)/len(latencies)):.2f}s",
                    )
                except Exception as exc:
                    fails += 1
                    bar.set_postfix(FAILED=clip_id, err=str(exc)[:40])
                    print(f"\n  [{clip_id}] FAILED: {exc}")
                    # not added to ckpt, will be retried on next run
                finally:
                    bar.update(1)

            bar.close()
            print(f"  -> {cat}: {len(wers)} ok, {fails} failed"
                  + (f", avg WER={sum(wers)/len(wers)*100:.1f}%" if wers else "")
                  + (f", avg latency={sum(latencies)/len(latencies):.2f}s" if latencies else ""))
    except KeyboardInterrupt:
        print("\n[interrupted] Checkpoint already saved up to the last completed clip. "
              "Re-run the script to resume.")
        raise

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
                         "c_ins": None, "ref_word_count": None,
                         "latency_sec": None, "error": error})

    for rec in records:
        clip_id = rec["clip_id"]
        ref_raw = rec["reference_text"]

        if ref_raw is None:
            _skip(rec, "no_reference")
            continue

        entry   = ckpt.get(clip_id)
        hyp_raw = ckpt_text(entry)
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
            **metrics, "latency_sec": ckpt_latency(entry), "error": None,
        })

    scored = sum(1 for r in results if r["wer"] is not None)
    print(f"\n  Scored {scored}/{len(records)} clips.")
    return results


# ---------------------------------------------------------------------------
# 10. AGGREGATION
# ---------------------------------------------------------------------------

def _percentile(vals: list[float], p: float) -> "float | None":
    """Nearest-rank percentile (0<=p<=100) over a sorted copy of vals."""
    if not vals:
        return None
    s = sorted(vals)
    idx = min(len(s) - 1, max(0, int(round(p / 100 * (len(s) - 1)))))
    return s[idx]


def _latency_stats(rows: list[dict]) -> dict:
    lats = [r["latency_sec"] for r in rows if r.get("latency_sec") is not None]
    if not lats:
        return {"lat_avg": None, "lat_min": None, "lat_max": None,
                "lat_p50": None, "lat_p95": None, "lat_n": 0}
    return {
        "lat_avg": sum(lats) / len(lats),
        "lat_min": min(lats),
        "lat_max": max(lats),
        "lat_p50": _percentile(lats, 50),
        "lat_p95": _percentile(lats, 95),
        "lat_n":   len(lats),
    }


def _agg_group(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return {"n": 0, "accuracy": None, "wer": None, "cer": None,
                "w_subs_rate": None, "w_dels_rate": None, "w_ins_rate": None,
                **_latency_stats([])}

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
        **_latency_stats(rows),
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


def _s(v, d=2):
    return f"{v:.{d}f}s" if v is not None else "--"


def write_report(agg: dict, path: Path) -> None:
    L = []
    L.append("# Gnani Prisma ASR Evaluation Report\n")
    L.append(f"_Generated: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}_\n")

    L.append("\n---\n\n## Summary — Results by Category\n")
    L.append("| Category | n | Accuracy % | WER % | CER % | Subs% | Dels% | Ins% | "
              "Avg Latency | P50 Latency | P95 Latency | Min/Max Latency | Notes |\n")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")

    for cat, g in sorted(agg["by_category"].items()):
        flag = " *" if g["n"] < LOW_SAMPLE_THRESHOLD else ""
        note = f"{g['skipped']} skipped" if g.get("skipped") else ""
        minmax = f"{_s(g['lat_min'])} / {_s(g['lat_max'])}" if g.get("lat_n") else "--"
        L.append(
            f"| {cat}{flag} | {g['n']} | {_p(g['accuracy'])} | {_p(g['wer'])} | {_p(g['cer'])} "
            f"| {_p(g['w_subs_rate'])} | {_p(g['w_dels_rate'])} | {_p(g['w_ins_rate'])} "
            f"| {_s(g['lat_avg'])} | {_s(g['lat_p50'])} | {_s(g['lat_p95'])} | {minmax} | {note} |\n"
        )
    L.append("\n_\\* fewer than 10 scored clips — average less reliable._\n")
    L.append("_Latency = exact wall-clock time of each clip's successful API call "
              "(retry backoff excluded); averaged per category above._\n")

    if agg["by_subcategory"]:
        L.append("\n---\n\n## Subcategory Breakdowns\n")
        for cat, sub_dict in sorted(agg["by_subcategory"].items()):
            L.append(f"\n### {cat}\n")
            L.append("| Subcategory | n | Accuracy % | WER % | CER % | Avg Latency |\n")
            L.append("|---|---|---|---|---|---|\n")
            for sc, g in sorted(sub_dict.items()):
                flag = " *" if g["n"] < LOW_SAMPLE_THRESHOLD else ""
                L.append(f"| {sc}{flag} | {g['n']} | {_p(g['accuracy'])} | {_p(g['wer'])} | {_p(g['cer'])} "
                         f"| {_s(g['lat_avg'])} |\n")

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
        "ref_word_count","latency_sec","error",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in results:
            row = {k: r.get(k) for k in fields}
            for k in ("wer","cer","accuracy"):
                if row[k] is not None:
                    row[k] = f"{row[k]*100:.4f}"
            if row["latency_sec"] is not None:
                row["latency_sec"] = f"{row['latency_sec']:.4f}"
            w.writerow(row)
    print(f"[OK] CSV: {path}")


# ---------------------------------------------------------------------------
# 12. MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  Gnani Prisma ASR Evaluation")
    print("=" * 70)

    # if not GNANI_API_KEY:
    #     print(
    #         "\n[ERROR] GNANI_API_KEY is not set.\n"
    #         "  Either export it:\n"
    #         "    export GNANI_API_KEY=<your_key>\n"
    #         "  or add it to the .env file in this directory.\n"
    #     )
    #     sys.exit(1)

    run_start = time.perf_counter()

    # Step 1 — load
    print("\n-- Step 1: Loading datasets --")
    records = load_records()
    print(f"\nTotal records: {len(records)}")

    # Step 2 — transcribe (REST, one clip at a time, checkpoint-resume)
    print("\n-- Step 2: Transcribing (REST API, checkpoint-resume) --")
    ckpt = load_checkpoint()
    try:
        ckpt = transcribe_all(records, ckpt)
    except KeyboardInterrupt:
        print(f"\n[STOPPED] Interrupted by user after {time.perf_counter()-run_start:.1f}s. "
              f"Checkpoint saved — re-run to resume from where you left off.")
        sys.exit(130)

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

    elapsed = time.perf_counter() - run_start
    print(f"\n[DONE] Evaluation complete in {elapsed:.1f}s.\n")


if __name__ == "__main__":
    main()