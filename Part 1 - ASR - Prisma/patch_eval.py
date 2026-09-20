import sys

with open('evaluate_prisma.py', 'r') as f:
    content = f.read()

batch_code = """
BATCH_JOBS_URL = "https://api.vachana.ai/stt/v3/batch/jobs"
TERMINAL_STATUSES = {"COMPLETED", "PARTIAL_FAILURE", "FAILED", "START_FAILED", "CANCELLED"}
POLL_INTERVAL_S = 15

def _create_batch_job(clip_batch: list[dict]) -> str | None:
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
        print(f"\\n    [batch] job created: {job_id}  ({len(clip_batch)} files, lang={lang})")
        return job_id
    except Exception as e:
        print(f"\\n    [ERROR] create batch job failed: {e}")
        return None

def _start_batch_job(job_id: str) -> bool:
    try:
        r = requests.post(f"{BATCH_JOBS_URL}/{job_id}/start", headers=_gnani_headers(), timeout=30)
        r.raise_for_status()
        print(f"    [batch] job {job_id} started.")
        return True
    except Exception as e:
        print(f"    [ERROR] start job {job_id} failed: {e}")
        return False

def _poll_until_done(job_id: str) -> str:
    while True:
        try:
            r = requests.get(f"{BATCH_JOBS_URL}/{job_id}", headers=_gnani_headers(), timeout=30)
            r.raise_for_status()
            status = r.json().get("status", "UNKNOWN")
            print(f"    [batch] {job_id} status: {status}")
            if status in TERMINAL_STATUSES:
                return status
        except Exception as e:
            print(f"    [WARN] poll error for {job_id}: {e}")
        time.sleep(POLL_INTERVAL_S)

def _collect_transcripts_from_job(job_id: str, clip_batch: list[dict]) -> dict:
    results = {}
    from pathlib import Path
    fname_to_id = {f"{item['clip_id']}.wav": item["clip_id"] for item in clip_batch}
    try:
        r = requests.get(f"{BATCH_JOBS_URL}/{job_id}/files", headers=_gnani_headers(), timeout=30)
        r.raise_for_status()
        files_data = r.json()
    except Exception as e:
        print(f"    [ERROR] get files for {job_id}: {e}")
        return results
    file_list = files_data if isinstance(files_data, list) else files_data.get("files", [])
    for f in file_list:
        fname = Path(f.get("original_path", "")).name
        clip_id = fname_to_id.get(fname)
        if not clip_id: continue
        t_url = f.get("transcript_url")
        if not t_url: continue
        try:
            dl = requests.get(t_url, timeout=60)
            dl.raise_for_status()
            results[clip_id] = {"text": dl.json().get("full_transcript", "").strip(), "latency_sec": 0.0}
        except Exception as e:
            print(f"    [WARN] transcript download failed for {fname}: {e}")
    return results

def run_batch_for_group(clip_batch: list[dict], ckpt: dict) -> dict:
    job_id = _create_batch_job(clip_batch)
    if not job_id: return {}
    if not _start_batch_job(job_id): return {}
    final_status = _poll_until_done(job_id)
    if final_status in ("FAILED", "START_FAILED", "CANCELLED"):
        print(f"    [ERROR] job {job_id} did not complete — no transcripts.")
        return {}
    transcripts = _collect_transcripts_from_job(job_id, clip_batch)
    ckpt.update(transcripts)
    save_checkpoint(ckpt)
    print(f"    [batch] {len(transcripts)}/{len(clip_batch)} clips transcribed. Checkpoint saved.")
    return transcripts

# ---------------------------------------------------------------------------
"""

# Insert batch_code right before transcribe_all
marker = "def transcribe_all(records: list[dict], ckpt: dict) -> dict:"
if marker not in content:
    print("Could not find transcribe_all")
    sys.exit(1)

content = content.replace(marker, batch_code + "\n" + marker)

# Replace the loop in transcribe_all
old_loop = """        for cat in [c for c in cats_order if c in by_cat]:
            cat_recs = by_cat[cat]
            bar = tqdm(total=len(cat_recs), desc=f"{cat:<18}", unit="clip")"""

new_loop = """        for cat in [c for c in cats_order if c in by_cat]:
            cat_recs = by_cat[cat]
            if cat in ("code_switch", "noise"):
                print(f"\\n  Routing '{cat}' ({len(cat_recs)} clips) to Batch API due to >30s duration...")
                run_batch_for_group(cat_recs, ckpt)
                continue

            bar = tqdm(total=len(cat_recs), desc=f"{cat:<18}", unit="clip")"""

if old_loop not in content:
    print("Could not find old loop")
    sys.exit(1)

content = content.replace(old_loop, new_loop)

with open('evaluate_prisma.py', 'w') as f:
    f.write(content)
print("Patch applied successfully.")
