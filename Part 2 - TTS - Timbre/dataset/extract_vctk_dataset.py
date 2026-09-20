import os
import json
import time
from dotenv import load_dotenv
from datasets import load_dataset, Audio

TECHNICAL_WORDS = [
    "photosynthesis", "electromagnetic", "architecture", "hypothesis",
    "microprocessor", "algorithm", "infrastructure", "parameter",
    "quantum", "thermodynamics", "mechanism", "cognitive",
    "spectrum", "molecule", "chromosome", "velocity",
    "evaluation", "analysis", "significant", "development",
    "international", "organization", "technology", "community"
]

# Safety / progress settings
MAX_ROWS = 50000          # hard cap so the loop can never run forever
PROGRESS_EVERY = 200      # print a progress line every N rows
SHUFFLE_BUFFER = 2000     # shuffle buffer so accents/speakers interleave better


def contains_technical_word(text):
    text_lower = text.lower()
    for word in TECHNICAL_WORDS:
        if word in text_lower:
            return True

    # Fallback: check for any word with 11 or more characters
    words = [w.strip('.,!?;:"()') for w in text_lower.split()]
    if any(len(w) >= 11 for w in words):
        return True

    return False


def save_category(rows, folder_name, category_name, base_dir="./VCTK_dataset"):
    output_dir = os.path.join(base_dir, folder_name)
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "manifest.jsonl")

    print(f"Saving {len(rows)} records to {manifest_path}...")
    with open(manifest_path, "w", encoding="utf-8") as f:
        for row in rows:
            # row.get(key, default) only falls back when the key is MISSING.
            # Several fields here can be present but set to None, so guard
            # with `or` as well to avoid AttributeError on .replace()/etc.
            accent = (row.get("accent") or "").strip()
            gender = row.get("gender") or ""
            age = row.get("age") or ""
            file_id = row.get("file_id") or f"{row['speaker_id']}_{hash(row['text'])}"

            # Add specific accent to category if it's accent diversity
            cat = category_name
            if category_name == "vctk_accent_diversity":
                accent_slug = accent.replace(" ", "_").lower() if accent else "unknown"
                cat = f"vctk_accent_diversity_{accent_slug}"

            record = {
                "clip_id": file_id,
                "category": cat,
                "reference_text": row["text"],
                "accent": accent,
                "speaker_id": row["speaker_id"],
                "gender": gender,
                "age": age
            }
            f.write(json.dumps(record) + "\n")


def main():
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")

    print("Loading saeedzou/vctk-16khz in streaming mode...")
    dataset = load_dataset("saeedzou/vctk-16khz", split="train", streaming=True, token=hf_token)

    # Skip audio decoding entirely -- we never touch row['audio'], and decoding it
    # on every row is the main reason this was slow.
    if "audio" in dataset.features:
        dataset = dataset.cast_column("audio", Audio(decode=False))

    # Shuffle so speakers/accents interleave instead of arriving in long runs,
    # which lets the accent-diversity and technical-vocab quotas fill faster.
    dataset = dataset.shuffle(seed=42, buffer_size=SHUFFLE_BUFFER)

    general_english_rows = []
    accent_diversity_rows = []
    technical_rows = []

    general_speakers = set()
    accent_speakers = set()
    technical_speakers = set()
    accents_collected = set()

    print("Collecting records from stream...")
    start_time = time.time()
    rows_scanned = 0

    for row in dataset:
        rows_scanned += 1

        if rows_scanned % PROGRESS_EVERY == 0:
            elapsed = time.time() - start_time
            rate = rows_scanned / elapsed if elapsed > 0 else 0
            print(
                f"...scanned {rows_scanned} rows in {elapsed:.1f}s "
                f"({rate:.1f} rows/s) | "
                f"general={len(general_english_rows)}/15 "
                f"accent={len(accent_diversity_rows)}/10 "
                f"technical={len(technical_rows)}/10"
            )

        if rows_scanned >= MAX_ROWS:
            print(f"Hit MAX_ROWS={MAX_ROWS} scan limit, stopping early with whatever was collected.")
            break

        speaker_id = row['speaker_id']
        accent = (row.get('accent') or '').strip()
        text = row.get('text') or ''

        if not text:
            # Nothing to match against; skip this row entirely.
            continue

        is_general_english = (accent == 'English')

        # 1. Technical Vocabulary (10 clips)
        if len(technical_rows) < 10 and speaker_id not in technical_speakers:
            if contains_technical_word(text):
                technical_rows.append(row)
                technical_speakers.add(speaker_id)
                continue

        # 2. General English (15 clips)
        if is_general_english:
            if len(general_english_rows) < 15 and speaker_id not in general_speakers:
                if speaker_id not in accent_speakers and speaker_id not in technical_speakers:
                    general_english_rows.append(row)
                    general_speakers.add(speaker_id)
        # 3. Accent Diversity (10 clips)
        else:
            if len(accent_diversity_rows) < 10 and speaker_id not in accent_speakers:
                if speaker_id not in general_speakers and speaker_id not in technical_speakers:
                    if accent not in accents_collected or len(accents_collected) >= 8:
                        accent_diversity_rows.append(row)
                        accent_speakers.add(speaker_id)
                        accents_collected.add(accent)

        if len(general_english_rows) >= 15 and len(accent_diversity_rows) >= 10 and len(technical_rows) >= 10:
            print(f"All quotas met after scanning {rows_scanned} rows.")
            break

    elapsed = time.time() - start_time
    print(f"\nFinished scanning {rows_scanned} rows in {elapsed:.1f}s.")
    print(f"Collected {len(general_english_rows)} General English records.")
    print(f"Collected {len(accent_diversity_rows)} Accent-Diversity records.")
    print(f"Collected {len(technical_rows)} Technical records.")

    save_category(general_english_rows, "General_English", "vctk_general_english")
    save_category(accent_diversity_rows, "Accent_Diversity", "vctk_accent_diversity")
    save_category(technical_rows, "Technical_Vocabulary", "vctk_technical_vocab")

    print("\nSuccess!")


if __name__ == "__main__":
    main()