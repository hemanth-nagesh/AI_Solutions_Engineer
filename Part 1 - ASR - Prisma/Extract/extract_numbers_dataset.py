import os
import re
import random
from collections import defaultdict
from dotenv import load_dotenv

from datasets import (
    load_dataset,
    Dataset,
    Audio,
)

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_NAME = "hf-audio/open-asr-leaderboard"
SUBSET = "librispeech"
SOURCE_SPLIT = "test.clean"  # librispeech uses test.clean and test.other

OUTPUT_DIR = "numbers_dataset"

TOTAL_CLIPS = 25

# We try to make the dataset balanced.
TARGET_PER_CATEGORY = 5

RANDOM_SEED = 42

# ============================================================
# NUMBER DETECTION
# ============================================================

NUMBER_WORDS = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", 
    "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty", 
    "sixty", "seventy", "eighty", "ninety", "hundred", "thousand", "million", "billion",
]

NUMBER_WORD_PATTERN = r"\b(?:" + "|".join(NUMBER_WORDS) + r")\b"
DIGIT_PATTERN = r"\b\d+(?:[.,:/-]\d+)*\b"

ORDINAL_PATTERN = r"""
\b(?:
    first|second|third|fourth|fifth|sixth|seventh|eighth|
    ninth|tenth|eleventh|twelfth|thirteenth|fourteenth|
    fifteenth|sixteenth|seventeenth|eighteenth|nineteenth|
    twentieth|thirtieth|fortieth|fiftieth|sixtieth|
    seventieth|eightieth|ninetieth
)\b
"""

MONTH_PATTERN = r"""
\b(?:
    january|february|march|april|may|june|july|august|
    september|october|november|december
)\b
"""

CURRENCY_PATTERN = r"(?:\$|€|£|₹|rs\.?|rupees?|dollars?|euros?|pounds?)"
TIME_PATTERN = r"\b\d{1,2}(?::\d{2})?\s?(?:am|pm)\b"
DECIMAL_PATTERN = r"\b\d+\.\d+\b"
PERCENTAGE_PATTERN = r"\b\d+(?:\.\d+)?\s?(?:%|percent|percentage)\b"
PHONE_PATTERN = r"\b(?:\+?\d[\d\s\-()]{7,}\d)\b"

# ============================================================
# CLASSIFICATION
# ============================================================

def classify_number_type(text: str) -> str | None:
    text_lower = text.lower()
    
    if re.search(PHONE_PATTERN, text_lower, flags=re.VERBOSE): return "phone_number"
    if re.search(TIME_PATTERN, text_lower, flags=re.VERBOSE): return "time"
    if re.search(CURRENCY_PATTERN, text_lower, flags=re.VERBOSE): return "currency"
    if re.search(PERCENTAGE_PATTERN, text_lower, flags=re.VERBOSE): return "percentage"
    if re.search(DECIMAL_PATTERN, text_lower, flags=re.VERBOSE): return "decimal"
    if re.search(MONTH_PATTERN, text_lower, flags=re.VERBOSE): return "date"
    if re.search(ORDINAL_PATTERN, text_lower, flags=re.VERBOSE): return "ordinal"
    if re.search(DIGIT_PATTERN, text_lower): return "cardinal"
    if re.search(NUMBER_WORD_PATTERN, text_lower): return "cardinal"
    return None

# ============================================================
# LOAD DATASET
# ============================================================

def load_librispeech_dataset():
    print("=" * 70)
    print("Loading Librispeech from open-asr-leaderboard...")
    print("=" * 70)

    hf_token = os.getenv("HF_TOKEN")
    
    dataset = load_dataset(
        DATASET_NAME,
        SUBSET,
        split=SOURCE_SPLIT,
        token=hf_token
    )

    # Disable decoding for speed and to avoid codec errors
    dataset = dataset.cast_column("audio", Audio(decode=False))
    return dataset

# ============================================================
# SELECT 25 CLIPS
# ============================================================

def select_number_clips(dataset):
    print("\nSearching for number-related clips...\n")
    random.seed(RANDOM_SEED)

    candidates = defaultdict(list)
    inspected = 0

    for row in dataset:
        inspected += 1
        
        # open-asr-leaderboard uses "text" column for transcripts
        sentence = row.get("text")
        
        if not sentence:
            continue

        category = classify_number_type(sentence)
        if category is None:
            continue

        candidates[category].append(row)

        enough = all(
            len(candidates[c]) >= TARGET_PER_CATEGORY
            for c in ["cardinal", "date", "time", "currency", "decimal"]
        )

        if enough:
            break
        if inspected >= 500_000:
            break

    print(f"Rows inspected: {inspected}")
    print("\nCandidates found:")
    for category, rows in candidates.items():
        print(f"  {category:15s}: {len(rows)}")

    selected = []
    preferred_categories = ["cardinal", "date", "time", "currency", "decimal"]

    for category in preferred_categories:
        rows = candidates.get(category, [])
        if not rows: continue
        random.shuffle(rows)
        selected.extend(rows[:TARGET_PER_CATEGORY])

    if len(selected) < TOTAL_CLIPS:
        remaining = []
        for category, rows in candidates.items():
            if category in preferred_categories: continue
            remaining.extend(rows)
        random.shuffle(remaining)
        selected.extend(remaining[:TOTAL_CLIPS - len(selected)])

    if len(selected) < TOTAL_CLIPS:
        print(f"\nWarning: only found {len(selected)} suitable clips instead of {TOTAL_CLIPS}.")

    selected = selected[:TOTAL_CLIPS]
    return selected

# ============================================================
# CREATE FINAL DATASET
# ============================================================

def build_final_dataset(rows):
    records = []
    for idx, row in enumerate(rows, start=1):
        sentence = row.get("text", "")
        category = classify_number_type(sentence)
        audio = row.get("audio", {})

        record = {
            "clip_id": row.get("id", f"number_{idx:03d}"),
            "audio": audio,
            "transcript": sentence,
            "number_type": category,
            "audio_length_s": row.get("audio_length_s"),
        }
        records.append(record)

    return Dataset.from_list(records)

# ============================================================
# SAVE DATASET
# ============================================================

def save_dataset(dataset):
    print("\n" + "=" * 70)
    print("Saving Arrow dataset...")
    print("=" * 70)

    dataset.save_to_disk(OUTPUT_DIR, num_shards=1)
    print("\nDataset saved to:")
    print(os.path.abspath(OUTPUT_DIR))

# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(dataset):
    print("\n" + "=" * 70)
    print("FINAL DATASET SUMMARY")
    print("=" * 70)

    print(f"Number of clips: {len(dataset)}")
    print("\nNumber categories:")
    category_counts = defaultdict(int)
    for category in dataset["number_type"]:
        category_counts[category] += 1
    for category, count in category_counts.items():
        print(f"  {category:15s}: {count}")

    print("\nSample records:\n")
    for i in range(min(5, len(dataset))):
        print(f"Clip ID : {dataset[i]['clip_id']}")
        print(f"Type    : {dataset[i]['number_type']}")
        print(f"Text    : {dataset[i]['transcript']}")
        print("-" * 70)

# ============================================================
# MAIN
# ============================================================

def main():
    load_dotenv()
    dataset = load_librispeech_dataset()
    selected_rows = select_number_clips(dataset)
    final_dataset = build_final_dataset(selected_rows)
    print_summary(final_dataset)
    save_dataset(final_dataset)
    print("\nDone!")

if __name__ == "__main__":
    main()
