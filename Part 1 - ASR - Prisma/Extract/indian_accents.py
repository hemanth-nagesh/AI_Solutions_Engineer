import os
from collections import Counter
from dotenv import load_dotenv
from datasets import load_dataset, Audio

def main():
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")

    print("Loading ai4bharat/Svarah dataset (test split)...")
    dataset = load_dataset("ai4bharat/Svarah", split="test", token=hf_token)
    
    # Disable audio decoding to drastically speed up processing
    dataset = dataset.cast_column("audio", Audio(decode=False))

    print("Counting languages...")
    # Quickly get all primary languages and count them
    languages = dataset["primary_language"]
    language_counts = Counter(languages)

    malayalam_count = language_counts.get("Malayalam", 0)
    bengali_count = language_counts.get("Bengali", 0)

    print(f"\n--- Language Counts ---")
    print(f"Malayalam: {malayalam_count}")
    print(f"Bengali:   {bengali_count}")

    # Create the filtered datasets
    print("\nFiltering datasets...")
    malayalam_ds = dataset.filter(lambda lang: lang == "Malayalam", input_columns=["primary_language"])
    bengali_ds = dataset.filter(lambda lang: lang == "Bengali", input_columns=["primary_language"])
    
    # Restrict to only 20 records each
    malayalam_20 = malayalam_ds.select(range(min(20, len(malayalam_ds))))
    bengali_20 = bengali_ds.select(range(min(20, len(bengali_ds))))

    # Save to disk with proper folder names
    print("Saving 20 records each to disk...")
    malayalam_20.save_to_disk("./Malayalam_accent_India_dataset")
    bengali_20.save_to_disk("./Bengali_accent_India_dataset")
    
    print("\nSuccess!")
    print(f"Saved {len(malayalam_20)} records to ./Malayalam_accent_India_dataset")
    print(f"Saved {len(bengali_20)} records to ./Bengali_accent_India_dataset")

if __name__ == "__main__":
    main()
