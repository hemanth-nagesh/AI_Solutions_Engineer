import os
from dotenv import load_dotenv
from datasets import load_dataset, Audio

def main():
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")

    print("Loading 'ami_cleaned' dataset from open-asr-leaderboard...")
    dataset = load_dataset("hf-audio/open-asr-leaderboard", "ami_cleaned", split="test", token=hf_token)
    
    # Disable audio decoding to speed up processing and prevent codec issues
    dataset = dataset.cast_column("audio", Audio(decode=False))

    # Limit to 25 rows
    total = min(25, len(dataset))
    print(f"Selecting first {total} records...")
    clean_ds = dataset.select(range(total))

    # Save locally
    output_dir = "./clean_audio_dataset"
    print(f"Saving datasets to {output_dir}...")
    clean_ds.save_to_disk(output_dir)

    print(f"\nSuccess!")
    print(f"Saved {len(clean_ds)} records to {output_dir}")

if __name__ == "__main__":
    main()
