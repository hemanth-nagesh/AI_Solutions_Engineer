from datasets import load_dataset, Audio

def main():
    print("Loading 'common_voice' dataset...")
    # Load the common_voice subset from the dataset (which automatically downloads the audio)
    dataset = load_dataset("hf-audio/open-asr-leaderboard", "common_voice", split="test")

    # Define thresholds for 'small' and 'large' audio (in seconds)
    SMALL_AUDIO_THRESHOLD = 5.0
    LARGE_AUDIO_THRESHOLD = 15.0

    # Filter for small audio and limit to 10 rows
    print(f"Filtering small audio (less than {SMALL_AUDIO_THRESHOLD} seconds)...")
    small_audio_ds = dataset.filter(lambda x: x["audio_length_s"] < SMALL_AUDIO_THRESHOLD)
    small_audio_10 = small_audio_ds.select(range(min(10, len(small_audio_ds))))

    # Filter for large audio and limit to 10 rows
    print(f"Filtering large audio (greater than {LARGE_AUDIO_THRESHOLD} seconds)...")
    large_audio_ds = dataset.filter(lambda x: x["audio_length_s"] > LARGE_AUDIO_THRESHOLD)
    large_audio_10 = large_audio_ds.select(range(min(10, len(large_audio_ds))))

    # We want to store the raw bytes so they don't rely on HF cached-assets URLs that expire.
    # By casting to Audio(decode=False) *after* the dataset is loaded, it will pack the downloaded
    # bytes into the dataset directory.
    print("Packing raw bytes into datasets...")
    small_audio_10 = small_audio_10.cast_column("audio", Audio(decode=False))
    large_audio_10 = large_audio_10.cast_column("audio", Audio(decode=False))

    # Save the separate datasets locally to the machine
    print("Saving datasets to local disk...")
    small_audio_10.save_to_disk("./small_audio_dataset")
    large_audio_10.save_to_disk("./large_audio_dataset")

    print(f"\nSuccess!")
    print(f"Saved {len(small_audio_10)} records to './small_audio_dataset'")
    print(f"Saved {len(large_audio_10)} records to './large_audio_dataset'")

if __name__ == "__main__":
    main()
