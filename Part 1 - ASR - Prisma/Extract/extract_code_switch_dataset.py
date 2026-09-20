import os
from dotenv import load_dotenv
from datasets import load_dataset, Dataset, Audio

def main():
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")

    print("Loading soketlabs/CoSHE-Eval (Code-Switching dataset) in streaming mode...")
    # Use streaming=True so we only download what we need instead of the full ~9GB dataset
    dataset = load_dataset("soketlabs/CoSHE-Eval", split="eval", streaming=True, token=hf_token)

    # Collect only 10 records from the stream
    total = 10
    print(f"Collecting first {total} records from stream...")
    rows = []
    for i, row in enumerate(dataset):
        # Keep the audio bytes for evaluation
        rows.append(row)
        if len(rows) >= total:
            break

    print(f"Collected {len(rows)} records. Converting to Dataset...")
    code_switch_ds = Dataset.from_list(rows)
    
    # Store the raw audio bytes locally
    code_switch_ds = code_switch_ds.cast_column("audio", Audio(decode=False))

    # Save to disk with a proper folder name
    output_dir = "./Code_Switch_dataset"
    print(f"Saving {len(code_switch_ds)} records to {output_dir}...")
    code_switch_ds.save_to_disk(output_dir)

    print("\nSuccess!")
    print(f"Saved {len(code_switch_ds)} records to {output_dir}")
    print(f"Columns: {code_switch_ds.column_names}")

if __name__ == "__main__":
    main()
