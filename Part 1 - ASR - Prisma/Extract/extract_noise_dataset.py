import json
import random
from pathlib import Path
from datasets import Dataset, Audio

def main():
    print("Building Noise dataset from danielrosehill/ASR-WPM-And-Background-Noise-Eval...")
    
    metadata_path = Path("raw_noise_dataset/metadata.jsonl")
    audio_dir = Path("raw_noise_dataset/audio")
    samples_dir = Path("/tmp/Whisper-WPM-Eval/samples")
    
    if not metadata_path.exists():
        print(f"Error: {metadata_path} not found. Did you clone the repo?")
        return
        
    records = []
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            
            clip_id = item.get("id")
            sample_file = item.get("sample_file")
            
            # audio file
            audio_path = audio_dir / f"{clip_id}.wav"
            
            # text file
            text_path = samples_dir / sample_file
            
            if not audio_path.exists():
                print(f"Warning: Audio file {audio_path} missing.")
                continue
                
            # If it's just an LFS pointer, it will be very small (< 1KB)
            if audio_path.stat().st_size < 1000:
                print(f"Downloading real audio for {clip_id}.wav...")
                import requests
                url = f"https://huggingface.co/datasets/danielrosehill/ASR-WPM-And-Background-Noise-Eval/resolve/main/audio/{clip_id}.wav"
                r = requests.get(url)
                r.raise_for_status()
                audio_path.write_bytes(r.content)
                
            if not text_path.exists():
                print(f"Warning: Text file {text_path} missing.")
                continue
                
            with open(text_path, 'r', encoding='utf-8') as tf:
                text = tf.read().strip()
                
            # Extract noise type
            noise_type = "none"
            annots = item.get("annotations")
            if annots and isinstance(annots, dict):
                noise_type = annots.get("background_noise", "none")
                
            records.append({
                "sample_id": clip_id,
                "text": text,
                "audio": str(audio_path.absolute()),
                "noise_type": noise_type,
            })
            
    # Shuffle and limit to 20 clips
    random.seed(42)
    random.shuffle(records)
    records = records[:10]
            
    print(f"Successfully processed {len(records)} records.")
    
    hf_dict = {
        "sample_id": [r["sample_id"] for r in records],
        "text": [r["text"] for r in records],
        "audio": [r["audio"] for r in records],
        "noise_type": [r["noise_type"] for r in records],
    }
    
    dataset = Dataset.from_dict(hf_dict)
    
    # Cast audio to Hugging Face Audio feature (decode=False to store bytes directly)
    dataset = dataset.cast_column("audio", Audio(decode=False))
    
    output_dir = "Noise_English_dataset"
    print(f"Saving dataset to {output_dir}...")
    dataset.save_to_disk(output_dir)
    
    print("Done!")

if __name__ == "__main__":
    main()
