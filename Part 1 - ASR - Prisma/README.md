# Audio Datasets Workspace

This directory contains various subsets of audio data extracted from Hugging Face for ASR (Automatic Speech Recognition) and TTS (Text-to-Speech) experiments. It also contains the Python scripts used to generate and filter them.

## 📂 Dataset Folders

All datasets below are saved in the Hugging Face native `Arrow` format. You can easily load any of them back into Python using `datasets.load_from_disk("folder_name")`.

### 1. Duration-Based Filtering (Common Voice)
- **`small_audio_dataset/`**: Contains 10 audio records where the audio length is strictly **less than 5.0 seconds**. Extracted from the `common_voice` subset of the Open ASR Leaderboard.
- **`large_audio_dataset/`**: Contains 10 audio records where the audio length is strictly **greater than 15.0 seconds**. Extracted from the `common_voice` subset of the Open ASR Leaderboard.

### 2. Language-Specific Accents (Svarah)
- **`Malayalam_accent_India_dataset/`**: Contains exactly 20 records where the primary spoken language is Malayalam. Extracted from the `ai4bharat/Svarah` dataset.
- **`Bengali_accent_India_dataset/`**: Contains exactly 20 records where the primary spoken language is Bengali. Extracted from the `ai4bharat/Svarah` dataset.

### 3. Medical Domain (Eka Care)
- **`Medical_ASR_dataset/`**: Contains the first 25 evaluation records extracted from `ekacare/eka-medical-asr-evaluation-dataset`. Intended for ASR testing.
- **`Medical_TTS_dataset/`**: Contains the next 25 evaluation records (rows 25-50) from the same medical dataset. Intended for TTS testing.

### 4. Categorical / Numerical Evaluation
- **`numbers_dataset/`**: A balanced evaluation dataset of 25 records containing specific numerical usages (cardinals, currencies, dates, and ordinals). Extracted from the `librispeech` split of the Open ASR Leaderboard.

### 5. Code-Switching (CoSHE-Eval)
- **`Code_Switch_dataset/`**: Contains 10 records from the `soketlabs/CoSHE-Eval` dataset, which consists of speech samples where speakers naturally switch between two or more languages within a single utterance. Useful for evaluating code-switch ASR models.

### 6. Noisy Speech
- **`Noise_English_dataset/`**: Contains 10 English records sampled uniformly across 8 different environmental noise types. Processed locally from the raw Kaggle dataset.

### 7. Clean Speech
- **`clean_audio_dataset/`**: Contains 25 clean speech records extracted from the `ami_cleaned` subset of the Open ASR Leaderboard.

---

## 📦 What files are inside each dataset folder?

Since these datasets are saved using the Hugging Face `datasets` library (`save_to_disk`), each dataset folder contains the following standard file structure:

- **`dataset_info.json`**: Contains the metadata about the dataset (feature types, column names, split size, etc.).
- **`state.json`**: A state tracking file used internally by Hugging Face.
- **`dataset.arrow`** (or similar `.arrow` files): The actual data stored in the highly optimized Apache Arrow binary format. This single file contains the raw audio bytes, transcripts, and all other row information.

---

## 📜 Scripts Included

- **`extract_audio_datasets.py`**: Generates the `small` and `large` duration datasets.
- **`count_languages.py`**: Reads the `Svarah` dataset to quickly count total occurrences of regional languages (without decoding the audio).
- **`indian_accents.py`**: Filters and generates the 20-record Malayalam and Bengali datasets.
- **`extract_medical_datasets.py`**: Slices the Eka Care medical dataset into ASR and TTS datasets.
- **`extract_numbers_dataset.py`**: Contains advanced regex filtering logic to scrape and balance number-specific sentences into a 25-record evaluation dataset.
- **`extract_code_switch_dataset.py`**: Loads the `soketlabs/CoSHE-Eval` code-switching dataset and saves 10 records to `Code_Switch_dataset/`.
- **`extract_noise_dataset.py`**: Parses the local `raw_noice` CSV and `.wav` audio directory to compile 10 diverse English records into the `Noise_English_dataset/`.
- **`extract_clean_dataset.py`**: Loads the `ami_cleaned` subset from the Open ASR Leaderboard and saves 25 records to `clean_audio_dataset/`.

---

## 🚀 Setup & Usage

To interact with these scripts and datasets, make sure your virtual environment is active:
```bash
source venv/bin/activate
```

If you haven't already, install the necessary dependencies:
```bash
pip install -r requirements.txt
```

*(Note: Ensure you have a `.env` file containing your `HF_TOKEN` in this directory so the scripts can securely authenticate against gated datasets like Svarah).*

---

## 🌳 Project Directory Tree

Below is the complete overview of the workspace structure, highlighting all the generated datasets and extraction scripts:

```text
📁 Gnani_ai/
├── 📁 Bengali_accent_India_dataset/
│   ├── 📄 data-00000-of-00001.arrow
│   ├── 📄 dataset_info.json
│   └── 📄 state.json
├── 📁 Code_Switch_dataset/
│   ├── 📄 data-00000-of-00001.arrow
│   ├── 📄 dataset_info.json
│   └── 📄 state.json
├── 📁 Malayalam_accent_India_dataset/
│   ├── 📄 data-00000-of-00001.arrow
│   ├── 📄 dataset_info.json
│   └── 📄 state.json
├── 📁 Medical_ASR_dataset/
│   ├── 📄 data-00000-of-00001.arrow
│   ├── 📄 dataset_info.json
│   └── 📄 state.json
├── 📁 Medical_TTS_dataset/
│   ├── 📄 data-00000-of-00001.arrow
│   ├── 📄 dataset_info.json
│   └── 📄 state.json
├── 📁 Noise_English_dataset/
│   ├── 📄 data-00000-of-00001.arrow
│   ├── 📄 dataset_info.json
│   └── 📄 state.json
├── 📄 README.md
├── 📁 clean_audio_dataset/
│   ├── 📄 data-00000-of-00001.arrow
│   ├── 📄 dataset_info.json
│   └── 📄 state.json
├── 📄 count_languages.py
├── 📄 extract_audio_datasets.py
├── 📄 extract_clean_dataset.py
├── 📄 extract_code_switch_dataset.py
├── 📄 extract_noise_dataset.py
├── 📄 extract_numbers_dataset.py
├── 📄 indian_accents.py
├── 📁 large_audio_dataset/
│   ├── 📄 data-00000-of-00001.arrow
│   ├── 📄 dataset_info.json
│   └── 📄 state.json
├── 📁 numbers_dataset/
│   ├── 📄 data-00000-of-00001.arrow
│   ├── 📄 dataset_info.json
│   └── 📄 state.json
├── 📁 raw_noice/
│   ├── 📁 Audio/
│   └── 📄 English_Chinese_Interpretation_Audio_Access_Dataset.csv
├── 📄 requirements.txt
└── 📁 small_audio_dataset/
    ├── 📄 data-00000-of-00001.arrow
    ├── 📄 dataset_info.json
    └── 📄 state.json
```
