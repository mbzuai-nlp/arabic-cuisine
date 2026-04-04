# Arabic Cuisine

A comprehensive framework for Arabic dialect to MSA translation supporting both zero-shot evaluation and supervised fine-tuning (SFT).

## Setup

### 1. Install Dependencies

```bash
pip install pandas openpyxl openai transformers torch google-generativeai datasets trl peft nltk bert-score
```

### 2. Configure API Keys

Edit the `.env` file with your actual API keys and make it executable:

```bash
# Make .env file executable
chmod +x .env

# Edit .env file with your actual API keys
# OPENAI_API_KEY="sk-your-actual-openai-key-here"
# GOOGLE_API_KEY="your-actual-google-api-key-here"

# Source the environment variables before running evaluations
source .env
```

## Usage

### Zero-Shot Evaluation

#### 1. Process Data

```bash
python scripts/zero-shot/process_data.py
```

Creates JSONL files in `data_translate/` with sequential IDs (0 to n).

#### 2. Evaluate with Models

```bash
# Source environment variables first
source .env

# Then run evaluations
# OpenAI GPT models
python scripts/zero-shot/translate_zeroshot.py --model_type openai --model_name gpt-4 --data_path data_translate/test.jsonl

# Google Gemini models
python scripts/zero-shot/translate_zeroshot.py --model_type gemini --model_name gemini-pro --data_path data_translate/test.jsonl

# HuggingFace models
python scripts/zero-shot/translate_zeroshot.py --model_type huggingface --model_name google/gemma-3-1b-it --data_path data_translate/test.jsonl

# With custom output path and limit for testing
python scripts/zero-shot/translate_zeroshot.py --model_type openai --model_name gpt-4 --data_path data_translate/test.jsonl --output_path output/my_results.jsonl --limit 10
```

### Supervised Fine-Tuning (SFT)

#### 1. Prepare Dataset

```bash
python scripts/sft/prepare_dataset.py
```

Processes the data into the instruction format and creates train/test splits in `processed_dataset/`.

#### 2. Train SFT Model

```bash
python scripts/sft/train.py
```

Fine-tunes a model (default: `google/gemma-3-1b-it`) using LoRA and saves it to `models/sft-msa-dialect-final/`.

#### 3. Test SFT Model

```bash
# Test the trained model
python scripts/sft/test.py

# Test with limited samples
python scripts/sft/test.py --max_samples 50

# Specify custom paths
python scripts/sft/test.py \
    --model_path models/sft-msa-dialect-final \
    --data_path data_translate/test.jsonl \
    --output_dir output/sft
```

#### 4. Single Inference

```bash
python scripts/sft/infer.py
```

Runs inference on a single example using the trained model.

## Project Structure

```
dialect_translation/
├── data/                          # Original Excel data
│   ├── train_dummy.xlsx
│   └── test_dummy.xlsx
├── data_translate/                # Processed JSONL data
│   ├── train.jsonl
│   └── test.jsonl
├── processed_dataset/             # SFT training data (HuggingFace format)
│   ├── train/
│   └── test/
├── models/                        # Trained models
│   └── sft-msa-dialect-final/
├── output/                        # Model outputs
│   ├── zero-shot/                 # Zero-shot results
│   └── sft/                       # SFT results
├── results/                       # Evaluation results
│   ├── zero-shot/
│   └── sft/
└── scripts/
    ├── zero-shot/                 # Zero-shot scripts
    │   ├── process_data.py
    │   └── translate_zeroshot.py
    └── sft/                       # SFT scripts
        ├── prepare_dataset.py
        ├── train.py
        ├── test.py
        └── infer.py
```

## Data Formats

### JSONL Format (Zero-shot)

Each record in `data_translate/*.jsonl` contains:

```json
{
  "id": 0,
  "dialogue_msa": "MSA text",
  "dialogue_dialect": "Dialect text",
  "region": "North Africa",
  "country": "Morocco",
  "worker_id": "Morocco-1",
  "sample_id": "1 | Breakfast",
  "sub_topic": "الافطار"
}
```

### SFT Dataset Format

Processed dataset in `processed_dataset/` uses instruction format:

```json
{
  "instruction": "Translate the following Modern Standard Arabic (MSA) text into the North Africa dialect of Morocco.",
  "input": "MSA text here",
  "output": "Dialect translation here"
}
```

### Output Format

Both zero-shot and SFT results follow the same structure:

```json
{
  "id": 0,
  "region": "North Africa",
  "country": "Morocco",
  "original_msa": "MSA text",
  "original_dialect": "Dialect text",
  "msa_to_dialect_prediction": "Generated dialect translation",
  "dialect_to_msa_prediction": "Generated MSA translation",
  "model": "model-name"
}
```

## Evaluation

The evaluation script (`evaluate.py`) works with outputs from both zero-shot and SFT models to compute comprehensive translation metrics.

### Basic Usage

```bash
# Evaluate zero-shot results
python evaluate.py --results_file output/zero-shot/gemini-1.5-pro.json
python evaluate.py --results_file output/zero-shot/gpt-4.json

# Evaluate SFT results
python evaluate.py --results_file output/sft/sft-msa-dialect-final.json

# Specify custom BERT model for BERTScore
python evaluate.py --results_file output/sft/results.json --bert_model mbert-uncased
```

### Metrics Computed

The script calculates the following metrics for both translation directions (MSA↔Dialect):

1. **BLEU Scores**: BLEU-1, BLEU-2, BLEU-3, BLEU-4
2. **BERTScore**: Precision, Recall, F1 (semantic similarity)
3. **Individual & Average**: Per-sample and corpus-level metrics

### Output Files

Results are saved in the `results/` directory with detailed JSON files:

- **Individual scores**: Per-sample metrics for error analysis
- **Summary statistics**: Average scores and corpus-level metrics
- **Both directions**: Separate evaluation for MSA→Dialect and Dialect→MSA

### Input Format

The script accepts JSONL files with the standard output format from both zero-shot and SFT models:

```json
{
  "original_msa": "source MSA text",
  "original_dialect": "reference dialect text",
  "msa_to_dialect_prediction": "generated dialect translation",
  "dialect_to_msa_prediction": "generated MSA translation"
}
```

## Model Training Details

### SFT Configuration

- **Base Model**: `google/gemma-3-1b-it` (configurable in `train.py`)
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **LoRA Config**: r=64, alpha=16, dropout=0.05
- **Training**: 3 epochs, learning rate 2e-4
- **Batch Size**: 4 per device, gradient accumulation 8 steps
- **Max Length**: 512 tokens

### Zero-shot Models Supported

- **OpenAI**: GPT-4, GPT-3.5-turbo
- **Google**: Gemini-Pro, Gemini-1.5-Pro
- **HuggingFace**: Any causal LM (Gemma, Qwen, etc.)

## API Key Management

API keys are loaded from environment variables. To use:

1. **Make .env executable**: `chmod +x .env`
2. **Edit .env file** with your actual API keys
3. **Source before running**: `source .env`
4. **Run evaluations** - the script will use the environment variables

The script will show clear error messages if API keys are missing.

## Translation Approaches

### Zero-shot Evaluation

The framework performs zero-shot translation evaluation with inline prompts:

- **MSA → Dialect**: Translates Modern Standard Arabic to regional dialects
- **Dialect → MSA**: Translates regional dialects to Modern Standard Arabic

Uses API-based models (OpenAI, Google) or HuggingFace models without fine-tuning.

### Supervised Fine-Tuning (SFT)

Fine-tunes models on the dialect translation task:

- **Instruction Format**: Uses structured prompts with Instruction/Input/Response
- **Bidirectional**: Trains on both MSA→Dialect and Dialect→MSA directions
- **LoRA Fine-tuning**: Efficient parameter updates using Low-Rank Adaptation
- **Local Inference**: Runs entirely offline after training

Both approaches generate results in the same format for easy comparison and evaluation.
