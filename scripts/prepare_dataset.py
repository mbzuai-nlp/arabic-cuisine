import json
import pandas as pd
from datasets import Dataset, DatasetDict

# ------------------------------------------------
# Load original data
# ------------------------------------------------
with open("dialect_translation/data/train.jsonl", "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

print(f"Loaded {len(data)} original dialogue samples")


# ------------------------------------------------
# Expand each dialogue into two directions
# ------------------------------------------------
def expand_records(data):
    records = []
    for record in data:
        # MSA → Dialect
        records.append(
            {
                "instruction": (
                    f"Translate the following Modern Standard Arabic (MSA) text "
                    f"into the {record['region']} dialect of {record['country']}."
                ),
                "input": record["dialogue_msa"],
                "output": record["dialogue_dialect"],
            }
        )

        # Dialect → MSA
        records.append(
            {
                "instruction": (
                    f"Translate the following Arabic text from the "
                    f"{record['region']} dialect of {record['country']} "
                    f"into Modern Standard Arabic (MSA)."
                ),
                "input": record["dialogue_dialect"],
                "output": record["dialogue_msa"],
            }
        )

    print(f"Expanded to {len(records)} training samples ({len(data)} × 2)")
    return records


# ------------------------------------------------
# Build training dataset
# ------------------------------------------------
train_records = expand_records(data)
train_df = pd.DataFrame(train_records)

train_dataset = Dataset.from_pandas(train_df, preserve_index=False)

dataset = DatasetDict({"train": train_dataset})

print(dataset["train"][0])


# ------------------------------------------------
# Save to disk
# ------------------------------------------------
output_path = "dialect_translation/processed_dataset"
dataset.save_to_disk(output_path)

print(f"✅ Dataset prepared and saved to {output_path}")
