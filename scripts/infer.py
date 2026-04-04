from transformers import AutoTokenizer, pipeline

model_path = "dialect_translation/models/sft-msa-dialect-final"
tokenizer = AutoTokenizer.from_pretrained(model_path)

pipe = pipeline("text-generation", model=model_path, tokenizer=tokenizer, device=0)

prompt = """Instruction: Translate the following Modern Standard Arabic (MSA) text into the North Africa dialect of Morocco.
Input: يحرص الناس على النظافة لأنها جزء مهم من حياتهم اليومية.
Response:"""

result = pipe(prompt, max_new_tokens=200)[0]["generated_text"]
print("🔮 Model Output:\n", result)
