import torch
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = 'Qwen/Qwen3-8B'
save_path = "./Qwen3-8B-4bit"
SUBSET_RATIO = 0.2

def load_mmlu(ratio=0.2):
    print(f"Loading MMLU subset ({ratio*100}%)...")
    dataset = load_dataset("cais/mmlu", "all", split="test")
    df = dataset.to_pandas()
    
    subset_df = df.groupby('subject', group_keys=False).apply(
        lambda x: x.sample(frac=ratio, random_state=42)
    )
    
    return subset_df

def evaluate_model(model, tokenizer, data_df):
    model.eval()
    correct = 0
    choices = ['A', 'B', 'C', 'D']
    
    choice_ids = [tokenizer.encode(c, add_special_tokens=False)[-1] for c in choices]
    prompt_template = "Question: {question}\nChoices:\nA. {a}\nB. {b}\nC. {c}\nD. {d}\nAnswer:"

    for _, row in tqdm(data_df.iterrows(), total=len(data_df), desc="Evaluating"):
        prompt = prompt_template.format(
            question=row['question'],
            a=row['choices'][0], b=row['choices'][1],
            c=row['choices'][2], d=row['choices'][3]
        )
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Берем логиты последнего предсказанного токена
            last_logits = outputs.logits[0, -1, choice_ids]
            prediction = torch.argmax(last_logits).item()
            
            if prediction == row['answer']:
                correct += 1
                
    return correct / len(data_df)

# Загрузка токенизатора и модели в FP16
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model_baseline = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.float16, 
    device_map="auto"
)

# Считаем параметры
orig_params = sum(p.numel() for p in model_baseline.parameters())
print(f"Original parameters: {orig_params / 1e9:.2f}B")

test_df = load_mmlu(SUBSET_RATIO)
test_df

orig_metric = evaluate_model(model_baseline, tokenizer, test_df)
print(f"Original Metric (MMLU subset): {orig_metric:.4f}")


model_4bit = AutoModelForCausalLM.from_pretrained(
    save_path,
    load_in_4bit=True,
    device_map="auto"
)

# Считаем параметры
compressed_params = sum(p.numel() for p in model_4bit.parameters())
print(f"Compressed parameters: {compressed_params / 1e9:.2f}B")

compressed_metric = evaluate_model(model_4bit, tokenizer, test_df)
print(f"Compressed Metric (MMLU subset): {compressed_metric:.4f}")
