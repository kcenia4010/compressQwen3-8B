from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = 'Qwen/Qwen3-8B'

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

save_path = "./Qwen3-8B-4bit"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16"
)

model_4bit = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto"
)

print(f"Save 4-bit model to {save_path}...")
model_4bit.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
