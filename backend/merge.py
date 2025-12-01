from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

BASE_MODEL = "google/gemma-2b"   # << CHANGE THIS to your original base model
ADAPTER = "./models/esports-llm_20251130_153250/checkpoint-4300"
OUTPUT = "./models/esports-llm_final"

print("Loading base model...")
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="cuda"
)

print("Loading LoRA adapter...")
lora = PeftModel.from_pretrained(base, ADAPTER)

print("Merging and unloading...")
merged = lora.merge_and_unload()

print("Saving final merged model...")
merged.save_pretrained(OUTPUT)

print("Saving tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.save_pretrained(OUTPUT)

print("\n✅ DONE — Final model saved at:", OUTPUT)
