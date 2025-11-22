"""
Model Training Script for ESports Strategy AI
Fine-tunes an open-source LLM using LoRA for efficient training.

Supported Models:
- google/gemma-2b-it (default, efficient)
- meta-llama/Llama-2-7b-chat-hf
- mistralai/Mistral-7B-Instruct-v0.2
- tiiuae/falcon-7b-instruct

Usage:
    python train.py                          # Train with defaults
    python train.py --model gemma-2b         # Specify model
    python train.py --epochs 5 --batch-size 4
    python train.py --resume ./checkpoints/checkpoint-500
"""

import os
import sys
import json
import argparse
import torch
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List

from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
    PeftModel
)


# Available models
MODELS = {
    "gemma-2b": "google/gemma-2b-it",
    "gemma-7b": "google/gemma-7b-it",
    "llama2-7b": "meta-llama/Llama-2-7b-chat-hf",
    "llama3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.2",
    "falcon-7b": "tiiuae/falcon-7b-instruct",
}

# LoRA target modules per model
LORA_TARGETS = {
    "gemma": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "mistral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "falcon": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
}


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Model
    model_name: str = "google/gemma-2b-it"
    
    # Training
    num_epochs: int = 3
    batch_size: int = 2
    gradient_accumulation: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    max_seq_length: int = 2048
    
    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    
    # Quantization
    use_4bit: bool = True
    
    # Paths
    data_dir: str = "./data/combined_training"
    output_dir: str = "./models/esports-lora"
    checkpoint_dir: str = "./checkpoints"
    
    # Resume
    resume_from: Optional[str] = None


class ESportsTrainer:
    """Handles model training with LoRA."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
    def setup(self):
        """Setup model, tokenizer, and LoRA."""
        print("\n" + "=" * 60)
        print("   Setting up training environment")
        print("=" * 60)
        
        # Check GPU
        if torch.cuda.is_available():
            print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("⚠️  No GPU detected! Training will be very slow.")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                sys.exit(0)
        
        # Load tokenizer
        print(f"\n📥 Loading tokenizer: {self.config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Quantization config
        print("📦 Setting up 4-bit quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.config.use_4bit,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        # Load model
        print(f"📥 Loading model: {self.config.model_name}")
        print("   (This may take a few minutes...)")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        
        # Prepare for training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Setup LoRA
        print("🔧 Configuring LoRA adapters...")
        target_modules = self._get_target_modules()
        
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        trainable, total = self.model.get_nb_trainable_parameters()
        print(f"✅ Model setup complete!")
        print(f"   Trainable parameters: {trainable:,} ({100 * trainable / total:.2f}%)")
        print(f"   Total parameters: {total:,}")
    
    def _get_target_modules(self) -> List[str]:
        """Get LoRA target modules based on model architecture."""
        model_lower = self.config.model_name.lower()
        
        for key, targets in LORA_TARGETS.items():
            if key in model_lower:
                return targets
        
        # Default
        return ["q_proj", "v_proj"]
    
    def load_data(self):
        """Load and prepare training data."""
        print(f"\n📊 Loading dataset from {self.config.data_dir}")
        
        try:
            dataset = load_from_disk(self.config.data_dir)
            print(f"   ✅ Loaded {len(dataset['train'])} training samples")
            print(f"   ✅ Loaded {len(dataset['validation'])} validation samples")
        except Exception as e:
            print(f"   ❌ Error loading dataset: {e}")
            print("   Run prepare_data.py first!")
            sys.exit(1)
        
        # Format prompts
        print("📝 Formatting prompts...")
        
        def format_prompt(example):
            if example.get("input", "").strip():
                text = f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}</s>"""
            else:
                text = f"""### Instruction:
{example['instruction']}

### Response:
{example['output']}</s>"""
            return {"text": text}
        
        train_formatted = dataset['train'].map(format_prompt, remove_columns=dataset['train'].column_names)
        val_formatted = dataset['validation'].map(format_prompt, remove_columns=dataset['validation'].column_names)
        
        # Tokenize
        print("🔤 Tokenizing...")
        
        def tokenize(examples):
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.config.max_seq_length,
                padding="max_length",
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized
        
        self.train_dataset = train_formatted.map(tokenize, batched=True, remove_columns=["text"])
        self.val_dataset = val_formatted.map(tokenize, batched=True, remove_columns=["text"])
        
        print(f"   ✅ Training samples: {len(self.train_dataset)}")
        print(f"   ✅ Validation samples: {len(self.val_dataset)}")
    
    def train(self):
        """Run training."""
        print("\n" + "=" * 60)
        print("   Starting Training")
        print("=" * 60)
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{self.config.output_dir}_{timestamp}"
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            logging_steps=10,
            save_strategy="steps",
            save_steps=100,
            eval_strategy="steps",
            eval_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=True,
            report_to="none",  # Set to "wandb" if using Weights & Biases
            optim="paged_adamw_8bit",
            gradient_checkpointing=True,
            max_grad_norm=0.3,
            group_by_length=True,
            save_total_limit=3,
        )
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
        )
        
        # Trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Resume from checkpoint if specified
        resume_checkpoint = None
        if self.config.resume_from:
            resume_checkpoint = self.config.resume_from
            print(f"📂 Resuming from checkpoint: {resume_checkpoint}")
        
        # Train!
        print("\n🚀 Training started!")
        print(f"   Epochs: {self.config.num_epochs}")
        print(f"   Batch size: {self.config.batch_size}")
        print(f"   Gradient accumulation: {self.config.gradient_accumulation}")
        print(f"   Effective batch size: {self.config.batch_size * self.config.gradient_accumulation}")
        print(f"   Learning rate: {self.config.learning_rate}")
        print()
        
        train_result = self.trainer.train(resume_from_checkpoint=resume_checkpoint)
        
        # Print results
        print("\n" + "=" * 60)
        print("   Training Complete!")
        print("=" * 60)
        print(f"   Training loss: {train_result.metrics['train_loss']:.4f}")
        
        # Save final model
        final_path = f"{self.config.output_dir}_final"
        print(f"\n💾 Saving model to {final_path}")
        self.model.save_pretrained(final_path)
        self.tokenizer.save_pretrained(final_path)
        
        # Save config
        config_path = Path(final_path) / "training_config.json"
        with open(config_path, "w") as f:
            json.dump(vars(self.config), f, indent=2)
        
        print(f"✅ Model saved successfully!")
        print(f"\nTo use your trained model:")
        print(f"   1. Set LOCAL_MODEL_PATH={final_path} in .env")
        print(f"   2. Set LLM_BACKEND=local in .env")
        print(f"   3. Restart the server")
        
        return train_result
    
    def test_generation(self, prompt: str = None):
        """Test the trained model with a sample prompt."""
        print("\n🧪 Testing model generation...")
        
        if prompt is None:
            prompt = "Summarize a League of Legends match where the blue team won through superior teamfighting."
        
        formatted = f"""### Instruction:
{prompt}

### Response:
"""
        
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "### Response:" in response:
            response = response.split("### Response:")[-1].strip()
        
        print(f"\nPrompt: {prompt}")
        print(f"\nGenerated Response:\n{response}")


def main():
    parser = argparse.ArgumentParser(description="Train ESports Strategy AI model")
    parser.add_argument("--model", type=str, default="gemma-2b", 
                       choices=list(MODELS.keys()),
                       help="Model to fine-tune")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Training batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--data-dir", type=str, default="./data/combined_training", help="Training data path")
    parser.add_argument("--output-dir", type=str, default="./models/esports-lora", help="Output directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--test-only", action="store_true", help="Only test existing model")
    
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig(
        model_name=MODELS[args.model],
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        lora_r=args.lora_r,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        resume_from=args.resume,
    )
    
    print("=" * 60)
    print("   ESports Strategy AI - Model Training")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"   Model: {config.model_name}")
    print(f"   Epochs: {config.num_epochs}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Learning rate: {config.learning_rate}")
    print(f"   LoRA rank: {config.lora_r}")
    
    # Initialize trainer
    trainer = ESportsTrainer(config)
    
    # Setup
    trainer.setup()
    
    if not args.test_only:
        # Load data
        trainer.load_data()
        
        # Train
        trainer.train()
    
    # Test
    trainer.test_generation()
    
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
