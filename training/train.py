"""
M4 Pro Optimized Training Script
Maximum speed optimizations for Apple Silicon M4

Key optimizations:
- Smaller, faster models
- Optimized batch sizes for M4
- Better MPS utilization
- Memory-efficient processing
- Faster tokenization
- Fixed macOS multiprocessing issues
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Optional

import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm


# ============================================================
# M4 Pro Optimized Models - FASTEST
# ============================================================
MODELS = {
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # 1.1B - FASTEST ⚡
    "phi-2": "microsoft/phi-2",                          # 2.7B - Good speed/quality
    "qwen-0.5b": "Qwen/Qwen2-0.5B-Instruct",           # 0.5B - ULTRA FAST
    "stablelm-2": "stabilityai/stablelm-2-1_6b",       # 1.6B - Fast & good
}

# Optimized LoRA targets for speed
LORA_TARGETS = {
    "tinyllama": ["q_proj", "v_proj"],  # Minimal for speed
    "phi": ["q_proj", "v_proj"],
    "gemma": ["q_proj", "v_proj"],
    "qwen": ["q_proj", "v_proj"],
    "stablelm": ["q_proj", "v_proj"],
}


@dataclass
class M4ProConfig:
    """M4 Pro optimized training configuration."""
    # Model
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    # M4 Pro optimized settings
    num_epochs: int = 3
    batch_size: int = 4  # M4 Pro can handle more
    gradient_accumulation: int = 2  # Smaller accumulation
    learning_rate: float = 3e-4  # Slightly higher for faster convergence
    max_seq_length: int = 384  # Shorter for speed
    warmup_steps: int = 20  # Less warmup
    
    # LoRA - minimal for speed
    lora_r: int = 4  # Smaller rank = faster
    lora_alpha: int = 8
    lora_dropout: float = 0.05
    
    # Speed optimizations
    use_gradient_checkpointing: bool = False  # Faster without it
    max_grad_norm: float = 1.0
    dataloader_num_workers: int = 0  # Disable multiprocessing for macOS compatibility
    
    # Device
    device: str = "mps"
    
    # Paths
    data_dir: str = "./data/training_data"
    output_dir: str = "./models/esports-m4-fast"
    
    # Resume
    resume_from: Optional[str] = None
    
    # Logging - less frequent for speed
    logging_steps: int = 25
    save_steps: int = 200
    eval_steps: int = 200


class M4ProTrainer:
    """M4 Pro optimized trainer with maximum speed."""
    
    def __init__(self, config: M4ProConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        # Verify MPS
        if not torch.backends.mps.is_available():
            print("⚠️  MPS not available! Using CPU (will be slow)")
            self.config.device = "cpu"
        else:
            print("✅ M4 Pro detected - using MPS acceleration")
            # Enable MPS optimizations
            os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
        
    def setup_model(self):
        """Load and prepare model - optimized for speed."""
        print("\n" + "=" * 60)
        print("   M4 Pro Fast Model Setup")
        print("=" * 60)
        
        print(f"\n📥 Loading model: {self.config.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            use_fast=True  # Use fast tokenizer
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load model - optimized for MPS
        print(f"📦 Loading to MPS...")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,  # Use FP16 for speed
            low_cpu_mem_usage=True,
        )
        
        # Move to MPS
        self.model = self.model.to(self.config.device)
        
        print(f"✅ Model loaded!")
        print(f"   Parameters: {self.model.num_parameters():,}")
        print(f"   Size: {self.model.num_parameters() * 2 / 1e9:.2f} GB (FP16)")
        
        # Setup LoRA - minimal for speed
        print("\n🔧 Setting up minimal LoRA (for speed)...")
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
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        print(f"✅ LoRA configured!")
        print(f"   Trainable: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    def _get_target_modules(self):
        """Get minimal LoRA target modules for speed."""
        model_lower = self.config.model_name.lower()
        
        for key, targets in LORA_TARGETS.items():
            if key in model_lower:
                return targets
        
        return ["q_proj", "v_proj"]  # Minimal default
    
    def load_data(self):
        """Load and prepare training data - optimized for macOS."""
        print("\n📊 Loading dataset...")
        
        try:
            dataset = load_from_disk(self.config.data_dir)
            print(f"   ✓ Train: {len(dataset['train']):,} samples")
            print(f"   ✓ Validation: {len(dataset['validation']):,} samples")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            print("   Run: python prepare_data.py --max-samples 5000")
            sys.exit(1)
        
        # Format prompts - simplified for speed
        print("📝 Formatting prompts (single process for macOS stability)...")
        
        def format_prompt(example):
            # Simpler format for speed
            if example.get("input", "").strip():
                text = f"{example['instruction']}\n\n{example['input']}\n\n{example['output']}</s>"
            else:
                text = f"{example['instruction']}\n\n{example['output']}</s>"
            return {"text": text}
        
        # Single process to avoid macOS multiprocessing issues
        train_formatted = dataset['train'].map(
            format_prompt, 
            remove_columns=dataset['train'].column_names,
            desc="Formatting train"
        )
        val_formatted = dataset['validation'].map(
            format_prompt, 
            remove_columns=dataset['validation'].column_names,
            desc="Formatting validation"
        )
        
        # Tokenize - single process for stability
        print("🔤 Tokenizing (single process for macOS stability)...")
        
        def tokenize(examples):
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.config.max_seq_length,
                padding="max_length",
                return_tensors=None,  # Don't convert to tensors yet
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized
        
        # Use batched processing but single process
        self.train_dataset = train_formatted.map(
            tokenize, 
            batched=True,
            batch_size=1000,  # Large batches for speed
            remove_columns=["text"],
            desc="Tokenizing train"
        )
        
        self.val_dataset = val_formatted.map(
            tokenize, 
            batched=True,
            batch_size=1000,
            remove_columns=["text"],
            desc="Tokenizing validation"
        )
        
        print(f"✅ Data ready!")
        print(f"   Train: {len(self.train_dataset):,} samples")
        print(f"   Validation: {len(self.val_dataset):,} samples")
    
    def train(self):
        """Run training - maximum speed optimization."""
        print("\n" + "=" * 60)
        print("   M4 Pro Fast Training")
        print("=" * 60)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{self.config.output_dir}_{timestamp}"
        
        # M4 Pro optimized training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            
            # Logging - less frequent
            logging_steps=self.config.logging_steps,
            logging_first_step=True,
            
            # Evaluation - less frequent
            eval_strategy="steps",
            eval_steps=self.config.eval_steps,
            
            # Saving - less frequent
            save_strategy="steps",
            save_steps=self.config.save_steps,
            save_total_limit=1,  # Keep only 1 checkpoint
            
            # Speed optimizations - disabled multiprocessing for macOS
            dataloader_num_workers=self.config.dataloader_num_workers,
            dataloader_pin_memory=False,  # Disabled for macOS stability
            
            # Gradient
            max_grad_norm=self.config.max_grad_norm,
            
            # MPS specific
            use_cpu=False,
            no_cuda=True,
            fp16=True,  # FP16 for speed on MPS
            
            # Disable slow features
            report_to="none",
            load_best_model_at_end=False,
            metric_for_best_model=None,
            greater_is_better=None,
            
            # Memory optimization
            gradient_checkpointing=self.config.use_gradient_checkpointing,
            optim="adamw_torch",  # Faster than default
            
            # Disable unnecessary features
            push_to_hub=False,
            hub_strategy="end",
        )
        
        # Fast data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
            return_tensors="pt"
        )
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=data_collator,
        )
        
        # Print info
        total_batch = self.config.batch_size * self.config.gradient_accumulation
        total_steps = len(self.train_dataset) * self.config.num_epochs // total_batch
        
        print(f"\n📊 M4 Pro Fast Configuration:")
        print(f"   Model: {self.config.model_name.split('/')[-1]}")
        print(f"   Device: {self.config.device.upper()}")
        print(f"   Epochs: {self.config.num_epochs}")
        print(f"   Batch size: {self.config.batch_size}")
        print(f"   Gradient accumulation: {self.config.gradient_accumulation}")
        print(f"   Effective batch: {total_batch}")
        print(f"   Sequence length: {self.config.max_seq_length}")
        print(f"   Total steps: {total_steps}")
        print(f"   LoRA rank: {self.config.lora_r} (minimal for speed)")
        
        # Estimate time
        samples_per_sec = {
            "tinyllama": 12,   # ~12 samples/sec on M4 Pro
            "qwen-0.5b": 20,   # ~20 samples/sec
            "stablelm": 10,    # ~10 samples/sec
            "phi": 6,          # ~6 samples/sec
        }
        
        model_key = next((k for k in samples_per_sec.keys() if k in self.config.model_name.lower()), "tinyllama")
        est_samples_per_sec = samples_per_sec[model_key]
        total_samples = len(self.train_dataset) * self.config.num_epochs
        est_time_min = total_samples / est_samples_per_sec / 60
        
        print(f"\n⏱️  Estimated time on M4 Pro: {est_time_min:.0f} minutes")
        print(f"   (~{est_samples_per_sec} samples/second)")
        
        input("\n✅ Press Enter to start fast training...")
        
        # Train!
        print("\n🚀 Training started!")
        print(f"💨 Maximum speed mode enabled!\n")
        
        start_time = time.time()
        
        try:
            train_result = self.trainer.train(
                resume_from_checkpoint=self.config.resume_from
            )
            
            elapsed = time.time() - start_time
            
            print("\n" + "=" * 60)
            print("   Training Complete!")
            print("=" * 60)
            print(f"   Time: {elapsed/60:.1f} minutes")
            print(f"   Speed: {total_samples / elapsed:.1f} samples/second")
            print(f"   Final loss: {train_result.metrics.get('train_loss', 'N/A'):.4f}")
            
        except KeyboardInterrupt:
            print("\n⚠️  Training interrupted")
            elapsed = time.time() - start_time
            print(f"   Trained for {elapsed/60:.1f} minutes")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        
        # Save final model
        final_path = f"{self.config.output_dir}_final"
        print(f"\n💾 Saving to {final_path}...")
        
        self.model.save_pretrained(final_path)
        self.tokenizer.save_pretrained(final_path)
        
        # Save stats
        with open(Path(final_path) / "training_info.json", "w") as f:
            json.dump({
                "model": self.config.model_name,
                "device": "M4 Pro (MPS)",
                "time_minutes": elapsed/60,
                "samples_per_second": total_samples / elapsed,
                "final_loss": train_result.metrics.get('train_loss'),
                "epochs": self.config.num_epochs,
                "lora_rank": self.config.lora_r,
            }, f, indent=2)
        
        print(f"✅ Model saved!")
        
        return train_result


def main():
    parser = argparse.ArgumentParser(description="M4 Pro Fast Training")
    parser.add_argument("--model", type=str, default="tinyllama",
                       choices=list(MODELS.keys()),
                       help="Model (tinyllama=fastest, qwen-0.5b=ultra fast)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Batch size (4-8 for M4 Pro)")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Limit training samples")
    parser.add_argument("--quick", action="store_true",
                       help="Quick mode: 1 epoch, 2000 samples")
    
    args = parser.parse_args()
    
    # Quick mode
    if args.quick:
        print("🚀 Quick mode enabled!")
        args.epochs = 1
        # Prepare quick data
        os.system("python prepare_data.py --max-samples 2000")
    
    config = M4ProConfig(
        model_name=MODELS[args.model],
        num_epochs=args.epochs,
        batch_size=args.batch_size,
    )
    
    print("=" * 60)
    print("   M4 Pro Fast Training (macOS Fixed)")
    print("=" * 60)
    print(f"\n💨 Maximum speed optimizations enabled")
    print(f"   Model: {args.model}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch: {args.batch_size}")
    print(f"\n⚠️  Note: Multiprocessing disabled for macOS stability")
    
    trainer = M4ProTrainer(config)
    trainer.setup_model()
    trainer.load_data()
    trainer.train()
    
    print("\n✅ Done! Your model is blazing fast! 🔥")


if __name__ == "__main__":
    main()