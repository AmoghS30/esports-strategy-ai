"""
AWS GPU Optimized Training Script
Optimized for g5.2xlarge (NVIDIA A10G GPU)

Key optimizations:
- CUDA acceleration
- Larger batch sizes for GPU
- Mixed precision training (FP16)
- Faster data loading
- Optimized for cloud GPU instances
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
# GPU Optimized Models
# ============================================================
MODELS = {
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # 1.1B - FASTEST ⚡
    "phi-2": "microsoft/phi-2",                          # 2.7B - Good speed/quality
    "qwen-0.5b": "Qwen/Qwen2-0.5B-Instruct",           # 0.5B - ULTRA FAST
    "stablelm-2": "stabilityai/stablelm-2-1_6b",       # 1.6B - Fast & good
    "phi-3": "microsoft/Phi-3-mini-4k-instruct",       # 3.8B - Better quality
}

# Optimized LoRA targets
LORA_TARGETS = {
    "tinyllama": ["q_proj", "v_proj", "k_proj", "o_proj"],  # More targets for GPU
    "phi": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "gemma": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "qwen": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "stablelm": ["q_proj", "v_proj", "k_proj", "o_proj"],
}


@dataclass
class GPUConfig:
    """GPU optimized training configuration for g5.2xlarge."""
    # Model
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    # GPU optimized settings (g5.2xlarge has 24GB VRAM)
    num_epochs: int = 3
    batch_size: int = 16  # GPU can handle much larger batches!
    gradient_accumulation: int = 2
    learning_rate: float = 2e-4
    max_seq_length: int = 512  # Longer sequences on GPU
    warmup_steps: int = 50
    
    # LoRA - can use larger rank on GPU
    lora_r: int = 16  # Larger rank for better quality
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    
    # GPU optimizations
    use_gradient_checkpointing: bool = False  # Not needed with 24GB VRAM
    max_grad_norm: float = 1.0
    dataloader_num_workers: int = 4  # Use parallel data loading
    use_flash_attention: bool = True  # Enable if available
    
    # Device
    device: str = "cuda"
    
    # Paths
    data_dir: str = "./data/training_data"
    output_dir: str = "./models/esports-gpu-fast"
    
    # Resume
    resume_from: Optional[str] = None
    
    # Logging
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100


class GPUTrainer:
    """GPU optimized trainer for AWS instances."""
    
    def __init__(self, config: GPUConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        # Verify CUDA
        if not torch.cuda.is_available():
            print("❌ CUDA not available!")
            print("   Make sure you're on a GPU instance")
            sys.exit(1)
        
        print("=" * 60)
        print("   🚀 AWS GPU Training (g5.2xlarge)")
        print("=" * 60)
        print(f"\n✅ GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Set optimal CUDA settings
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        
    def setup_model(self):
        """Load and prepare model - optimized for GPU."""
        print("\n" + "=" * 60)
        print("   GPU Model Setup")
        print("=" * 60)
        
        print(f"\n📥 Loading model: {self.config.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            use_fast=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load model - optimized for GPU
        print(f"📦 Loading to GPU...")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,  # FP16 for speed
            device_map="auto",  # Automatically map to GPU
            low_cpu_mem_usage=True,
        )
        
        print(f"✅ Model loaded to GPU!")
        print(f"   Parameters: {self.model.num_parameters():,}")
        print(f"   Size: {self.model.num_parameters() * 2 / 1e9:.2f} GB (FP16)")
        
        # Check GPU memory
        allocated = torch.cuda.memory_allocated(0) / 1e9
        print(f"   GPU Memory Used: {allocated:.2f} GB")
        
        # Setup LoRA
        print("\n🔧 Setting up LoRA for GPU...")
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
        print(f"   LoRA Rank: {self.config.lora_r} (higher quality)")
    
    def _get_target_modules(self):
        """Get LoRA target modules."""
        model_lower = self.config.model_name.lower()
        
        for key, targets in LORA_TARGETS.items():
            if key in model_lower:
                return targets
        
        return ["q_proj", "v_proj", "k_proj", "o_proj"]
    
    def load_data(self):
        """Load and prepare training data - GPU optimized."""
        print("\n📊 Loading dataset...")
        
        try:
            dataset = load_from_disk(self.config.data_dir)
            print(f"   ✓ Train: {len(dataset['train']):,} samples")
            print(f"   ✓ Validation: {len(dataset['validation']):,} samples")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            print("   Make sure training data is available")
            sys.exit(1)
        
        # Format prompts
        print("📝 Formatting prompts...")
        
        def format_prompt(example):
            if example.get("input", "").strip():
                text = f"{example['instruction']}\n\n{example['input']}\n\n{example['output']}</s>"
            else:
                text = f"{example['instruction']}\n\n{example['output']}</s>"
            return {"text": text}
        
        train_formatted = dataset['train'].map(
            format_prompt, 
            remove_columns=dataset['train'].column_names,
            desc="Formatting train",
            num_proc=4  # Parallel processing on Linux
        )
        val_formatted = dataset['validation'].map(
            format_prompt, 
            remove_columns=dataset['validation'].column_names,
            desc="Formatting validation",
            num_proc=4
        )
        
        # Tokenize
        print("🔤 Tokenizing (parallel processing)...")
        
        def tokenize(examples):
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.config.max_seq_length,
                padding="max_length",
                return_tensors=None,
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized
        
        self.train_dataset = train_formatted.map(
            tokenize, 
            batched=True,
            batch_size=1000,
            remove_columns=["text"],
            num_proc=4,
            desc="Tokenizing train"
        )
        
        self.val_dataset = val_formatted.map(
            tokenize, 
            batched=True,
            batch_size=1000,
            remove_columns=["text"],
            num_proc=4,
            desc="Tokenizing validation"
        )
        
        print(f"✅ Data ready!")
        print(f"   Train: {len(self.train_dataset):,} samples")
        print(f"   Validation: {len(self.val_dataset):,} samples")
    
    def train(self):
        """Run training - GPU optimized."""
        print("\n" + "=" * 60)
        print("   🚀 GPU Training Started")
        print("=" * 60)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{self.config.output_dir}_{timestamp}"
        
        # GPU optimized training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            
            # Logging
            logging_steps=self.config.logging_steps,
            logging_first_step=True,
            
            # Evaluation
            eval_strategy="steps",
            eval_steps=self.config.eval_steps,
            
            # Saving
            save_strategy="steps",
            save_steps=self.config.save_steps,
            save_total_limit=2,
            
            # GPU optimizations
            fp16=True,  # Mixed precision training
            dataloader_num_workers=self.config.dataloader_num_workers,
            dataloader_pin_memory=True,
            dataloader_persistent_workers=True,
            
            # Gradient
            max_grad_norm=self.config.max_grad_norm,
            
            # Memory optimization
            gradient_checkpointing=self.config.use_gradient_checkpointing,
            optim="adamw_torch_fused",  # Faster fused optimizer
            
            # Reporting
            report_to="none",
            load_best_model_at_end=False,
            
            # Disable unnecessary features
            push_to_hub=False,
        )
        
        # Data collator
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
        
        print(f"\n📊 GPU Training Configuration:")
        print(f"   Model: {self.config.model_name.split('/')[-1]}")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Epochs: {self.config.num_epochs}")
        print(f"   Batch size: {self.config.batch_size}")
        print(f"   Gradient accumulation: {self.config.gradient_accumulation}")
        print(f"   Effective batch: {total_batch}")
        print(f"   Sequence length: {self.config.max_seq_length}")
        print(f"   Total steps: {total_steps}")
        print(f"   LoRA rank: {self.config.lora_r}")
        print(f"   Mixed precision: FP16 ✅")
        
        # Estimate time on g5.2xlarge
        samples_per_sec = {
            "tinyllama": 80,   # ~80 samples/sec on g5.2xlarge
            "qwen-0.5b": 120,  # ~120 samples/sec
            "stablelm": 60,    # ~60 samples/sec
            "phi": 40,         # ~40 samples/sec
        }
        
        model_key = next((k for k in samples_per_sec.keys() if k in self.config.model_name.lower()), "tinyllama")
        est_samples_per_sec = samples_per_sec[model_key]
        total_samples = len(self.train_dataset) * self.config.num_epochs
        est_time_min = total_samples / est_samples_per_sec / 60
        
        print(f"\n⏱️  Estimated time on g5.2xlarge: {est_time_min:.1f} minutes")
        print(f"   (~{est_samples_per_sec} samples/second)")
        print(f"\n💰 Estimated cost: ${est_time_min/60 * 0.80:.3f} (spot pricing)")
        
        input("\n✅ Press Enter to start GPU training...")
        
        # Train!
        print("\n🚀 Training started!")
        print(f"⚡ GPU acceleration enabled!\n")
        
        start_time = time.time()
        
        try:
            train_result = self.trainer.train(
                resume_from_checkpoint=self.config.resume_from
            )
            
            elapsed = time.time() - start_time
            
            print("\n" + "=" * 60)
            print("   ✅ Training Complete!")
            print("=" * 60)
            print(f"   Time: {elapsed/60:.1f} minutes ({elapsed:.0f} seconds)")
            print(f"   Speed: {total_samples / elapsed:.1f} samples/second")
            print(f"   Final loss: {train_result.metrics.get('train_loss', 'N/A'):.4f}")
            print(f"   Cost: ${elapsed/3600 * 0.80:.3f} (spot)")
            
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
                "device": f"GPU - {torch.cuda.get_device_name(0)}",
                "instance": "g5.2xlarge",
                "time_minutes": elapsed/60,
                "samples_per_second": total_samples / elapsed,
                "final_loss": train_result.metrics.get('train_loss'),
                "epochs": self.config.num_epochs,
                "lora_rank": self.config.lora_r,
                "batch_size": self.config.batch_size,
                "cost_usd": elapsed/3600 * 0.80,
            }, f, indent=2)
        
        print(f"✅ Model saved!")
        print(f"\n🎉 Training complete! Model ready for inference.")
        
        return train_result


def main():
    parser = argparse.ArgumentParser(description="AWS GPU Training (g5.2xlarge)")
    parser.add_argument("--model", type=str, default="tinyllama",
                       choices=list(MODELS.keys()),
                       help="Model to train")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Batch size (16-32 for g5.2xlarge)")
    parser.add_argument("--lora-rank", type=int, default=16,
                       help="LoRA rank (8-32)")
    parser.add_argument("--quick", action="store_true",
                       help="Quick mode: 1 epoch")
    
    args = parser.parse_args()
    
    # Quick mode
    if args.quick:
        print("🚀 Quick mode enabled!")
        args.epochs = 1
    
    config = GPUConfig(
        model_name=MODELS[args.model],
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lora_r=args.lora_rank,
    )
    
    print("=" * 60)
    print("   AWS GPU Training (g5.2xlarge)")
    print("=" * 60)
    print(f"\n⚡ GPU acceleration enabled")
    print(f"   Model: {args.model}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch: {args.batch_size}")
    print(f"   LoRA Rank: {args.lora_rank}")
    
    trainer = GPUTrainer(config)
    trainer.setup_model()
    trainer.load_data()
    trainer.train()
    
    print("\n✅ Done! Your model is trained on GPU! 🚀")


if __name__ == "__main__":
    main()