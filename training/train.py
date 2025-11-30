"""
Enhanced ESports Strategy Analyzer - Training Script
Supports Gemma-2B, Falcon-7B-Instruct and other open LLMs
Includes data preprocessing for game logs and commentary alignment

Key Features:
- Multiple open LLM support (Gemma, Falcon, Phi, Qwen, etc.)
- Game log parsing and preprocessing
- Commentary-event alignment
- Prompt engineering templates
- Fine-tuning on summarization tasks
"""

import os
import sys
import json
import argparse
import time
import re
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import pandas as pd

import torch
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
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
# Enhanced Model Support
# ============================================================
MODELS = {
    # Ultra-lightweight models (< 1B)
    "qwen-0.5b": "Qwen/Qwen2-0.5B-Instruct",
    
    # Lightweight models (1-2B) - RECOMMENDED FOR SUMMARIZATION
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "stablelm-2": "stabilityai/stablelm-2-1_6b",
    "gemma-2b": "google/gemma-2b-it",  # NEW: Gemma 2B Instruct
    
    # Medium models (2-4B)
    "phi-2": "microsoft/phi-2",
    
    # Larger models (4-8B) - Better quality, slower
    "phi-3": "microsoft/Phi-3-mini-4k-instruct",
    "falcon-7b": "tiiuae/falcon-7b-instruct",  # NEW: Falcon 7B Instruct
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.2",
}

# Optimized LoRA targets for each model family
LORA_TARGETS = {
    "tinyllama": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "phi": ["q_proj", "v_proj", "k_proj", "o_proj", "fc1", "fc2"],
    "gemma": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "qwen": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "stablelm": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "falcon": ["query_key_value", "dense"],  # Falcon-specific
    "mistral": ["q_proj", "v_proj", "k_proj", "o_proj"],
}


# ============================================================
# Data Preprocessing Utilities
# ============================================================

class GameLogParser:
    """Parse structured game logs and stats"""
    
    @staticmethod
    def parse_event_log(log_file: str) -> pd.DataFrame:
        """
        Parse game event log into structured DataFrame
        Expected format: timestamp, event_type, team, details
        """
        events = []
        
        with open(log_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Parse different event formats
                # Format 1: "00:05:23 | KILL | Team A | Player1 killed Player2"
                # Format 2: "00:12:45 | OBJECTIVE | Team B | Baron secured"
                match = re.match(r'(\d{2}:\d{2}:\d{2})\s*\|\s*(\w+)\s*\|\s*([^|]+)\s*\|\s*(.+)', line)
                
                if match:
                    timestamp, event_type, team, details = match.groups()
                    events.append({
                        'timestamp': timestamp,
                        'event_type': event_type.strip(),
                        'team': team.strip(),
                        'details': details.strip(),
                        'seconds': GameLogParser._time_to_seconds(timestamp)
                    })
        
        return pd.DataFrame(events)
    
    @staticmethod
    def _time_to_seconds(timestamp: str) -> int:
        """Convert HH:MM:SS to total seconds"""
        parts = timestamp.split(':')
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    
    @staticmethod
    def extract_match_metadata(log_file: str) -> Dict:
        """Extract match metadata from log header"""
        metadata = {
            'team_a': 'Team A',
            'team_b': 'Team B',
            'date': None,
            'game': 'Unknown',
            'duration': None
        }
        
        with open(log_file, 'r') as f:
            for line in f:
                if line.startswith('# Team A:'):
                    metadata['team_a'] = line.split(':', 1)[1].strip()
                elif line.startswith('# Team B:'):
                    metadata['team_b'] = line.split(':', 1)[1].strip()
                elif line.startswith('# Date:'):
                    metadata['date'] = line.split(':', 1)[1].strip()
                elif line.startswith('# Game:'):
                    metadata['game'] = line.split(':', 1)[1].strip()
                elif line.startswith('# Duration:'):
                    metadata['duration'] = line.split(':', 1)[1].strip()
        
        return metadata
    
    @staticmethod
    def parse_team_composition(comp_file: str) -> Dict[str, List[str]]:
        """Parse team composition (hero/champion picks)"""
        compositions = {'team_a': [], 'team_b': []}
        
        with open(comp_file, 'r') as f:
            current_team = None
            for line in f:
                line = line.strip()
                if line.startswith('Team A:'):
                    current_team = 'team_a'
                elif line.startswith('Team B:'):
                    current_team = 'team_b'
                elif line and current_team:
                    # Parse champion/hero names
                    compositions[current_team].append(line)
        
        return compositions


class CommentaryAligner:
    """Align commentary text with game events"""
    
    @staticmethod
    def parse_commentary(commentary_file: str) -> List[Dict]:
        """
        Parse commentary with timestamps
        Format: "[00:05:23] Commentary text here..."
        """
        commentary_segments = []
        
        with open(commentary_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Extract timestamp and text
                match = re.match(r'\[(\d{2}:\d{2}:\d{2})\]\s*(.+)', line)
                if match:
                    timestamp, text = match.groups()
                    commentary_segments.append({
                        'timestamp': timestamp,
                        'text': text.strip(),
                        'seconds': GameLogParser._time_to_seconds(timestamp)
                    })
        
        return commentary_segments
    
    @staticmethod
    def align_events_with_commentary(
        events_df: pd.DataFrame,
        commentary: List[Dict],
        window_seconds: int = 30
    ) -> List[Dict]:
        """
        Align game events with commentary within a time window
        Returns aligned data suitable for training
        """
        aligned_data = []
        
        for _, event in events_df.iterrows():
            event_time = event['seconds']
            
            # Find commentary within window
            relevant_commentary = []
            for comm in commentary:
                time_diff = abs(comm['seconds'] - event_time)
                if time_diff <= window_seconds:
                    relevant_commentary.append(comm['text'])
            
            if relevant_commentary:
                aligned_data.append({
                    'event': event.to_dict(),
                    'commentary': ' '.join(relevant_commentary),
                    'timestamp': event['timestamp']
                })
        
        return aligned_data
    
    @staticmethod
    def identify_turning_points(events_df: pd.DataFrame) -> List[Dict]:
        """
        Identify potential turning points from event log
        Heuristics: multi-kills, objectives, gold swings, etc.
        """
        turning_points = []
        
        # Key event types that often indicate turning points
        key_events = ['ACE', 'BARON', 'ELDER', 'INHIBITOR', 'PENTAKILL', 'TEAMFIGHT']
        
        for idx, event in events_df.iterrows():
            if any(key in event['event_type'] for key in key_events):
                turning_points.append({
                    'timestamp': event['timestamp'],
                    'event_type': event['event_type'],
                    'team': event['team'],
                    'details': event['details'],
                    'significance': 'high'
                })
        
        return turning_points


class DatasetBuilder:
    """Build training datasets for esports summarization"""
    
    @staticmethod
    def create_summarization_dataset(
        match_logs_dir: str,
        commentary_dir: str,
        output_dir: str = "./data/training_data"
    ):
        """
        Create training dataset from match logs and commentary
        Combines DialogSum-style data with esports-specific examples
        """
        print("\n" + "=" * 60)
        print("   Building ESports Summarization Dataset")
        print("=" * 60)
        
        # Step 1: Load base summarization dataset (DialogSum)
        print("\n📥 Loading base summarization dataset...")
        try:
            dialogsum = load_dataset("knkarthick/dialogsum", trust_remote_code=True)
            print(f"   ✓ Loaded {len(dialogsum['train'])} DialogSum examples")
        except Exception as e:
            print(f"   ⚠️  Could not load DialogSum: {e}")
            print("   Continuing with esports data only...")
            dialogsum = None
        
        # Step 2: Process esports match data
        print("\n🎮 Processing esports match data...")
        esports_examples = []
        
        match_files = list(Path(match_logs_dir).glob("*.log"))
        print(f"   Found {len(match_files)} match log files")
        
        for match_file in tqdm(match_files, desc="Processing matches"):
            try:
                # Parse match data
                events_df = GameLogParser.parse_event_log(str(match_file))
                metadata = GameLogParser.extract_match_metadata(str(match_file))
                
                # Find corresponding commentary
                commentary_file = Path(commentary_dir) / f"{match_file.stem}_commentary.txt"
                if commentary_file.exists():
                    commentary = CommentaryAligner.parse_commentary(str(commentary_file))
                    aligned = CommentaryAligner.align_events_with_commentary(events_df, commentary)
                    turning_points = CommentaryAligner.identify_turning_points(events_df)
                    
                    # Create training example
                    example = DatasetBuilder._create_training_example(
                        metadata, events_df, commentary, aligned, turning_points
                    )
                    esports_examples.append(example)
            
            except Exception as e:
                print(f"   ⚠️  Error processing {match_file.name}: {e}")
                continue
        
        print(f"\n   ✓ Created {len(esports_examples)} esports examples")
        
        # Step 3: Combine datasets
        print("\n🔗 Combining datasets...")
        all_examples = esports_examples
        
        if dialogsum:
            # Convert DialogSum to our format
            for example in dialogsum['train']:
                all_examples.append({
                    'instruction': 'Summarize the following dialogue:',
                    'input': example['dialogue'],
                    'output': example['summary'],
                    'category': 'dialogue'
                })
        
        # Step 4: Split and save
        print("\n💾 Saving dataset...")
        train_size = int(0.9 * len(all_examples))
        
        train_data = all_examples[:train_size]
        val_data = all_examples[train_size:]
        
        dataset = DatasetDict({
            'train': Dataset.from_list(train_data),
            'validation': Dataset.from_list(val_data)
        })
        
        dataset.save_to_disk(output_dir)
        
        print(f"\n✅ Dataset saved to {output_dir}")
        print(f"   Train: {len(train_data)} examples")
        print(f"   Validation: {len(val_data)} examples")
        print(f"   ESports: {len(esports_examples)} examples")
        if dialogsum:
            print(f"   DialogSum: {len(dialogsum['train'])} examples")
        
        return dataset
    
    @staticmethod
    def _create_training_example(
        metadata: Dict,
        events_df: pd.DataFrame,
        commentary: List[Dict],
        aligned: List[Dict],
        turning_points: List[Dict]
    ) -> Dict:
        """Create a single training example from match data"""
        
        # Create match context
        context = f"""Match: {metadata['team_a']} vs {metadata['team_b']}
Duration: {metadata['duration']}
Game: {metadata['game']}

Key Events:
"""
        
        # Add key events
        key_events = events_df.head(20)  # Top 20 events
        for _, event in key_events.iterrows():
            context += f"[{event['timestamp']}] {event['event_type']}: {event['details']}\n"
        
        # Add commentary snippets
        if commentary:
            context += "\nCommentary:\n"
            for comm in commentary[:10]:  # First 10 commentary pieces
                context += f"[{comm['timestamp']}] {comm['text']}\n"
        
        # Generate summary (ground truth would come from human annotation)
        # For now, create a template-based summary
        summary = f"In this {metadata['duration']} match between {metadata['team_a']} and {metadata['team_b']}, "
        
        if turning_points:
            summary += f"key turning points included {len(turning_points)} major events. "
            for tp in turning_points[:3]:
                summary += f"At {tp['timestamp']}, {tp['details']}. "
        
        return {
            'instruction': 'Analyze this esports match and provide a summary highlighting key moments and turning points:',
            'input': context,
            'output': summary,
            'category': 'esports',
            'metadata': metadata
        }


# ============================================================
# Prompt Engineering Templates
# ============================================================

class PromptTemplates:
    """Enhanced prompt templates for different analysis types"""
    
    @staticmethod
    def turning_points_prompt(match_context: str) -> str:
        """Template for identifying turning points"""
        return f"""Analyze the following esports match and identify the 3 most critical turning points that determined the outcome.

{match_context}

For each turning point, explain:
1. What happened (specific event and timing)
2. Why it was significant (impact on game state)
3. How it affected the final outcome

Turning Points:"""
    
    @staticmethod
    def tactical_recommendations_prompt(match_context: str, team: str) -> str:
        """Template for generating tactical recommendations"""
        return f"""Based on this match analysis, provide 3 specific tactical recommendations for {team}'s next game.

{match_context}

Focus on:
- Team composition adjustments
- Objective timing optimization
- Strategic positioning improvements
- Win condition execution

Recommendations:"""
    
    @staticmethod
    def focus_area_prompt(match_context: str, focus: str) -> str:
        """Template with specific focus area"""
        focus_descriptions = {
            'team_fights': 'team fight execution and positioning',
            'economy': 'economic decisions and resource management',
            'objectives': 'objective control and map pressure',
            'composition': 'team composition and draft strategy',
            'macro': 'macro-level strategy and rotations'
        }
        
        focus_desc = focus_descriptions.get(focus, 'overall gameplay')
        
        return f"""Analyze this esports match with a focus on {focus_desc}.

{match_context}

Provide a detailed analysis of how this aspect influenced the match outcome:"""
    
    @staticmethod
    def style_prompt(match_context: str, style: str) -> str:
        """Template with different output styles"""
        style_instructions = {
            'analytical': 'Provide a detailed analytical breakdown with statistics and reasoning',
            'narrative': 'Tell the story of the match in an engaging narrative format',
            'bullet': 'Provide a concise bullet-point summary of key events',
            'tactical': 'Focus on tactical decisions and strategic implications for future matches'
        }
        
        instruction = style_instructions.get(style, style_instructions['analytical'])
        
        return f"""{instruction}.

{match_context}

Analysis:"""


# ============================================================
# GPU Configuration
# ============================================================

@dataclass
class GPUConfig:
    """Enhanced GPU training configuration"""
    # Model
    model_name: str = "google/gemma-2b-it"  # Default to Gemma-2B
    
    # GPU optimized settings
    num_epochs: int = 3
    batch_size: int = 16
    gradient_accumulation: int = 2
    learning_rate: float = 2e-4
    max_seq_length: int = 1024  # Longer for match analysis
    warmup_steps: int = 50
    
    # LoRA settings
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    
    # GPU optimizations
    use_gradient_checkpointing: bool = False
    max_grad_norm: float = 1.0
    dataloader_num_workers: int = 4
    use_flash_attention: bool = True
    
    # Device
    device: str = "cuda"
    
    # Paths
    data_dir: str = "./data/training_data"
    output_dir: str = "./models/esports-llm"
    
    # Data preprocessing
    match_logs_dir: Optional[str] = None
    commentary_dir: Optional[str] = None
    
    # Resume
    resume_from: Optional[str] = None
    
    # Logging
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100


class EnhancedTrainer:
    """Enhanced trainer with data preprocessing and prompt engineering"""
    
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
        print("   🚀 Enhanced ESports LLM Training")
        print("=" * 60)
        print(f"\n✅ GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Set optimal CUDA settings
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    
    def preprocess_data(self):
        """Preprocess match logs and commentary if provided"""
        if self.config.match_logs_dir and self.config.commentary_dir:
            print("\n" + "=" * 60)
            print("   Data Preprocessing")
            print("=" * 60)
            
            DatasetBuilder.create_summarization_dataset(
                match_logs_dir=self.config.match_logs_dir,
                commentary_dir=self.config.commentary_dir,
                output_dir=self.config.data_dir
            )
        else:
            print("\n⚠️  No match logs provided - using existing dataset")
    
    def setup_model(self):
        """Load and prepare model"""
        print("\n" + "=" * 60)
        print("   Model Setup")
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
        
        # Load model
        print(f"📦 Loading to GPU...")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        
        print(f"✅ Model loaded!")
        print(f"   Parameters: {self.model.num_parameters():,}")
        print(f"   Size: {self.model.num_parameters() * 2 / 1e9:.2f} GB (FP16)")
        
        # Check GPU memory
        allocated = torch.cuda.memory_allocated(0) / 1e9
        print(f"   GPU Memory Used: {allocated:.2f} GB")
        
        # Setup LoRA
        print("\n🔧 Setting up LoRA...")
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
        print(f"   LoRA Rank: {self.config.lora_r}")
    
    def _get_target_modules(self):
        """Get LoRA target modules for the model"""
        model_lower = self.config.model_name.lower()
        
        for key, targets in LORA_TARGETS.items():
            if key in model_lower:
                return targets
        
        # Default fallback
        return ["q_proj", "v_proj", "k_proj", "o_proj"]
    
    def load_data(self):
        """Load training data"""
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
            desc="Formatting train"
        )
        val_formatted = dataset['validation'].map(
            format_prompt,
            remove_columns=dataset['validation'].column_names,
            desc="Formatting validation"
        )
        
        # Tokenize
        print("🔤 Tokenizing...")
        
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
        """Run training"""
        print("\n" + "=" * 60)
        print("   🚀 Training Started")
        print("=" * 60)
        
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
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            logging_first_step=True,
            eval_strategy="steps",
            eval_steps=self.config.eval_steps,
            save_strategy="steps",
            save_steps=self.config.save_steps,
            save_total_limit=2,
            fp16=True,
            dataloader_num_workers=self.config.dataloader_num_workers,
            dataloader_pin_memory=True,
            dataloader_persistent_workers=True,
            max_grad_norm=self.config.max_grad_norm,
            gradient_checkpointing=self.config.use_gradient_checkpointing,
            optim="adamw_torch_fused",
            report_to="none",
            load_best_model_at_end=False,
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
        
        # Print training info
        total_batch = self.config.batch_size * self.config.gradient_accumulation
        total_steps = len(self.train_dataset) * self.config.num_epochs // total_batch
        
        print(f"\n📊 Training Configuration:")
        print(f"   Model: {self.config.model_name.split('/')[-1]}")
        print(f"   Epochs: {self.config.num_epochs}")
        print(f"   Batch size: {self.config.batch_size}")
        print(f"   Sequence length: {self.config.max_seq_length}")
        print(f"   Total steps: {total_steps}")
        print(f"   LoRA rank: {self.config.lora_r}")
        
        input("\n✅ Press Enter to start training...")
        
        # Train
        start_time = time.time()
        
        try:
            train_result = self.trainer.train(
                resume_from_checkpoint=self.config.resume_from
            )
            
            elapsed = time.time() - start_time
            
            print("\n" + "=" * 60)
            print("   ✅ Training Complete!")
            print("=" * 60)
            print(f"   Time: {elapsed/60:.1f} minutes")
            print(f"   Final loss: {train_result.metrics.get('train_loss', 'N/A'):.4f}")
            
        except KeyboardInterrupt:
            print("\n⚠️  Training interrupted")
            elapsed = time.time() - start_time
            print(f"   Trained for {elapsed/60:.1f} minutes")
        
        # Save model
        final_path = f"{self.config.output_dir}_final"
        print(f"\n💾 Saving to {final_path}...")
        
        self.model.save_pretrained(final_path)
        self.tokenizer.save_pretrained(final_path)
        
        print(f"✅ Training complete!")
        
        return train_result


def main():
    parser = argparse.ArgumentParser(description="Enhanced ESports LLM Training")
    parser.add_argument("--model", type=str, default="gemma-2b",
                       choices=list(MODELS.keys()),
                       help="Model to train")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--match-logs", type=str, default=None,
                       help="Directory with match log files")
    parser.add_argument("--commentary", type=str, default=None,
                       help="Directory with commentary files")
    parser.add_argument("--preprocess-only", action="store_true",
                       help="Only preprocess data, don't train")
    
    args = parser.parse_args()
    
    config = GPUConfig(
        model_name=MODELS[args.model],
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lora_r=args.lora_rank,
        match_logs_dir=args.match_logs,
        commentary_dir=args.commentary,
    )
    
    print("=" * 60)
    print("   Enhanced ESports LLM Training")
    print("=" * 60)
    print(f"\n   Model: {args.model}")
    print(f"   Epochs: {args.epochs}")
    
    trainer = EnhancedTrainer(config)
    
    # Preprocess data if logs provided
    if args.match_logs and args.commentary:
        trainer.preprocess_data()
        
        if args.preprocess_only:
            print("\n✅ Data preprocessing complete!")
            return
    
    trainer.setup_model()
    trainer.load_data()
    trainer.train()
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()