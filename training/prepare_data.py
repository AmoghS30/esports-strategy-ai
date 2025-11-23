"""
Data Preparation Script for ESports Strategy AI (MacBook Optimized)
Uses the provided dialogue summarization datasets

Usage:
    python prepare_data.py
    python prepare_data.py --max-samples 5000  # Limit training samples
    python prepare_data.py --add-esports       # Add synthetic esports data
"""

import os
import csv
import json
import argparse
from pathlib import Path
from typing import List, Dict
from datasets import Dataset, DatasetDict
from tqdm import tqdm


class DataPreparation:
    """Prepares training data from uploaded CSV files."""
    
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir = Path("./data")
        
    def load_csv_data(self, filename: str, max_samples: int = None) -> List[Dict]:
        """Load data from CSV file."""
        filepath = self.upload_dir / filename
        
        if not filepath.exists():
            print(f"⚠️  File not found: {filepath}")
            return []
        
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if max_samples and i >= max_samples:
                    break
                data.append(row)
        
        return data
    
    def convert_to_instruction_format(self, dialogue_data: List[Dict]) -> List[Dict]:
        """Convert dialogue data to instruction-following format."""
        training_data = []
        
        for item in dialogue_data:
            # Main summarization task
            training_data.append({
                "instruction": "Summarize the following dialogue concisely, capturing the main points and outcome of the conversation.",
                "input": item['dialogue'],
                "output": item['summary']
            })
            
            # Add variation - focus on specific aspects
            if 'topic' in item and item['topic']:
                training_data.append({
                    "instruction": f"Summarize this dialogue about {item['topic']}. Focus on the key points and decisions made.",
                    "input": item['dialogue'],
                    "output": item['summary']
                })
        
        return training_data
    
    def create_esports_samples(self) -> List[Dict]:
        """Create synthetic esports training samples."""
        esports_samples = [
            {
                "instruction": "Provide an analytical summary of this League of Legends match.",
                "input": """Match: Cloud9 vs Team Liquid (Finals)
Duration: 35 minutes, Winner: Cloud9

Key Events:
- [5:00] First blood to Liquid
- [15:00] Liquid leads by 1500 gold
- [18:00] Dragon fight - C9 wins 4-1 after Azir shuffle
- [25:00] Baron - C9 secures after Thresh hook
- [32:00] Dragon Soul to C9
- [35:00] C9 wins

Team Comps:
C9: Gnar, Lee Sin, Azir, Jinx, Thresh (scaling)
TL: Renekton, Viego, Viktor, Kai'Sa, Nautilus (early game)""",
                "output": "Cloud9 defeated Team Liquid in a 35-minute comeback victory. Despite Liquid's strong early game execution leading to first blood and a 1500 gold lead at 15 minutes, Cloud9's scaling composition prevailed. The turning point came at 18 minutes when Azir's shuffle won a crucial dragon fight 4-1. Cloud9 then secured Baron at 25 minutes after a Thresh hook, and closed the game with Dragon Soul at 32 minutes. This match demonstrates the importance of teamfight execution and respecting scaling compositions."
            },
            {
                "instruction": "Identify the key turning points in this match.",
                "input": """Match: Phoenix vs Dragon Gaming (Dota 2)
Duration: 42 minutes, Winner: Dragon Gaming

Timeline:
- [0-15] Phoenix dominates with push lineup, 8k gold lead
- [18:00] Phoenix fails high ground push, loses 3 heroes
- [25:00] Dragon takes Roshan
- [32:00] Dragon's cores fully online (AM, Storm)
- [42:00] Dragon wins

Compositions:
Phoenix: Luna, Chen (early push)
Dragon: Anti-Mage, Storm Spirit (late game)""",
                "output": """Key Turning Points:

1. Failed High Ground Push (18:00) - CRITICAL
Phoenix attempted to push high ground with their timing advantage but lost 3 heroes. This single mistake gave Dragon's carries the time they desperately needed to farm.

2. Roshan Secure (25:00) - HIGH IMPACT
Dragon secured Roshan with Aegis, providing safety for their cores to farm and fight. This formalized their recovery.

3. Power Spike Reached (32:00) - DECISIVE
Anti-Mage and Storm Spirit reached their item timings. Phoenix's window had completely closed.

The lesson: Timing-based strategies require flawless execution. One failed push can lose the entire game against hard carries."""
            },
            {
                "instruction": "Provide strategic recommendations for the losing team.",
                "input": """Team: Team Liquid (Lost to Cloud9)
What Happened: Strong early game (1500 gold lead at 15min), lost teamfight at dragon (18min), lost Baron fight (25min)
Their Comp: Early game focused (Renekton, Viego)
Enemy Comp: Late game scaling (Jinx, Azir)""",
                "output": """Strategic Recommendations for Team Liquid:

IMMEDIATE FIXES:
1. Dragon Fight Positioning - Spread formation vs Azir to avoid multi-target shuffle
2. Baron Vision - Clear all surrounding vision before starting objective
3. Closing Speed - With 1500 gold lead, force end before enemy scales

SHORT-TERM IMPROVEMENTS:
1. Draft Backup Plans - Add late-game insurance to early comps
2. 1-3-1 Split Push - Practice splitting when can't teamfight
3. Objective Trading - Don't force fights at disadvantageous timings

LONG-TERM:
1. Viego Positioning - Practice safe positioning when ahead
2. High Ground Execution - Drill tower sieges with early comps
3. Mental Reset - After losing key fight, don't force desperation plays"""
            },
        ]
        
        return esports_samples
    
    def prepare_datasets(self, max_train: int = None, add_esports: bool = False):
        """Prepare train, validation, and test datasets."""
        print("=" * 60)
        print("   Data Preparation for ESports Strategy AI")
        print("=" * 60)
        
        # Load CSV files
        print("\n📥 Loading CSV files...")
        train_data = self.load_csv_data("train.csv", max_samples=max_train)
        val_data = self.load_csv_data("validation.csv")
        test_data = self.load_csv_data("test.csv", max_samples=500)  # Limit test set
        
        print(f"   ✓ Train: {len(train_data)} samples")
        print(f"   ✓ Validation: {len(val_data)} samples")
        print(f"   ✓ Test: {len(test_data)} samples")
        
        # Convert to instruction format
        print("\n📝 Converting to instruction format...")
        train_formatted = self.convert_to_instruction_format(train_data)
        val_formatted = self.convert_to_instruction_format(val_data)
        test_formatted = self.convert_to_instruction_format(test_data)
        
        print(f"   ✓ Train: {len(train_formatted)} instruction samples")
        print(f"   ✓ Validation: {len(val_formatted)} instruction samples")
        print(f"   ✓ Test: {len(test_formatted)} instruction samples")
        
        # Add esports samples if requested
        if add_esports:
            print("\n🎮 Adding synthetic esports samples...")
            esports_samples = self.create_esports_samples()
            
            # Add to training data (repeat multiple times for emphasis)
            train_formatted.extend(esports_samples * 10)  # Repeat 10 times
            val_formatted.extend(esports_samples)
            
            print(f"   ✓ Added {len(esports_samples)} unique esports samples (repeated in training)")
        
        # Create HuggingFace datasets
        print("\n📦 Creating datasets...")
        dataset_dict = DatasetDict({
            'train': Dataset.from_list(train_formatted),
            'validation': Dataset.from_list(val_formatted),
            'test': Dataset.from_list(test_formatted)
        })
        
        # Save
        output_path = self.data_dir / "training_data"
        dataset_dict.save_to_disk(output_path)
        
        print(f"   ✓ Saved to {output_path}")
        
        # Save JSON backup
        json_path = self.data_dir / "training_data.json"
        sample_data = {
            'train_sample': train_formatted[0] if train_formatted else {},
            'val_sample': val_formatted[0] if val_formatted else {},
            'stats': {
                'train_size': len(train_formatted),
                'val_size': len(val_formatted),
                'test_size': len(test_formatted),
                'includes_esports': add_esports
            }
        }
        with open(json_path, 'w') as f:
            json.dump(sample_data, f, indent=2)
        
        print(f"   ✓ Sample saved to {json_path}")
        
        print("\n✅ Data preparation complete!")
        print(f"\nDataset Summary:")
        print(f"  Training: {len(dataset_dict['train']):,} samples")
        print(f"  Validation: {len(dataset_dict['validation']):,} samples")
        print(f"  Test: {len(dataset_dict['test']):,} samples")
        
        return dataset_dict


def main():
    parser = argparse.ArgumentParser(description="Prepare training data")
    parser.add_argument("--max-samples", type=int, default=None, 
                       help="Maximum training samples to use (None = all)")
    parser.add_argument("--add-esports", action="store_true",
                       help="Add synthetic esports training samples")
    parser.add_argument("--data-dir", type=str, default="./data",
                       help="Output directory")
    
    args = parser.parse_args()
    
    prep = DataPreparation(data_dir=args.data_dir)
    prep.prepare_datasets(
        max_train=args.max_samples,
        add_esports=args.add_esports
    )


if __name__ == "__main__":
    main()