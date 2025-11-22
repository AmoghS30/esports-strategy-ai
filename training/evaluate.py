"""
Evaluation Script for ESports Strategy AI
Tests model quality using ROUGE scores and custom metrics.

Usage:
    python evaluate.py --model ./models/esports-lora_final
    python evaluate.py --model ./models/esports-lora_final --detailed
"""

import os
import json
import argparse
import torch
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_from_disk
import numpy as np


class ModelEvaluator:
    """Evaluates trained model quality."""
    
    def __init__(self, model_path: str, base_model: str = None):
        self.model_path = Path(model_path)
        self.model = None
        self.tokenizer = None
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Detect base model from config
        config_path = self.model_path / "training_config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
                base_model = config.get("model_name", base_model)
        
        self.base_model = base_model or "google/gemma-2b-it"
        
    def load_model(self):
        """Load the trained model."""
        print(f"📥 Loading model from {self.model_path}")
        
        # Quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        # Load base model
        print(f"   Loading base: {self.base_model}")
        base = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load LoRA
        print(f"   Loading LoRA adapters")
        self.model = PeftModel.from_pretrained(base, str(self.model_path))
        self.model.eval()
        
        print("✅ Model loaded!")
    
    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate response from prompt."""
        formatted = f"""### Instruction:
{prompt}

### Response:
"""
        
        inputs = self.tokenizer(formatted, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "### Response:" in response:
            response = response.split("### Response:")[-1].strip()
        
        return response
    
    def evaluate_rouge(self, generated: str, reference: str) -> Dict[str, float]:
        """Calculate ROUGE scores."""
        scores = self.scorer.score(reference, generated)
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure,
        }
    
    def evaluate_dataset(self, data_path: str, max_samples: int = 50) -> Dict:
        """Evaluate on dataset."""
        print(f"\n📊 Evaluating on {data_path}")
        
        try:
            dataset = load_from_disk(data_path)
            eval_data = dataset['validation']
        except:
            print("   ❌ Could not load dataset")
            return {}
        
        results = []
        
        for i, sample in enumerate(tqdm(eval_data[:max_samples], desc="Evaluating")):
            # Generate
            prompt = sample['instruction']
            if sample.get('input'):
                prompt += f"\n\n{sample['input']}"
            
            generated = self.generate(prompt)
            reference = sample['output']
            
            # Score
            scores = self.evaluate_rouge(generated, reference)
            scores['sample_id'] = i
            results.append(scores)
        
        # Aggregate
        avg_scores = {
            'avg_rouge1': np.mean([r['rouge1'] for r in results]),
            'avg_rouge2': np.mean([r['rouge2'] for r in results]),
            'avg_rougeL': np.mean([r['rougeL'] for r in results]),
            'num_samples': len(results),
        }
        
        return avg_scores
    
    def run_test_cases(self) -> List[Dict]:
        """Run predefined test cases."""
        print("\n🧪 Running test cases...")
        
        test_cases = [
            {
                "name": "Match Summary",
                "prompt": "Summarize a League of Legends match where the blue team came back from a 5k gold deficit to win through superior teamfighting at Baron.",
            },
            {
                "name": "Turning Points",
                "prompt": "Identify the key turning points when a team loses despite having an early game lead.",
            },
            {
                "name": "Recommendations",
                "prompt": "What strategic recommendations would you give to a team that keeps losing late-game teamfights despite having good compositions?",
            },
            {
                "name": "Draft Analysis",
                "prompt": "Analyze a team composition of Gnar, Lee Sin, Azir, Jinx, Thresh. What are the win conditions?",
            },
        ]
        
        results = []
        
        for tc in test_cases:
            print(f"\n   Testing: {tc['name']}")
            response = self.generate(tc['prompt'])
            
            results.append({
                "name": tc['name'],
                "prompt": tc['prompt'],
                "response": response,
                "response_length": len(response),
            })
            
            print(f"   Response length: {len(response)} chars")
            print(f"   Preview: {response[:200]}...")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--data", type=str, default="./data/combined_training", help="Evaluation data path")
    parser.add_argument("--samples", type=int, default=50, help="Number of samples to evaluate")
    parser.add_argument("--detailed", action="store_true", help="Show detailed test results")
    parser.add_argument("--output", type=str, default="./evaluation_results.json", help="Output file")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("   ESports Strategy AI - Model Evaluation")
    print("=" * 60)
    
    # Initialize
    evaluator = ModelEvaluator(args.model)
    evaluator.load_model()
    
    # Run evaluation
    all_results = {}
    
    # ROUGE evaluation
    rouge_results = evaluator.evaluate_dataset(args.data, max_samples=args.samples)
    all_results['rouge_scores'] = rouge_results
    
    print("\n📊 ROUGE Scores:")
    print(f"   ROUGE-1: {rouge_results.get('avg_rouge1', 0):.4f}")
    print(f"   ROUGE-2: {rouge_results.get('avg_rouge2', 0):.4f}")
    print(f"   ROUGE-L: {rouge_results.get('avg_rougeL', 0):.4f}")
    
    # Test cases
    test_results = evaluator.run_test_cases()
    all_results['test_cases'] = test_results
    
    if args.detailed:
        print("\n" + "=" * 60)
        print("   Detailed Test Results")
        print("=" * 60)
        for tc in test_results:
            print(f"\n### {tc['name']}")
            print(f"Prompt: {tc['prompt']}")
            print(f"\nResponse:\n{tc['response']}")
            print("-" * 40)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n💾 Results saved to {args.output}")
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
