"""
Enhanced ESports Strategy Analyzer - Flask Backend
Supports Gemma-2B, Falcon-7B and other open LLMs
Includes advanced prompt engineering and evaluation metrics

Features:
- Multiple prompt templates (turning points, tactics, focus areas)
- Different summary styles (analytical, narrative, bullet, tactical)
- Evaluation metrics (ROUGE, human evaluation framework)
- Batch processing capabilities
"""

import os
import json
import torch
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging
from collections import defaultdict

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Try to import evaluation libraries
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("⚠️  rouge-score not available. Install with: pip install rouge-score")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*", "methods": ["GET", "POST", "OPTIONS"], "allow_headers": ["Content-Type"]}})

# ============================================================
# Configuration
# ============================================================

class Config:
    """Application configuration"""
    MODEL_DIR = os.getenv("MODEL_DIR", "./models/esports-llm_final")
    MAX_LENGTH = int(os.getenv("MAX_LENGTH", "1024"))
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
    TOP_P = float(os.getenv("TOP_P", "0.9"))
    TOP_K = int(os.getenv("TOP_K", "50"))
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Prompt templates
    SUMMARY_STYLES = {
        "analytical": {
            "description": "Detailed analytical breakdown with statistics and reasoning",
            "instruction": "Provide a comprehensive analytical breakdown of the match, including key statistics, decision points, and strategic reasoning. Focus on cause-and-effect relationships."
        },
        "narrative": {
            "description": "Engaging story-driven match summary",
            "instruction": "Tell the story of this match in an engaging narrative format, capturing the drama, momentum shifts, and emotional arc of the game."
        },
        "bullet": {
            "description": "Concise bullet-point summary",
            "instruction": "Provide a concise bullet-point summary of the match, highlighting only the most important events and outcomes."
        },
        "tactical": {
            "description": "Focus on tactical decisions and strategic implications",
            "instruction": "Analyze the tactical decisions made by both teams, explaining the strategic implications of key choices and their impact on future matches."
        }
    }
    
    # Focus areas for analysis
    FOCUS_AREAS = {
        "team_fights": {
            "description": "Combat engagements and team fighting",
            "keywords": ["positioning", "engagement", "target priority", "cooldown management", "fight execution"]
        },
        "economy": {
            "description": "Resource management and economic decisions",
            "keywords": ["gold efficiency", "farming patterns", "item timings", "resource allocation", "economic advantage"]
        },
        "objectives": {
            "description": "Objective control and map pressure",
            "keywords": ["baron", "dragon", "towers", "map control", "objective priorities", "vision control"]
        },
        "composition": {
            "description": "Team composition and draft strategy",
            "keywords": ["draft phase", "champion synergy", "power spikes", "win conditions", "composition analysis"]
        },
        "macro": {
            "description": "Macro-level strategy and rotations",
            "keywords": ["rotation timing", "wave management", "split push", "map pressure", "strategic decision-making"]
        }
    }
    
    # Evaluation criteria
    EVALUATION_CRITERIA = {
        "turning_point_coverage": "Does the summary identify the key moments that changed the match?",
        "tactical_accuracy": "Are the tactical recommendations realistic and actionable?",
        "clarity": "Is the summary clear and easy to understand?",
        "completeness": "Does the summary cover all important aspects of the match?",
        "actionability": "Are the recommendations specific enough to implement?"
    }

config = Config()

# ============================================================
# Model Management
# ============================================================

class ModelManager:
    """Enhanced model manager with evaluation capabilities"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = config.DEVICE
        self.loaded = False
        self.generation_stats = defaultdict(list)
        
    def load_model(self):
        """Load the fine-tuned model"""
        try:
            logger.info(f"Loading model from {config.MODEL_DIR}")
            logger.info(f"Using device: {self.device}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.MODEL_DIR,
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            # Load model
            if self.device == "cuda":
                self.model = AutoModelForCausalLM.from_pretrained(
                    config.MODEL_DIR,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    low_cpu_mem_usage=True
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    config.MODEL_DIR,
                    trust_remote_code=True,
                    torch_dtype=torch.float32,
                    device_map="cpu"
                )
            
            self.model.eval()
            self.loaded = True
            
            logger.info("✅ Model loaded successfully")
            logger.info(f"   Device: {self.device}")
            logger.info(f"   Parameters: {self.model.num_parameters():,}")
            
            if self.device == "cuda":
                allocated = torch.cuda.memory_allocated(0) / 1e9
                logger.info(f"   GPU Memory: {allocated:.2f} GB")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            return False
    
    def generate(self, prompt: str, max_length: int = None,
                 temperature: float = None, top_p: float = None,
                 top_k: int = None, track_stats: bool = True) -> Dict:
        """Generate text with optional statistics tracking"""
        if not self.loaded:
            raise RuntimeError("Model not loaded")
        
        # Use config defaults
        max_length = max_length or config.MAX_LENGTH
        temperature = temperature or config.TEMPERATURE
        top_p = top_p or config.TOP_P
        top_k = top_k or config.TOP_K
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length
            ).to(self.device)
            
            # Track generation time
            import time
            start_time = time.time()
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    return_dict_in_generate=True,
                    output_scores=True
                )
            
            generation_time = time.time() - start_time
            
            # Decode
            generated_text = self.tokenizer.decode(
                outputs.sequences[0],
                skip_special_tokens=True
            )
            
            # Extract only generated part
            if prompt in generated_text:
                generated_text = generated_text.split(prompt)[-1].strip()
            
            # Calculate statistics
            num_tokens = len(outputs.sequences[0]) - len(inputs.input_ids[0])
            tokens_per_second = num_tokens / generation_time if generation_time > 0 else 0
            
            # Track stats
            if track_stats:
                self.generation_stats['generation_time'].append(generation_time)
                self.generation_stats['num_tokens'].append(num_tokens)
                self.generation_stats['tokens_per_second'].append(tokens_per_second)
            
            result = {
                'text': generated_text,
                'stats': {
                    'generation_time': generation_time,
                    'num_tokens': num_tokens,
                    'tokens_per_second': tokens_per_second,
                    'prompt_length': len(inputs.input_ids[0])
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise
    
    def get_stats_summary(self) -> Dict:
        """Get summary of generation statistics"""
        if not self.generation_stats['generation_time']:
            return {"message": "No generations yet"}
        
        return {
            'total_generations': len(self.generation_stats['generation_time']),
            'avg_generation_time': np.mean(self.generation_stats['generation_time']),
            'avg_tokens': np.mean(self.generation_stats['num_tokens']),
            'avg_tokens_per_second': np.mean(self.generation_stats['tokens_per_second']),
            'total_time': sum(self.generation_stats['generation_time'])
        }

# Initialize model manager
model_manager = ModelManager()

# ============================================================
# Enhanced Prompt Engineering
# ============================================================

class EnhancedPromptBuilder:
    """Advanced prompt builder with multiple template strategies"""
    
    @staticmethod
    def build_match_context(match_data: Dict, max_events: int = 15) -> str:
        """Build comprehensive match context from data"""
        context = f"""Match Information:
Team A: {match_data.get('team_a', 'Unknown')}
Team B: {match_data.get('team_b', 'Unknown')}
Winner: {match_data.get('winner', 'Unknown')}
Duration: {match_data.get('duration', 'Unknown')}

"""
        
        # Team compositions
        if 'team_a_composition' in match_data:
            context += f"Team A Composition: {match_data['team_a_composition']}\n"
        if 'team_b_composition' in match_data:
            context += f"Team B Composition: {match_data['team_b_composition']}\n"
        
        # Key events
        if 'events' in match_data and match_data['events']:
            context += "\nKey Match Events:\n"
            for event in match_data['events'][:max_events]:
                context += f"- {event}\n"
        
        # Statistics
        if 'statistics' in match_data:
            context += "\nMatch Statistics:\n"
            for key, value in match_data['statistics'].items():
                context += f"- {key}: {value}\n"
        
        # Commentary excerpts
        if 'commentary' in match_data:
            commentary_text = match_data['commentary'][:800]
            context += f"\nCommentary Excerpt:\n{commentary_text}\n"
        
        return context
    
    @staticmethod
    def build_summary_prompt(match_data: Dict, style: str = "analytical",
                            focus: str = "team_fights", length: str = "medium") -> str:
        """Build customized summary prompt"""
        context = EnhancedPromptBuilder.build_match_context(match_data)
        
        # Get style instructions
        style_config = config.SUMMARY_STYLES.get(style, config.SUMMARY_STYLES["analytical"])
        style_instruction = style_config["instruction"]
        
        # Get focus area
        focus_config = config.FOCUS_AREAS.get(focus, config.FOCUS_AREAS["team_fights"])
        focus_keywords = ", ".join(focus_config["keywords"])
        
        # Length guidance
        length_guide = {
            "short": "Keep it concise (2-3 sentences)",
            "medium": "Provide a moderate summary (2-3 paragraphs)",
            "long": "Provide a comprehensive analysis (4-5 paragraphs)"
        }
        
        prompt = f"""{context}

Task: {style_instruction}

Focus particularly on: {focus_config["description"]} - pay attention to {focus_keywords}.

{length_guide.get(length, length_guide['medium'])}

Analysis:"""
        
        return prompt
    
    @staticmethod
    def build_turning_points_prompt(match_data: Dict, num_points: int = 3) -> str:
        """Build prompt for identifying turning points"""
        context = EnhancedPromptBuilder.build_match_context(match_data)
        
        prompt = f"""{context}

Task: Identify the {num_points} most critical turning points that determined the match outcome.

For each turning point, explain:
1. Timestamp and what happened (specific event)
2. Why it was significant (impact on game state, gold swing, momentum shift)
3. How it influenced the final outcome
4. What the losing team could have done differently

Turning Points:"""
        
        return prompt
    
    @staticmethod
    def build_tactical_recommendations_prompt(
        match_data: Dict,
        team: str = "team_a",
        num_recommendations: int = 3,
        focus_areas: List[str] = None
    ) -> str:
        """Build prompt for tactical recommendations"""
        context = EnhancedPromptBuilder.build_match_context(match_data)
        
        team_name = match_data.get(team, 'Unknown Team')
        
        if focus_areas:
            focus_text = f"Focus specifically on: {', '.join(focus_areas)}"
        else:
            focus_text = "Consider all aspects: draft, objectives, positioning, and timing"
        
        prompt = f"""{context}

Task: As an expert esports coach, provide {num_recommendations} specific, actionable tactical recommendations for {team_name} to improve in their next match.

{focus_text}

For each recommendation:
1. Identify the specific issue or missed opportunity
2. Explain the recommended change or adjustment
3. Describe the expected impact on match outcomes
4. Provide concrete steps to implement the change

Recommendations:"""
        
        return prompt
    
    @staticmethod
    def build_comparative_analysis_prompt(match_data: Dict) -> str:
        """Build prompt for comparing team performances"""
        context = EnhancedPromptBuilder.build_match_context(match_data)
        
        prompt = f"""{context}

Task: Provide a comparative analysis of both teams' performances.

Compare:
1. Draft strategy and team composition effectiveness
2. Early, mid, and late game execution
3. Objective prioritization and timing
4. Team fighting and positioning
5. Decision-making quality under pressure

For each aspect, identify which team performed better and why.

Comparative Analysis:"""
        
        return prompt


# ============================================================
# Evaluation Framework
# ============================================================

class EvaluationFramework:
    """Framework for evaluating generated summaries and recommendations"""
    
    def __init__(self):
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'],
                use_stemmer=True
            )
        else:
            self.rouge_scorer = None
    
    def calculate_rouge(self, generated: str, reference: str) -> Dict:
        """Calculate ROUGE scores"""
        if not self.rouge_scorer:
            return {"error": "ROUGE not available"}
        
        scores = self.rouge_scorer.score(reference, generated)
        
        return {
            'rouge1': {
                'precision': scores['rouge1'].precision,
                'recall': scores['rouge1'].recall,
                'fmeasure': scores['rouge1'].fmeasure
            },
            'rouge2': {
                'precision': scores['rouge2'].precision,
                'recall': scores['rouge2'].recall,
                'fmeasure': scores['rouge2'].fmeasure
            },
            'rougeL': {
                'precision': scores['rougeL'].precision,
                'recall': scores['rougeL'].recall,
                'fmeasure': scores['rougeL'].fmeasure
            }
        }
    
    def evaluate_turning_point_coverage(self, generated: str, key_events: List[str]) -> Dict:
        """Evaluate if generated text covers key turning points"""
        generated_lower = generated.lower()
        
        covered_events = []
        missed_events = []
        
        for event in key_events:
            # Check if event keywords appear in generated text
            event_keywords = event.lower().split()
            if any(keyword in generated_lower for keyword in event_keywords):
                covered_events.append(event)
            else:
                missed_events.append(event)
        
        coverage_ratio = len(covered_events) / len(key_events) if key_events else 0
        
        return {
            'coverage_ratio': coverage_ratio,
            'covered_events': covered_events,
            'missed_events': missed_events,
            'total_events': len(key_events)
        }
    
    def evaluate_recommendation_quality(self, recommendation: str) -> Dict:
        """Heuristic evaluation of recommendation quality"""
        # Check for specificity indicators
        specificity_keywords = [
            'specific', 'particular', 'exactly', 'precisely',
            'should', 'must', 'need to', 'recommend'
        ]
        
        # Check for actionability indicators
        actionability_keywords = [
            'practice', 'train', 'adjust', 'change', 'improve',
            'focus on', 'prioritize', 'avoid', 'implement'
        ]
        
        # Check for measurability
        measurability_keywords = [
            'increase', 'decrease', 'reduce', 'within',
            'by', 'percent', 'seconds', 'minutes'
        ]
        
        rec_lower = recommendation.lower()
        
        specificity_score = sum(1 for kw in specificity_keywords if kw in rec_lower)
        actionability_score = sum(1 for kw in actionability_keywords if kw in rec_lower)
        measurability_score = sum(1 for kw in measurability_keywords if kw in rec_lower)
        
        # Normalize scores
        total_keywords = len(specificity_keywords + actionability_keywords + measurability_keywords)
        total_found = specificity_score + actionability_score + measurability_score
        
        return {
            'specificity_score': specificity_score,
            'actionability_score': actionability_score,
            'measurability_score': measurability_score,
            'overall_quality': total_found / total_keywords if total_keywords > 0 else 0,
            'has_specificity': specificity_score > 0,
            'has_actionability': actionability_score > 0,
            'has_measurability': measurability_score > 0
        }
    
    def create_human_evaluation_template(self, generated: str, match_data: Dict) -> Dict:
        """Create template for human evaluation"""
        return {
            'generated_summary': generated,
            'match_info': {
                'teams': f"{match_data.get('team_a')} vs {match_data.get('team_b')}",
                'winner': match_data.get('winner'),
                'duration': match_data.get('duration')
            },
            'evaluation_criteria': config.EVALUATION_CRITERIA,
            'rating_instructions': "Rate each criterion on a scale of 1-5",
            'ratings': {criterion: None for criterion in config.EVALUATION_CRITERIA.keys()},
            'comments': "",
            'evaluator': "",
            'timestamp': datetime.now().isoformat()
        }

# Initialize evaluation framework
evaluator = EvaluationFramework()

# ============================================================
# API Endpoints
# ============================================================

@app.route('/')
def index():
    """Health check endpoint"""
    return jsonify({
        "service": "Enhanced ESports Strategy Analyzer API",
        "version": "2.0",
        "status": "running",
        "model_loaded": model_manager.loaded,
        "device": config.DEVICE,
        "rouge_available": ROUGE_AVAILABLE,
        "features": [
            "match_summarization",
            "turning_point_identification",
            "tactical_recommendations",
            "comparative_analysis",
            "evaluation_metrics"
        ]
    })

@app.route('/health')
def health():
    """Detailed health check"""
    health_status = {
        "status": "healthy" if model_manager.loaded else "unhealthy",
        "model_loaded": model_manager.loaded,
        "device": config.DEVICE,
        "rouge_available": ROUGE_AVAILABLE,
        "timestamp": datetime.now().isoformat()
    }
    
    if config.DEVICE == "cuda":
        health_status["gpu_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            health_status["gpu_name"] = torch.cuda.get_device_name(0)
            health_status["gpu_memory_allocated"] = f"{torch.cuda.memory_allocated(0) / 1e9:.2f} GB"
    
    # Add generation stats
    health_status["generation_stats"] = model_manager.get_stats_summary()
    
    return jsonify(health_status)

@app.route('/api/analyze', methods=['POST', 'OPTIONS'])
def analyze_complete():
    """
    Complete analysis endpoint - combines summary, turning points, and recommendations
    
    Request body:
    {
        "match_data": {...},
        "style": "analytical",
        "focus": "team_fights",
        "team": "team_a",
        "recommendation_depth": 3
    }
    """
    if request.method == 'OPTIONS':
        return '', 200
        
    try:
        if not model_manager.loaded:
            return jsonify({"error": "Model not loaded"}), 503
        
        data = request.get_json()
        
        if not data or 'match_data' not in data:
            return jsonify({"error": "Missing match_data in request"}), 400
        
        match_data = data['match_data']
        style = data.get('style', 'analytical')
        focus = data.get('focus', 'team_fights')
        team = data.get('team', 'team_a')
        recommendation_depth = data.get('recommendation_depth', 3)
        temperature = data.get('temperature', config.TEMPERATURE)
        
        logger.info(f"Complete analysis requested for {match_data.get('team_a')} vs {match_data.get('team_b')}")
        
        # Generate summary
        summary_prompt = EnhancedPromptBuilder.build_summary_prompt(
            match_data=match_data,
            style=style,
            focus=focus,
            length='medium'
        )
        summary_result = model_manager.generate(prompt=summary_prompt, temperature=temperature)
        
        # Generate turning points
        turning_points_prompt = EnhancedPromptBuilder.build_turning_points_prompt(match_data, num_points=3)
        turning_points_result = model_manager.generate(prompt=turning_points_prompt, temperature=temperature)
        
        # Generate recommendations
        recommendations_prompt = EnhancedPromptBuilder.build_tactical_recommendations_prompt(
            match_data=match_data,
            team=team,
            num_recommendations=recommendation_depth
        )
        recommendations_result = model_manager.generate(prompt=recommendations_prompt, temperature=temperature)
        
        return jsonify({
            "success": True,
            "analysis": {
                "summary": summary_result['text'],
                "turning_points": turning_points_result['text'],
                "recommendations": recommendations_result['text']
            },
            "metadata": {
                "match": {
                    "team_a": match_data.get('team_a'),
                    "team_b": match_data.get('team_b'),
                    "winner": match_data.get('winner'),
                    "duration": match_data.get('duration')
                },
                "style": style,
                "focus": focus,
                "team": team,
                "timestamp": datetime.now().isoformat()
            },
            "stats": {
                "summary": summary_result['stats'],
                "turning_points": turning_points_result['stats'],
                "recommendations": recommendations_result['stats']
            }
        })
        
    except Exception as e:
        logger.error(f"Error in analyze_complete: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/summarize', methods=['POST', 'OPTIONS'])
def summarize_match():
    """
    Generate match summary with customizable style and focus
    
    Request body:
    {
        "match_data": {...},
        "style": "analytical|narrative|bullet|tactical",
        "focus": "team_fights|economy|objectives|composition|macro",
        "length": "short|medium|long",
        "temperature": 0.7,
        "max_length": 1024,
        "include_evaluation": false
    }
    """
    if request.method == 'OPTIONS':
        return '', 200
        
    try:
        if not model_manager.loaded:
            return jsonify({"error": "Model not loaded"}), 503
        
        data = request.get_json()
        
        if not data or 'match_data' not in data:
            return jsonify({"error": "Missing match_data in request"}), 400
        
        match_data = data['match_data']
        style = data.get('style', 'analytical')
        focus = data.get('focus', 'team_fights')
        length = data.get('length', 'medium')
        temperature = data.get('temperature', config.TEMPERATURE)
        max_length = data.get('max_length', config.MAX_LENGTH)
        include_evaluation = data.get('include_evaluation', False)
        
        # Build prompt
        prompt = EnhancedPromptBuilder.build_summary_prompt(
            match_data=match_data,
            style=style,
            focus=focus,
            length=length
        )
        
        logger.info(f"Generating summary (style={style}, focus={focus}, length={length})")
        
        # Generate summary
        result = model_manager.generate(
            prompt=prompt,
            max_length=max_length,
            temperature=temperature
        )
        
        response = {
            "success": True,
            "summary": result['text'],
            "metadata": {
                "style": style,
                "focus": focus,
                "length": length,
                "temperature": temperature,
                "timestamp": datetime.now().isoformat()
            },
            "stats": result['stats']
        }
        
        # Add evaluation if requested
        if include_evaluation and 'key_events' in match_data:
            evaluation = evaluator.evaluate_turning_point_coverage(
                result['text'],
                match_data['key_events']
            )
            response["evaluation"] = evaluation
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in summarize_match: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/turning-points', methods=['POST', 'OPTIONS'])
def identify_turning_points():
    """
    Identify key turning points in a match
    
    Request body:
    {
        "match_data": {...},
        "num_points": 3,
        "temperature": 0.7
    }
    """
    if request.method == 'OPTIONS':
        return '', 200
        
    try:
        if not model_manager.loaded:
            return jsonify({"error": "Model not loaded"}), 503
        
        data = request.get_json()
        
        if not data or 'match_data' not in data:
            return jsonify({"error": "Missing match_data in request"}), 400
        
        match_data = data['match_data']
        num_points = data.get('num_points', 3)
        temperature = data.get('temperature', config.TEMPERATURE)
        
        # Build prompt
        prompt = EnhancedPromptBuilder.build_turning_points_prompt(
            match_data=match_data,
            num_points=num_points
        )
        
        logger.info(f"Identifying {num_points} turning points")
        
        # Generate analysis
        result = model_manager.generate(
            prompt=prompt,
            temperature=temperature
        )
        
        return jsonify({
            "success": True,
            "turning_points": result['text'],
            "metadata": {
                "num_points": num_points,
                "timestamp": datetime.now().isoformat()
            },
            "stats": result['stats']
        })
        
    except Exception as e:
        logger.error(f"Error in identify_turning_points: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/recommendations', methods=['POST', 'OPTIONS'])
def get_recommendations():
    """
    Generate tactical recommendations for a team
    
    Request body:
    {
        "match_data": {...},
        "team": "team_a|team_b",
        "num_recommendations": 3,
        "focus_areas": ["draft", "objectives"],
        "temperature": 0.7,
        "evaluate_quality": false
    }
    """
    if request.method == 'OPTIONS':
        return '', 200
        
    try:
        if not model_manager.loaded:
            return jsonify({"error": "Model not loaded"}), 503
        
        data = request.get_json()
        
        if not data or 'match_data' not in data:
            return jsonify({"error": "Missing match_data in request"}), 400
        
        match_data = data['match_data']
        team = data.get('team', 'team_a')
        num_recommendations = data.get('num_recommendations', 3)
        recommendation_depth = data.get('recommendation_depth', num_recommendations)
        focus_areas = data.get('focus_areas', None)
        temperature = data.get('temperature', config.TEMPERATURE)
        evaluate_quality = data.get('evaluate_quality', False)
        
        # Build prompt
        prompt = EnhancedPromptBuilder.build_tactical_recommendations_prompt(
            match_data=match_data,
            team=team,
            num_recommendations=recommendation_depth,
            focus_areas=focus_areas
        )
        
        logger.info(f"Generating {recommendation_depth} recommendations for {team}")
        
        # Generate recommendations
        result = model_manager.generate(
            prompt=prompt,
            temperature=temperature
        )
        
        response = {
            "success": True,
            "recommendations": result['text'],
            "metadata": {
                "team": team,
                "num_recommendations": recommendation_depth,
                "focus_areas": focus_areas,
                "timestamp": datetime.now().isoformat()
            },
            "stats": result['stats']
        }
        
        # Evaluate quality if requested
        if evaluate_quality:
            quality_eval = evaluator.evaluate_recommendation_quality(result['text'])
            response["quality_evaluation"] = quality_eval
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in get_recommendations: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/compare', methods=['POST', 'OPTIONS'])
def compare_teams():
    """
    Generate comparative analysis of team performances
    
    Request body:
    {
        "match_data": {...},
        "temperature": 0.7
    }
    """
    if request.method == 'OPTIONS':
        return '', 200
        
    try:
        if not model_manager.loaded:
            return jsonify({"error": "Model not loaded"}), 503
        
        data = request.get_json()
        
        if not data or 'match_data' not in data:
            return jsonify({"error": "Missing match_data in request"}), 400
        
        match_data = data['match_data']
        temperature = data.get('temperature', config.TEMPERATURE)
        
        # Build prompt
        prompt = EnhancedPromptBuilder.build_comparative_analysis_prompt(match_data)
        
        logger.info("Generating comparative analysis")
        
        # Generate analysis
        result = model_manager.generate(
            prompt=prompt,
            temperature=temperature
        )
        
        return jsonify({
            "success": True,
            "analysis": result['text'],
            "metadata": {
                "teams": f"{match_data.get('team_a')} vs {match_data.get('team_b')}",
                "timestamp": datetime.now().isoformat()
            },
            "stats": result['stats']
        })
        
    except Exception as e:
        logger.error(f"Error in compare_teams: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/evaluate', methods=['POST', 'OPTIONS'])
def evaluate_summary():
    """
    Evaluate a generated summary
    
    Request body:
    {
        "generated_text": "...",
        "reference_text": "...",  # optional
        "match_data": {...},
        "evaluation_type": "rouge|coverage|quality|human_template"
    }
    """
    if request.method == 'OPTIONS':
        return '', 200
        
    try:
        data = request.get_json()
        
        if not data or 'generated_text' not in data:
            return jsonify({"error": "Missing generated_text in request"}), 400
        
        generated = data['generated_text']
        evaluation_type = data.get('evaluation_type', 'coverage')
        
        results = {}
        
        # ROUGE evaluation
        if evaluation_type == 'rouge' and 'reference_text' in data:
            if not ROUGE_AVAILABLE:
                return jsonify({"error": "ROUGE not available"}), 400
            results['rouge'] = evaluator.calculate_rouge(generated, data['reference_text'])
        
        # Coverage evaluation
        elif evaluation_type == 'coverage' and 'match_data' in data:
            match_data = data['match_data']
            if 'key_events' in match_data:
                results['coverage'] = evaluator.evaluate_turning_point_coverage(
                    generated, match_data['key_events']
                )
        
        # Quality evaluation
        elif evaluation_type == 'quality':
            results['quality'] = evaluator.evaluate_recommendation_quality(generated)
        
        # Human evaluation template
        elif evaluation_type == 'human_template':
            if 'match_data' not in data:
                return jsonify({"error": "Missing match_data for human template"}), 400
            results['human_template'] = evaluator.create_human_evaluation_template(
                generated, data['match_data']
            )
        
        else:
            return jsonify({"error": f"Unknown evaluation type: {evaluation_type}"}), 400
        
        return jsonify({
            "success": True,
            "evaluation": results,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in evaluate_summary: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/batch', methods=['POST', 'OPTIONS'])
def batch_process():
    """
    Process multiple matches in batch
    
    Request body:
    {
        "matches": [
            {"match_data": {...}, "analysis_type": "summary|turning_points|recommendations"},
            ...
        ],
        "style": "analytical",
        "temperature": 0.7
    }
    """
    if request.method == 'OPTIONS':
        return '', 200
        
    try:
        if not model_manager.loaded:
            return jsonify({"error": "Model not loaded"}), 503
        
        data = request.get_json()
        
        if not data or 'matches' not in data:
            return jsonify({"error": "Missing matches in request"}), 400
        
        matches = data['matches']
        style = data.get('style', 'analytical')
        temperature = data.get('temperature', config.TEMPERATURE)
        
        logger.info(f"Processing {len(matches)} matches in batch")
        
        results = []
        
        for i, match_config in enumerate(matches):
            try:
                match_data = match_config['match_data']
                analysis_type = match_config.get('analysis_type', 'summary')
                
                # Build appropriate prompt
                if analysis_type == 'summary':
                    prompt = EnhancedPromptBuilder.build_summary_prompt(
                        match_data=match_data,
                        style=style
                    )
                elif analysis_type == 'turning_points':
                    prompt = EnhancedPromptBuilder.build_turning_points_prompt(match_data)
                elif analysis_type == 'recommendations':
                    prompt = EnhancedPromptBuilder.build_tactical_recommendations_prompt(
                        match_data=match_data
                    )
                else:
                    results.append({"error": f"Unknown analysis type: {analysis_type}"})
                    continue
                
                # Generate
                result = model_manager.generate(prompt=prompt, temperature=temperature)
                
                results.append({
                    "match_index": i,
                    "teams": f"{match_data.get('team_a')} vs {match_data.get('team_b')}",
                    "analysis_type": analysis_type,
                    "result": result['text'],
                    "stats": result['stats']
                })
                
            except Exception as e:
                logger.error(f"Error processing match {i}: {e}")
                results.append({
                    "match_index": i,
                    "error": str(e)
                })
        
        return jsonify({
            "success": True,
            "total_matches": len(matches),
            "processed": len(results),
            "results": results,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in batch_process: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/config', methods=['GET', 'OPTIONS'])
def get_config():
    """Get available configuration options"""
    if request.method == 'OPTIONS':
        return '', 200
        
    return jsonify({
        "summary_styles": {
            k: v["description"] for k, v in config.SUMMARY_STYLES.items()
        },
        "focus_areas": {
            k: v["description"] for k, v in config.FOCUS_AREAS.items()
        },
        "length_options": ["short", "medium", "long"],
        "evaluation_criteria": config.EVALUATION_CRITERIA,
        "default_temperature": config.TEMPERATURE,
        "default_max_length": config.MAX_LENGTH,
        "device": config.DEVICE,
        "rouge_available": ROUGE_AVAILABLE
    })

@app.route('/api/stats', methods=['GET', 'OPTIONS'])
def get_stats():
    """Get generation statistics"""
    if request.method == 'OPTIONS':
        return '', 200
        
    return jsonify({
        "success": True,
        "stats": model_manager.get_stats_summary(),
        "timestamp": datetime.now().isoformat()
    })

# ============================================================
# Startup
# ============================================================

def main():
    """Main entry point"""
    print("=" * 60)
    print("   Enhanced ESports Strategy Analyzer - Backend API")
    print("=" * 60)
    
    # Load model
    print("\n📦 Loading model...")
    success = model_manager.load_model()
    
    if not success:
        print("❌ Failed to load model. Please check MODEL_DIR configuration.")
        print(f"   Current MODEL_DIR: {config.MODEL_DIR}")
        return
    
    # Start server
    print("\n" + "=" * 60)
    print("   🚀 Starting Flask Server")
    print("=" * 60)
    print(f"\n   Model: {config.MODEL_DIR}")
    print(f"   Device: {config.DEVICE}")
    print(f"   ROUGE Available: {ROUGE_AVAILABLE}")
    print(f"   Port: 8080")
    print("\n   API Endpoints:")
    print("   - POST /api/analyze          - Complete analysis")
    print("   - POST /api/summarize        - Generate match summary")
    print("   - POST /api/turning-points   - Identify turning points")
    print("   - POST /api/recommendations  - Get tactical recommendations")
    print("   - POST /api/compare          - Compare team performances")
    print("   - POST /api/evaluate         - Evaluate generated text")
    print("   - POST /api/batch            - Batch process matches")
    print("   - GET  /api/config           - Get configuration options")
    print("   - GET  /api/stats            - Get generation statistics")
    print("   - GET  /health               - Health check")
    print("\n" + "=" * 60)
    print("   ✅ Ready to serve requests!")
    print("=" * 60 + "\n")
    
    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=8080,
        debug=False,
        threaded=True
    )

if __name__ == '__main__':
    main()