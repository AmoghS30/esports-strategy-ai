"""
ESports Strategy Summarizer - Flask Backend
Handles match analysis, summarization, and tactical recommendations
"""

import os
import json
import torch
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import logging

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# ============================================================
# Configuration
# ============================================================

class Config:
    """Application configuration"""
    MODEL_DIR = os.getenv("MODEL_DIR", "./models/esports-gpu-fast_final")
    MAX_LENGTH = int(os.getenv("MAX_LENGTH", "512"))
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
    TOP_P = float(os.getenv("TOP_P", "0.9"))
    TOP_K = int(os.getenv("TOP_K", "50"))
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Summary styles
    SUMMARY_STYLES = {
        "analytical": "Provide a detailed analytical breakdown",
        "narrative": "Tell the story of the match in an engaging narrative",
        "bullet": "Provide a concise bullet-point summary",
        "tactical": "Focus on tactical decisions and strategic implications"
    }
    
    # Focus areas
    FOCUS_AREAS = {
        "team_fights": "team fights and combat engagements",
        "economy": "economic decisions and resource management",
        "objectives": "objective control and map pressure",
        "composition": "team composition and draft strategy",
        "macro": "macro-level strategy and decision-making"
    }

config = Config()

# ============================================================
# Model Management
# ============================================================

class ModelManager:
    """Manages model loading and inference"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = config.DEVICE
        self.loaded = False
        
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
                # GPU inference
                self.model = AutoModelForCausalLM.from_pretrained(
                    config.MODEL_DIR,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    low_cpu_mem_usage=True
                )
            else:
                # CPU inference
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
                 top_k: int = None) -> str:
        """Generate text from prompt"""
        if not self.loaded:
            raise RuntimeError("Model not loaded")
        
        # Use config defaults if not specified
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
                    repetition_penalty=1.1
                )
            
            # Decode
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part (after the prompt)
            if prompt in generated_text:
                generated_text = generated_text.split(prompt)[-1].strip()
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise

# Initialize model manager
model_manager = ModelManager()

# ============================================================
# Prompt Templates
# ============================================================

class PromptBuilder:
    """Builds prompts for different analysis types"""
    
    @staticmethod
    def build_summary_prompt(
        match_data: Dict,
        style: str = "analytical",
        focus: str = "team_fights",
        length: str = "medium"
    ) -> str:
        """Build match summary prompt"""
        
        style_instruction = config.SUMMARY_STYLES.get(style, config.SUMMARY_STYLES["analytical"])
        focus_instruction = config.FOCUS_AREAS.get(focus, config.FOCUS_AREAS["team_fights"])
        
        length_guide = {
            "short": "Keep it brief (2-3 sentences)",
            "medium": "Provide a moderate summary (1-2 paragraphs)",
            "long": "Provide a detailed analysis (3-4 paragraphs)"
        }
        
        prompt = f"""Analyze the following esports match and provide a summary.

Match Information:
Team A: {match_data.get('team_a', 'Unknown')}
Team B: {match_data.get('team_b', 'Unknown')}
Winner: {match_data.get('winner', 'Unknown')}
Duration: {match_data.get('duration', 'Unknown')}

"""
        
        # Add team compositions if available
        if 'team_a_composition' in match_data:
            prompt += f"Team A Composition: {match_data['team_a_composition']}\n"
        if 'team_b_composition' in match_data:
            prompt += f"Team B Composition: {match_data['team_b_composition']}\n"
        
        prompt += "\n"
        
        # Add match events
        if 'events' in match_data and match_data['events']:
            prompt += "Key Events:\n"
            for event in match_data['events'][:10]:  # Limit to 10 events
                prompt += f"- {event}\n"
            prompt += "\n"
        
        # Add commentary if available
        if 'commentary' in match_data and match_data['commentary']:
            commentary_text = match_data['commentary'][:1000]  # Limit length
            prompt += f"Commentary Excerpt:\n{commentary_text}\n\n"
        
        # Add statistics
        if 'statistics' in match_data:
            prompt += "Statistics:\n"
            stats = match_data['statistics']
            for key, value in stats.items():
                prompt += f"- {key}: {value}\n"
            prompt += "\n"
        
        # Add style and focus instructions
        prompt += f"""Instructions:
{style_instruction} focusing on {focus_instruction}.
{length_guide.get(length, length_guide['medium'])}.

Summarize the match, highlighting turning points and key strategic decisions:"""
        
        return prompt
    
    @staticmethod
    def build_tactics_prompt(
        match_data: Dict,
        team: str = "team_a",
        recommendation_depth: int = 3
    ) -> str:
        """Build tactical recommendations prompt"""
        
        team_name = match_data.get(team, 'Unknown Team')
        opponent_name = match_data.get('team_b' if team == 'team_a' else 'team_a', 'Opponent')
        
        prompt = f"""As an esports strategy analyst, provide tactical recommendations for {team_name}.

Match Context:
Team: {team_name}
Opponent: {opponent_name}
Result: {match_data.get('winner', 'Unknown')}
Duration: {match_data.get('duration', 'Unknown')}

"""
        
        # Add team composition
        comp_key = f"{team}_composition"
        if comp_key in match_data:
            prompt += f"Team Composition: {match_data[comp_key]}\n"
        
        # Add match events
        if 'events' in match_data and match_data['events']:
            prompt += "\nKey Match Events:\n"
            for event in match_data['events'][:10]:
                prompt += f"- {event}\n"
        
        # Add statistics
        if 'statistics' in match_data:
            prompt += "\nMatch Statistics:\n"
            for key, value in match_data['statistics'].items():
                prompt += f"- {key}: {value}\n"
        
        prompt += f"""
Based on this match analysis, provide {recommendation_depth} specific tactical recommendations for {team_name}'s next game. Focus on:
1. Draft/composition adjustments
2. Strategic timing improvements
3. Objective prioritization
4. Team fighting positioning

Recommendations:"""
        
        return prompt
    
    @staticmethod
    def build_turning_points_prompt(match_data: Dict) -> str:
        """Build prompt to identify turning points"""
        
        prompt = f"""Analyze this esports match and identify the key turning points that determined the outcome.

Match Information:
Team A: {match_data.get('team_a', 'Unknown')}
Team B: {match_data.get('team_b', 'Unknown')}
Winner: {match_data.get('winner', 'Unknown')}

"""
        
        if 'events' in match_data and match_data['events']:
            prompt += "Match Events:\n"
            for event in match_data['events']:
                prompt += f"- {event}\n"
            prompt += "\n"
        
        if 'commentary' in match_data:
            prompt += f"Commentary:\n{match_data['commentary'][:1000]}\n\n"
        
        prompt += """Identify and explain the 3 most critical turning points in this match. For each turning point, explain:
- What happened
- Why it was significant
- How it affected the match outcome

Turning Points:"""
        
        return prompt

# ============================================================
# API Endpoints
# ============================================================

@app.route('/')
def index():
    """Health check endpoint"""
    return jsonify({
        "service": "ESports Strategy Summarizer API",
        "version": "1.0",
        "status": "running",
        "model_loaded": model_manager.loaded,
        "device": config.DEVICE
    })

@app.route('/health')
def health():
    """Detailed health check"""
    health_status = {
        "status": "healthy" if model_manager.loaded else "unhealthy",
        "model_loaded": model_manager.loaded,
        "device": config.DEVICE,
        "timestamp": datetime.now().isoformat()
    }
    
    if config.DEVICE == "cuda":
        health_status["gpu_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            health_status["gpu_name"] = torch.cuda.get_device_name(0)
            health_status["gpu_memory_allocated"] = f"{torch.cuda.memory_allocated(0) / 1e9:.2f} GB"
    
    return jsonify(health_status)

@app.route('/api/summarize', methods=['POST'])
def summarize_match():
    """
    Summarize a match with customizable style and focus
    
    Request body:
    {
        "match_data": {
            "team_a": "Team Name",
            "team_b": "Team Name",
            "winner": "Team Name",
            "duration": "45:23",
            "team_a_composition": "Hero1, Hero2, ...",
            "team_b_composition": "Hero1, Hero2, ...",
            "events": ["event1", "event2", ...],
            "commentary": "full commentary text...",
            "statistics": {"stat1": "value1", ...}
        },
        "style": "analytical|narrative|bullet|tactical",
        "focus": "team_fights|economy|objectives|composition|macro",
        "length": "short|medium|long",
        "temperature": 0.7,
        "max_length": 512
    }
    """
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
        
        # Build prompt
        prompt = PromptBuilder.build_summary_prompt(
            match_data=match_data,
            style=style,
            focus=focus,
            length=length
        )
        
        logger.info(f"Generating summary (style={style}, focus={focus}, length={length})")
        
        # Generate summary
        summary = model_manager.generate(
            prompt=prompt,
            max_length=max_length,
            temperature=temperature
        )
        
        return jsonify({
            "success": True,
            "summary": summary,
            "metadata": {
                "style": style,
                "focus": focus,
                "length": length,
                "temperature": temperature,
                "timestamp": datetime.now().isoformat()
            }
        })
        
    except Exception as e:
        logger.error(f"Error in summarize_match: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/recommendations', methods=['POST'])
def get_recommendations():
    """
    Generate tactical recommendations for a team
    
    Request body:
    {
        "match_data": {...},
        "team": "team_a|team_b",
        "recommendation_depth": 3,
        "temperature": 0.7
    }
    """
    try:
        if not model_manager.loaded:
            return jsonify({"error": "Model not loaded"}), 503
        
        data = request.get_json()
        
        if not data or 'match_data' not in data:
            return jsonify({"error": "Missing match_data in request"}), 400
        
        match_data = data['match_data']
        team = data.get('team', 'team_a')
        recommendation_depth = data.get('recommendation_depth', 3)
        temperature = data.get('temperature', config.TEMPERATURE)
        
        # Build prompt
        prompt = PromptBuilder.build_tactics_prompt(
            match_data=match_data,
            team=team,
            recommendation_depth=recommendation_depth
        )
        
        logger.info(f"Generating recommendations for {team}")
        
        # Generate recommendations
        recommendations = model_manager.generate(
            prompt=prompt,
            temperature=temperature
        )
        
        return jsonify({
            "success": True,
            "recommendations": recommendations,
            "metadata": {
                "team": team,
                "recommendation_depth": recommendation_depth,
                "timestamp": datetime.now().isoformat()
            }
        })
        
    except Exception as e:
        logger.error(f"Error in get_recommendations: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/turning-points', methods=['POST'])
def identify_turning_points():
    """
    Identify key turning points in a match
    
    Request body:
    {
        "match_data": {...},
        "temperature": 0.7
    }
    """
    try:
        if not model_manager.loaded:
            return jsonify({"error": "Model not loaded"}), 503
        
        data = request.get_json()
        
        if not data or 'match_data' not in data:
            return jsonify({"error": "Missing match_data in request"}), 400
        
        match_data = data['match_data']
        temperature = data.get('temperature', config.TEMPERATURE)
        
        # Build prompt
        prompt = PromptBuilder.build_turning_points_prompt(match_data)
        
        logger.info("Identifying turning points")
        
        # Generate analysis
        turning_points = model_manager.generate(
            prompt=prompt,
            temperature=temperature
        )
        
        return jsonify({
            "success": True,
            "turning_points": turning_points,
            "metadata": {
                "timestamp": datetime.now().isoformat()
            }
        })
        
    except Exception as e:
        logger.error(f"Error in identify_turning_points: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_match():
    """
    Complete match analysis - summary, recommendations, and turning points
    
    Request body:
    {
        "match_data": {...},
        "style": "analytical",
        "focus": "team_fights",
        "team": "team_a",
        "recommendation_depth": 3
    }
    """
    try:
        if not model_manager.loaded:
            return jsonify({"error": "Model not loaded"}), 503
        
        data = request.get_json()
        
        if not data or 'match_data' not in data:
            return jsonify({"error": "Missing match_data in request"}), 400
        
        match_data = data['match_data']
        
        logger.info("Performing complete match analysis")
        
        # Generate summary
        summary_prompt = PromptBuilder.build_summary_prompt(
            match_data=match_data,
            style=data.get('style', 'analytical'),
            focus=data.get('focus', 'team_fights'),
            length=data.get('length', 'medium')
        )
        summary = model_manager.generate(summary_prompt)
        
        # Generate recommendations
        tactics_prompt = PromptBuilder.build_tactics_prompt(
            match_data=match_data,
            team=data.get('team', 'team_a'),
            recommendation_depth=data.get('recommendation_depth', 3)
        )
        recommendations = model_manager.generate(tactics_prompt)
        
        # Identify turning points
        turning_prompt = PromptBuilder.build_turning_points_prompt(match_data)
        turning_points = model_manager.generate(turning_prompt)
        
        return jsonify({
            "success": True,
            "analysis": {
                "summary": summary,
                "recommendations": recommendations,
                "turning_points": turning_points
            },
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "match": {
                    "team_a": match_data.get('team_a'),
                    "team_b": match_data.get('team_b'),
                    "winner": match_data.get('winner')
                }
            }
        })
        
    except Exception as e:
        logger.error(f"Error in analyze_match: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/config', methods=['GET'])
def get_config():
    """Get available configuration options"""
    return jsonify({
        "summary_styles": list(config.SUMMARY_STYLES.keys()),
        "focus_areas": list(config.FOCUS_AREAS.keys()),
        "length_options": ["short", "medium", "long"],
        "default_temperature": config.TEMPERATURE,
        "default_max_length": config.MAX_LENGTH,
        "device": config.DEVICE
    })

# ============================================================
# Startup
# ============================================================

@app.before_request
def before_first_request():
    """Load model before first request"""
    if not model_manager.loaded:
        logger.info("Loading model on first request...")
        success = model_manager.load_model()
        if not success:
            logger.error("Failed to load model on startup")

def main():
    """Main entry point"""
    print("=" * 60)
    print("   ESports Strategy Summarizer - Backend API")
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
    print(f"   Port: 5000")
    print("\n   API Endpoints:")
    print("   - POST /api/summarize       - Generate match summary")
    print("   - POST /api/recommendations - Get tactical recommendations")
    print("   - POST /api/turning-points  - Identify turning points")
    print("   - POST /api/analyze         - Complete match analysis")
    print("   - GET  /api/config          - Get configuration options")
    print("   - GET  /health              - Health check")
    print("\n" + "=" * 60)
    print("   ✅ Ready to serve requests!")
    print("=" * 60 + "\n")
    
    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True
    )

if __name__ == '__main__':
    main()
