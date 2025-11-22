#!/bin/bash

# ============================================================
# ESports Strategy AI - Complete Backend Setup Script
# ============================================================
# This script sets up the entire backend including:
# - Directory structure
# - All Python files
# - Requirements
# - Environment configuration
# ============================================================

set -e  # Exit on any error

echo "============================================================"
echo "    ESports Strategy AI - Backend Setup"
echo "============================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$SCRIPT_DIR"

echo -e "${BLUE}📁 Project directory: $PROJECT_DIR${NC}"
echo ""

# ============================================================
# Create directory structure
# ============================================================
echo -e "${YELLOW}📂 Creating directory structure...${NC}"

mkdir -p "$PROJECT_DIR/backend"
mkdir -p "$PROJECT_DIR/data"
mkdir -p "$PROJECT_DIR/models"
mkdir -p "$PROJECT_DIR/scripts"
mkdir -p "$PROJECT_DIR/sample_matches"

echo -e "${GREEN}✓ Directories created${NC}"

# ============================================================
# Create requirements.txt
# ============================================================
echo -e "${YELLOW}📦 Creating requirements.txt...${NC}"

cat > "$PROJECT_DIR/requirements.txt" << 'REQUIREMENTS_EOF'
# ESports Strategy AI - Requirements
# Python 3.9+ required

# ==== Core ML/AI ====
torch>=2.0.0
transformers>=4.36.0
datasets>=2.16.0
accelerate>=0.25.0
bitsandbytes>=0.41.0
peft>=0.7.0

# ==== API Clients ====
groq>=0.4.0
requests>=2.31.0

# ==== Web Framework ====
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
python-multipart>=0.0.6
pydantic>=2.5.0

# ==== Data Processing ====
pandas>=2.0.0
numpy>=1.24.0

# ==== Utilities ====
python-dotenv>=1.0.0
tqdm>=4.66.0
aiohttp>=3.9.0

# ==== Evaluation ====
rouge-score>=0.1.2
nltk>=3.8.1

# ==== Development ====
jupyter>=1.0.0
ipywidgets>=8.1.0
REQUIREMENTS_EOF

echo -e "${GREEN}✓ requirements.txt created${NC}"

# ============================================================
# Create .env template
# ============================================================
echo -e "${YELLOW}🔐 Creating .env template...${NC}"

cat > "$PROJECT_DIR/.env.example" << 'ENV_EOF'
# ESports Strategy AI - Environment Variables
# Copy this file to .env and fill in your values

# Groq API Key (Required) - Get free key at https://console.groq.com/
GROQ_API_KEY=your_groq_api_key_here

# Optional: LLM Backend (groq, local, auto)
LLM_BACKEND=groq

# Optional: Local model path (if using fine-tuned model)
LOCAL_MODEL_PATH=./models/esports-strategy-lora
ENV_EOF

# Create .env if it doesn't exist
if [ ! -f "$PROJECT_DIR/.env" ]; then
    cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"
    echo -e "${YELLOW}⚠️  Created .env file - Please add your GROQ_API_KEY${NC}"
else
    echo -e "${GREEN}✓ .env file already exists${NC}"
fi

# ============================================================
# Create backend/llm_engine.py
# ============================================================
echo -e "${YELLOW}🤖 Creating backend/llm_engine.py...${NC}"

cat > "$PROJECT_DIR/backend/llm_engine.py" << 'LLM_ENGINE_EOF'
import requests
import os
from dotenv import load_dotenv
import time

load_dotenv()

class ESportsLLM:
    def __init__(self):
        """
        Use Groq API - Fast, reliable, and free!
        Updated with current model names (as of Nov 2024)
        """
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.api_key = os.getenv('GROQ_API_KEY', '')
        
        if not self.api_key:
            print("❌ GROQ_API_KEY not found in .env file")
            print("   Get your free API key at: https://console.groq.com/")
            raise ValueError("GROQ_API_KEY is required")
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Current available models (Nov 2024)
        self.models = {
            "llama-3.1-70b": "llama-3.1-70b-versatile",     # Best quality
            "llama-3.1-8b": "llama-3.1-8b-instant",         # Fast & good
            "llama-3.2-3b": "llama-3.2-3b-preview",         # Lightweight
            "mixtral": "mixtral-8x7b-32768",                # Very good
            "gemma2-9b": "gemma2-9b-it"                     # Alternative
        }
        
        # Use the fast model by default
        self.current_model = self.models["llama-3.1-8b"]
        
        print("✅ Using Groq API (Fast & Free!)")
        print(f"📦 Model: {self.current_model}")
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self):
        """Test if API key works"""
        try:
            test_payload = {
                "model": self.current_model,
                "messages": [
                    {"role": "user", "content": "Hello"}
                ],
                "max_tokens": 10
            }
            
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=test_payload,
                timeout=10
            )
            
            if response.status_code == 200:
                print("✅ API connection successful")
                print("✅ System ready!")
                return True
            else:
                print(f"⚠️  API test failed: {response.status_code}")
                print(f"Response: {response.text}")
                
                # Try to auto-fix by listing available models
                if response.status_code == 400:
                    print("\n🔄 Attempting to fetch available models...")
                    self._get_available_models()
                
                return False
                
        except Exception as e:
            print(f"⚠️  Connection test error: {e}")
            return False
    
    def _get_available_models(self):
        """Fetch list of available models from Groq"""
        try:
            models_url = "https://api.groq.com/openai/v1/models"
            response = requests.get(
                models_url,
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                models_data = response.json()
                available_models = [model["id"] for model in models_data.get("data", [])]
                
                if available_models:
                    print("\n📋 Available models:")
                    for model in available_models:
                        print(f"   - {model}")
                    
                    # Auto-select first available model
                    self.current_model = available_models[0]
                    print(f"\n✅ Auto-selected: {self.current_model}")
                    
        except Exception as e:
            print(f"⚠️  Could not fetch models: {e}")
    
    def generate_response(self, prompt, max_length=1024, temperature=0.7, retry_count=3):
        """
        Generate response using Groq API
        Super fast - usually responds in 1-3 seconds!
        """
        
        payload = {
            "model": self.current_model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a professional esports analyst with deep knowledge of competitive gaming. Provide detailed, insightful analysis and strategic recommendations."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": temperature,
            "max_tokens": max_length,
            "top_p": 0.9
        }
        
        for attempt in range(retry_count):
            try:
                print(f"🔄 Generating response (attempt {attempt + 1}/{retry_count})...")
                start_time = time.time()
                
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=60
                )
                
                elapsed_time = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    text = result["choices"][0]["message"]["content"]
                    
                    print(f"✅ Response generated in {elapsed_time:.2f}s")
                    return text
                
                elif response.status_code == 400:
                    error_data = response.json()
                    error_msg = error_data.get("error", {}).get("message", "Unknown error")
                    
                    if "decommissioned" in error_msg or "not supported" in error_msg:
                        print(f"⚠️  Model {self.current_model} is no longer available")
                        print("🔄 Fetching available models...")
                        self._get_available_models()
                        
                        if attempt < retry_count - 1:
                            print("Retrying with new model...")
                            continue
                    
                    return f"Error: {error_msg}"
                
                elif response.status_code == 429:
                    print("⚠️  Rate limited, waiting 5 seconds...")
                    time.sleep(5)
                    continue
                
                elif response.status_code == 401:
                    return "Error: Invalid Groq API key. Please check your .env file."
                
                else:
                    print(f"❌ API Error: {response.status_code}")
                    error_text = response.text[:300]
                    print(f"Response: {error_text}")
                    
                    if attempt < retry_count - 1:
                        print("Retrying in 3 seconds...")
                        time.sleep(3)
                        continue
                    
                    return f"Error: API returned status code {response.status_code}"
            
            except requests.exceptions.Timeout:
                print(f"⏱️  Request timed out")
                if attempt < retry_count - 1:
                    time.sleep(3)
                    continue
                return "Error: Request timed out. Please try again."
            
            except requests.exceptions.RequestException as e:
                print(f"❌ Network error: {e}")
                if attempt < retry_count - 1:
                    time.sleep(3)
                    continue
                return f"Error: Network issue - {str(e)}"
            
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                return f"Error: {str(e)}"
        
        return "Error: Failed to generate response after multiple attempts."
    
    def switch_model(self, model_name):
        """Switch to a different Groq model"""
        if model_name in self.models:
            self.current_model = self.models[model_name]
            print(f"✅ Switched to model: {self.current_model}")
        else:
            print(f"⚠️  Unknown model: {model_name}")
            print(f"Available: {list(self.models.keys())}")
LLM_ENGINE_EOF

echo -e "${GREEN}✓ llm_engine.py created${NC}"

# ============================================================
# Create backend/prompt_templates.py
# ============================================================
echo -e "${YELLOW}📝 Creating backend/prompt_templates.py...${NC}"

cat > "$PROJECT_DIR/backend/prompt_templates.py" << 'PROMPT_TEMPLATES_EOF'
"""
Prompt Templates for ESports Strategy Summarizer
Comprehensive prompts for match analysis with customizable:
- Length (short, medium, long, detailed)
- Focus areas (overall, team_fights, economy, objectives, positioning, draft)
- Style (analytical, narrative, technical, casual)
"""

from typing import List, Dict, Any
import json


class PromptTemplates:
    """Collection of prompt templates for esports analysis."""
    
    # Style instructions
    STYLE_INSTRUCTIONS = {
        "analytical": "Use a professional, data-driven analytical tone. Focus on statistics, patterns, and objective observations. Structure your analysis with clear logical flow.",
        "narrative": "Use an engaging, story-telling approach. Make the analysis feel like a compelling match recap that captures the drama and excitement of key moments.",
        "technical": "Use precise technical terminology and game-specific jargon. Assume the reader has deep game knowledge. Focus on mechanical details and advanced concepts.",
        "casual": "Use a friendly, accessible tone. Explain concepts clearly for viewers who may be newer to competitive play. Keep it engaging and easy to follow."
    }
    
    # Length instructions
    LENGTH_INSTRUCTIONS = {
        "short": "Keep your response concise and focused. Aim for 200-400 words total. Hit only the most critical points.",
        "medium": "Provide a balanced analysis. Aim for 500-800 words total. Cover key points with moderate detail.",
        "long": "Provide comprehensive analysis. Aim for 1000-1500 words total. Include detailed breakdowns of each section.",
        "detailed": "Provide an exhaustive analysis. Aim for 1500-2500 words total. Leave no stone unturned - this is a full coaching-level breakdown."
    }
    
    # Focus area descriptions
    FOCUS_DESCRIPTIONS = {
        "overall": "Provide a holistic view of the match covering all aspects",
        "team_fights": "Focus heavily on teamfight execution, positioning, ability usage, and fight outcomes",
        "economy": "Focus on resource management, gold leads, item timings, and economic decisions",
        "objectives": "Focus on objective control, dragon/baron/tower plays, and map control",
        "positioning": "Focus on player positioning, rotations, vision control, and map movements",
        "draft": "Focus on draft strategy, team composition synergies, and pick/ban decisions"
    }
    
    @classmethod
    def _format_match_data(cls, match_data: Dict) -> str:
        """Format match data into a readable string for the prompt."""
        sections = []
        
        # Basic info
        if "match_id" in match_data:
            sections.append(f"Match ID: {match_data['match_id']}")
        
        if "game" in match_data:
            sections.append(f"Game: {match_data['game']}")
        
        # Teams
        if "teams" in match_data:
            teams = match_data["teams"]
            sections.append(f"\nTeams:")
            for team_key, team_name in teams.items():
                sections.append(f"  - {team_key}: {team_name}")
        
        # Duration and winner
        if "duration_minutes" in match_data:
            sections.append(f"\nDuration: {match_data['duration_minutes']} minutes")
        
        if "winner" in match_data:
            winner = match_data["winner"]
            if "teams" in match_data and winner in match_data["teams"]:
                winner = match_data["teams"][winner]
            sections.append(f"Winner: {winner}")
        
        # Stats
        if "stats" in match_data:
            stats = match_data["stats"]
            sections.append("\nMatch Statistics:")
            
            if "kills" in stats:
                sections.append(f"  Kills: {json.dumps(stats['kills'])}")
            if "towers" in stats:
                sections.append(f"  Towers: {json.dumps(stats['towers'])}")
            if "dragons" in stats:
                sections.append(f"  Dragons: {json.dumps(stats['dragons'])}")
            if "barons" in stats:
                sections.append(f"  Barons: {json.dumps(stats['barons'])}")
            if "gold" in stats:
                sections.append(f"  Total Gold: {json.dumps(stats['gold'])}")
            if "gold_diff_at_15" in stats:
                sections.append(f"  Gold Diff @15min: {stats['gold_diff_at_15']}")
            if "first_blood" in stats:
                sections.append(f"  First Blood: {stats['first_blood']}")
        
        # Compositions
        if "compositions" in match_data:
            sections.append("\nTeam Compositions:")
            for team, comp in match_data["compositions"].items():
                if isinstance(comp, dict):
                    comp_str = ", ".join([f"{role}: {champ}" for role, champ in comp.items()])
                else:
                    comp_str = str(comp)
                sections.append(f"  {team}: {comp_str}")
        
        # Key Events
        if "key_events" in match_data:
            sections.append("\nKey Events Timeline:")
            for event in match_data["key_events"]:
                time = event.get("time", "??:??")
                event_type = event.get("type", event.get("event", "Unknown"))
                team = event.get("team", "")
                desc = event.get("description", "")
                sections.append(f"  [{time}] {event_type} ({team}): {desc}")
        
        # Transcript
        if "transcript" in match_data:
            sections.append(f"\nMatch Commentary/Transcript:\n{match_data['transcript'][:2000]}")
        
        # Player stats
        if "player_stats" in match_data:
            sections.append("\nPlayer Performance:")
            for team, players in match_data["player_stats"].items():
                sections.append(f"  {team}:")
                for role, stats in players.items():
                    kda = f"{stats.get('kills', 0)}/{stats.get('deaths', 0)}/{stats.get('assists', 0)}"
                    sections.append(f"    {role}: {kda} KDA, {stats.get('cs', 0)} CS, {stats.get('damage', 0)} damage")
        
        return "\n".join(sections)
    
    @classmethod
    def _get_focus_instruction(cls, focus_areas: List[str]) -> str:
        """Generate focus instruction based on selected areas."""
        if "overall" in focus_areas or not focus_areas:
            return "Provide comprehensive coverage of all aspects of the match."
        
        focus_texts = []
        for area in focus_areas:
            if area in cls.FOCUS_DESCRIPTIONS:
                focus_texts.append(cls.FOCUS_DESCRIPTIONS[area])
        
        if focus_texts:
            return "Pay special attention to: " + "; ".join(focus_texts)
        return ""
    
    @classmethod
    def comprehensive_match_analysis(
        cls,
        match_data: Dict,
        length: str = "medium",
        focus: List[str] = None,
        style: str = "analytical",
        focus_team: str = "team_a"
    ) -> str:
        """
        Generate a comprehensive match analysis prompt.
        
        Returns analysis with:
        1. Match Overview/Summary
        2. Key Turning Points
        3. Tactical Recommendations
        4. Key Takeaways
        """
        focus = focus or ["overall"]
        
        # Get instructions
        style_instruction = cls.STYLE_INSTRUCTIONS.get(style, cls.STYLE_INSTRUCTIONS["analytical"])
        length_instruction = cls.LENGTH_INSTRUCTIONS.get(length, cls.LENGTH_INSTRUCTIONS["medium"])
        focus_instruction = cls._get_focus_instruction(focus)
        
        # Format match data
        formatted_data = cls._format_match_data(match_data)
        
        # Get team name for recommendations
        focus_team_name = match_data.get("teams", {}).get(focus_team, focus_team)
        
        prompt = f"""You are an elite professional esports analyst providing a comprehensive match breakdown.

{style_instruction}

{length_instruction}

{focus_instruction}

=== MATCH DATA ===
{formatted_data}
=== END MATCH DATA ===

Provide a COMPREHENSIVE analysis with the following sections:

## 1. MATCH OVERVIEW
Summarize the overall flow of the match. What was the story of this game? How did the match evolve from early game to conclusion? What defined the winning team's victory?

## 2. KEY TURNING POINTS
Identify 2-4 critical moments that significantly impacted the match outcome. For each turning point:
- What happened and when
- Why it was significant
- How it changed the game state
- Could it have been prevented or played differently?

## 3. TACTICAL RECOMMENDATIONS FOR {focus_team_name.upper()}
Provide specific, actionable recommendations for {focus_team_name} based on this match:

**Immediate Fixes** (for next game):
- 2-3 things to change immediately

**Strategic Improvements** (for practice):
- Draft/composition considerations
- Macro strategy adjustments
- Team coordination improvements

**Individual Focus Areas**:
- Specific skills or plays to practice

## 4. KEY TAKEAWAYS
Summarize the most important lessons from this match in 3-5 bullet points that any team could learn from.

---
Remember to be specific and reference actual events from the match data. Support your analysis with evidence from the game."""

        return prompt
    
    @classmethod
    def quick_summary(cls, match_data: Dict) -> str:
        """Generate a quick summary prompt for rapid analysis."""
        formatted_data = cls._format_match_data(match_data)
        
        return f"""Provide a brief 2-3 paragraph summary of this match.

{formatted_data}

Include: winner, match duration, the single most important factor in the outcome, and one key lesson.
Be concise but insightful."""
    
    @classmethod
    def turning_points_only(cls, match_data: Dict) -> str:
        """Generate a prompt focused only on turning points."""
        formatted_data = cls._format_match_data(match_data)
        
        return f"""Analyze the KEY TURNING POINTS in this match.

{formatted_data}

For each turning point (identify 2-4), provide:
1. **Timestamp/Moment**: When it happened
2. **What Happened**: Describe the event
3. **Impact**: How it changed the game state
4. **Analysis**: Why it mattered and could it have been different?

Rank them by importance to the final outcome."""
    
    @classmethod
    def recommendations_only(cls, match_data: Dict, team: str) -> str:
        """Generate a prompt focused only on recommendations."""
        formatted_data = cls._format_match_data(match_data)
        team_name = match_data.get("teams", {}).get(team, team)
        
        return f"""Based on this match, provide STRATEGIC RECOMMENDATIONS for {team_name}.

{formatted_data}

Structure your recommendations as:

## Immediate Fixes (Next Game)
- 3 things to change right now

## Short-Term Improvements (This Week)
- Draft adjustments
- Strategy changes
- Communication improvements

## Long-Term Development (Next Month)
- Skills to practice
- Patterns to break
- New strategies to develop

## Warning Signs to Watch
- Mistakes that should not be repeated
- Bad habits identified

Be specific and actionable. Reference events from this match as examples."""
    
    @classmethod
    def draft_analysis(cls, match_data: Dict) -> str:
        """Generate a prompt focused on draft/composition analysis."""
        formatted_data = cls._format_match_data(match_data)
        
        return f"""Analyze the DRAFT and TEAM COMPOSITIONS from this match.

{formatted_data}

Analyze:
1. **Draft Strategy**: What was each team trying to achieve?
2. **Win Conditions**: How was each composition supposed to win?
3. **Power Spikes**: When was each team strongest?
4. **Synergies**: What combos existed within each team?
5. **Counters**: How did picks interact/counter each other?
6. **Draft Grade**: Rate each draft (A-F) with explanation
7. **Alternative Picks**: What else could have worked?

Did the draft advantage (if any) translate to in-game advantage?"""

    @classmethod
    def teamfight_analysis(cls, match_data: Dict) -> str:
        """Generate a prompt focused on teamfight analysis."""
        formatted_data = cls._format_match_data(match_data)
        
        return f"""Analyze the TEAMFIGHT EXECUTION in this match.

{formatted_data}

Focus on:
1. **Fight Outcomes**: Which team won more fights and why?
2. **Engage Patterns**: How did teams initiate fights?
3. **Target Priority**: Who was focused and was it optimal?
4. **Ability Usage**: Were key abilities used effectively?
5. **Positioning**: Analyze frontline/backline positioning
6. **Critical Fight Breakdown**: Detail the most important teamfight

What separated the winning team's teamfight execution?"""
PROMPT_TEMPLATES_EOF

echo -e "${GREEN}✓ prompt_templates.py created${NC}"

# ============================================================
# Create backend/main.py
# ============================================================
echo -e "${YELLOW}🚀 Creating backend/main.py...${NC}"

cat > "$PROJECT_DIR/backend/main.py" << 'MAIN_EOF'
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import json
from llm_engine import ESportsLLM
from prompt_templates import PromptTemplates
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List

llm_engine = None
executor = ThreadPoolExecutor(max_workers=3)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm_engine
    print("🚀 Initializing ESports Strategy AI...")
    llm_engine = ESportsLLM()
    print("✅ System ready!")
    yield
    print("👋 Shutting down...")
    executor.shutdown(wait=True)

app = FastAPI(
    title="ESports Strategy AI",
    description="AI-powered comprehensive match analysis",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PromptControls(BaseModel):
    """Simplified prompt controls"""
    length: str = "medium"  # short, medium, long, detailed
    focus: List[str] = ["overall"]  # overall, team_fights, economy, objectives, positioning, draft
    style: str = "analytical"  # analytical, narrative, technical, casual

class MatchAnalysisRequest(BaseModel):
    match_data: dict
    focus_team: str = "team_a"
    prompt_controls: Optional[PromptControls] = None

class QuickAnalysisRequest(BaseModel):
    match_data: dict

class RecommendationsRequest(BaseModel):
    match_data: dict
    team: str = "team_a"

@app.post("/analyze")
async def analyze_match(request: MatchAnalysisRequest):
    """
    Comprehensive match analysis endpoint
    Returns: Overview, Turning Points, Tactical Recommendations, Key Takeaways
    """
    try:
        match_data = request.match_data
        
        if not match_data:
            return {"error": "No match data provided"}
        
        controls = request.prompt_controls or PromptControls()
        
        print(f"📊 Comprehensive analysis request: length={controls.length}, style={controls.style}, focus={controls.focus}")
        
        # Generate comprehensive analysis
        try:
            prompt = PromptTemplates.comprehensive_match_analysis(
                match_data,
                length=controls.length,
                focus=controls.focus,
                style=controls.style,
                focus_team=request.focus_team
            )
        except Exception as e:
            print(f"❌ Error creating prompt: {e}")
            import traceback
            traceback.print_exc()
            return {"error": f"Error creating prompt: {str(e)}"}
        
        # Adjust max_length based on length control
        max_length_map = {
            "short": 1500,
            "medium": 2500,
            "long": 3500,
            "detailed": 5000
        }
        max_length = max_length_map.get(controls.length, 2500)
        
        # Generate response
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                executor,
                lambda: llm_engine.generate_response(prompt, max_length=max_length)
            )
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            import traceback
            traceback.print_exc()
            return {"error": f"Error generating AI response: {str(e)}"}
        
        print("✅ Comprehensive analysis complete")
        
        return {
            "result": response,
            "match_id": match_data.get("match_id", "unknown"),
            "focus_team": match_data.get("teams", {}).get(request.focus_team, request.focus_team),
            "controls_used": controls.dict()
        }
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"Unexpected error: {str(e)}"}

@app.post("/quick-summary")
async def quick_summary(request: QuickAnalysisRequest):
    """Quick summary endpoint for rapid analysis"""
    try:
        prompt = PromptTemplates.quick_summary(request.match_data)
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            executor,
            lambda: llm_engine.generate_response(prompt, max_length=800)
        )
        
        return {"result": response}
    except Exception as e:
        return {"error": str(e)}

@app.post("/turning-points")
async def turning_points(request: QuickAnalysisRequest):
    """Analyze turning points only"""
    try:
        prompt = PromptTemplates.turning_points_only(request.match_data)
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            executor,
            lambda: llm_engine.generate_response(prompt, max_length=1500)
        )
        
        return {"result": response}
    except Exception as e:
        return {"error": str(e)}

@app.post("/recommendations")
async def recommendations(request: RecommendationsRequest):
    """Get recommendations for a specific team"""
    try:
        prompt = PromptTemplates.recommendations_only(request.match_data, request.team)
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            executor,
            lambda: llm_engine.generate_response(prompt, max_length=2000)
        )
        
        return {"result": response, "team": request.team}
    except Exception as e:
        return {"error": str(e)}

@app.post("/draft-analysis")
async def draft_analysis(request: QuickAnalysisRequest):
    """Analyze draft and compositions"""
    try:
        prompt = PromptTemplates.draft_analysis(request.match_data)
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            executor,
            lambda: llm_engine.generate_response(prompt, max_length=1500)
        )
        
        return {"result": response}
    except Exception as e:
        return {"error": str(e)}

@app.post("/teamfight-analysis")
async def teamfight_analysis(request: QuickAnalysisRequest):
    """Analyze teamfight execution"""
    try:
        prompt = PromptTemplates.teamfight_analysis(request.match_data)
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            executor,
            lambda: llm_engine.generate_response(prompt, max_length=1500)
        )
        
        return {"result": response}
    except Exception as e:
        return {"error": str(e)}

@app.post("/upload-match")
async def upload_match_file(file: UploadFile = File(...)):
    """Upload match data JSON file"""
    try:
        content = await file.read()
        match_data = json.loads(content)
        
        return {
            "status": "success",
            "match_id": match_data.get("match_id", "unknown"),
            "teams": match_data.get("teams", {}),
            "match_data": match_data
        }
    except Exception as e:
        print(f"❌ Error in upload endpoint: {e}")
        return {"error": str(e)}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": llm_engine is not None,
        "api_type": "groq"
    }

@app.get("/prompt-options")
async def get_prompt_options():
    return {
        "length_options": ["short", "medium", "long", "detailed"],
        "focus_options": ["overall", "team_fights", "economy", "objectives", "positioning", "draft"],
        "style_options": ["analytical", "narrative", "technical", "casual"],
        "endpoints": {
            "/analyze": "Comprehensive analysis with all sections",
            "/quick-summary": "Brief 2-3 paragraph summary",
            "/turning-points": "Key turning points analysis",
            "/recommendations": "Recommendations for specific team",
            "/draft-analysis": "Draft and composition analysis",
            "/teamfight-analysis": "Teamfight execution analysis"
        }
    }

@app.get("/")
async def root():
    return {
        "message": "ESports Strategy AI API - Comprehensive Match Analysis",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": ["/analyze", "/quick-summary", "/turning-points", "/recommendations", "/draft-analysis", "/teamfight-analysis"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
MAIN_EOF

echo -e "${GREEN}✓ main.py created${NC}"

# ============================================================
# Create sample match data
# ============================================================
echo -e "${YELLOW}📄 Creating sample match data...${NC}"

cat > "$PROJECT_DIR/sample_matches/sample_match.json" << 'SAMPLE_MATCH_EOF'
{
  "match_id": "lcs_finals_2024_g1",
  "game": "League of Legends",
  "teams": {
    "team_a": "Cloud9",
    "team_b": "Team Liquid"
  },
  "duration_minutes": 35,
  "winner": "team_a",
  "stats": {
    "kills": {"team_a": 22, "team_b": 15},
    "towers": {"team_a": 9, "team_b": 3},
    "dragons": {"team_a": 4, "team_b": 2},
    "barons": {"team_a": 1, "team_b": 0},
    "gold": {"team_a": 65200, "team_b": 58100},
    "gold_diff_at_15": -1500,
    "first_blood": "team_b",
    "first_tower": "team_b"
  },
  "compositions": {
    "team_a": {
      "top": "Gnar",
      "jungle": "Lee Sin",
      "mid": "Azir",
      "adc": "Jinx",
      "support": "Thresh"
    },
    "team_b": {
      "top": "Renekton",
      "jungle": "Viego",
      "mid": "Viktor",
      "adc": "Kai'Sa",
      "support": "Nautilus"
    }
  },
  "key_events": [
    {"time": "5:15", "type": "first_blood", "team": "team_b", "description": "Viego ganks mid, kills Azir"},
    {"time": "7:00", "type": "dragon", "team": "team_b", "description": "Mountain Drake secured"},
    {"time": "10:30", "type": "tower", "team": "team_b", "description": "First tower mid lane"},
    {"time": "18:00", "type": "teamfight", "team": "team_a", "description": "Azir shuffle catches 3, C9 wins 4-1 at dragon"},
    {"time": "22:00", "type": "dragon", "team": "team_a", "description": "Ocean Drake after skirmish win"},
    {"time": "25:30", "type": "baron", "team": "team_a", "description": "Thresh hook on Viego leads to Baron"},
    {"time": "32:00", "type": "dragon_soul", "team": "team_a", "description": "Infernal Soul secured"},
    {"time": "35:00", "type": "game_end", "team": "team_a", "description": "Jinx pentakill ends the game"}
  ],
  "transcript": "Caster 1: Welcome to the LCS Finals! Cloud9 vs Team Liquid!\nCaster 2: C9 drafts scaling with Jinx and Azir. Liquid goes early game.\nCaster 1: 5 minutes in, first blood to Liquid! Viego catches Azir.\nCaster 2: Liquid is executing their gameplan perfectly.\nCaster 1: 15 minutes, Liquid leads by 1500 gold.\nCaster 2: But here comes the dragon fight!\nCaster 1: AZIR SHUFFLE! He catches THREE members!\nCaster 2: Cloud9 wins the fight 4-1! What a turnaround!\nCaster 1: 25 minutes, Baron is up. Thresh lands the hook!\nCaster 2: That's Baron for Cloud9! The comeback is real!\nCaster 1: Dragon Soul to Cloud9! They push for the win!\nCaster 2: PENTAKILL! Jinx cleans up! Cloud9 wins!"
}
SAMPLE_MATCH_EOF

echo -e "${GREEN}✓ sample_match.json created${NC}"

# ============================================================
# Create test script
# ============================================================
echo -e "${YELLOW}🧪 Creating test script...${NC}"

cat > "$PROJECT_DIR/test_api.py" << 'TEST_EOF'
#!/usr/bin/env python3
"""
Test script for ESports Strategy AI API
Run this after starting the server to verify everything works
"""

import requests
import json

API_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("\n🔍 Testing /health...")
    response = requests.get(f"{API_URL}/health")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    return response.status_code == 200

def test_prompt_options():
    """Test prompt options endpoint"""
    print("\n🔍 Testing /prompt-options...")
    response = requests.get(f"{API_URL}/prompt-options")
    print(f"   Status: {response.status_code}")
    data = response.json()
    print(f"   Length options: {data.get('length_options')}")
    print(f"   Style options: {data.get('style_options')}")
    return response.status_code == 200

def test_analyze():
    """Test main analysis endpoint"""
    print("\n🔍 Testing /analyze...")
    
    # Load sample match
    with open("sample_matches/sample_match.json", "r") as f:
        match_data = json.load(f)
    
    payload = {
        "match_data": match_data,
        "focus_team": "team_a",
        "prompt_controls": {
            "length": "short",
            "focus": ["overall"],
            "style": "analytical"
        }
    }
    
    response = requests.post(f"{API_URL}/analyze", json=payload)
    print(f"   Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        if "error" in data:
            print(f"   ❌ Error: {data['error']}")
            return False
        result = data.get("result", "")
        print(f"   ✅ Response length: {len(result)} chars")
        print(f"   Preview: {result[:200]}...")
        return True
    return False

def test_quick_summary():
    """Test quick summary endpoint"""
    print("\n🔍 Testing /quick-summary...")
    
    with open("sample_matches/sample_match.json", "r") as f:
        match_data = json.load(f)
    
    payload = {"match_data": match_data}
    
    response = requests.post(f"{API_URL}/quick-summary", json=payload)
    print(f"   Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        result = data.get("result", "")
        print(f"   ✅ Response length: {len(result)} chars")
        return True
    return False

def main():
    print("=" * 60)
    print("   ESports Strategy AI - API Tests")
    print("=" * 60)
    
    results = {
        "Health": test_health(),
        "Prompt Options": test_prompt_options(),
        "Analyze": test_analyze(),
        "Quick Summary": test_quick_summary(),
    }
    
    print("\n" + "=" * 60)
    print("   Results Summary")
    print("=" * 60)
    
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {name}: {status}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    print(f"\n   Total: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n🎉 All tests passed! The API is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the server logs.")

if __name__ == "__main__":
    main()
TEST_EOF

chmod +x "$PROJECT_DIR/test_api.py"

echo -e "${GREEN}✓ test_api.py created${NC}"

# ============================================================
# Create run script
# ============================================================
echo -e "${YELLOW}🏃 Creating run script...${NC}"

cat > "$PROJECT_DIR/run.sh" << 'RUN_EOF'
#!/bin/bash

# ESports Strategy AI - Run Script

echo "============================================================"
echo "    ESports Strategy AI - Starting Server"
echo "============================================================"

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "❌ .env file not found!"
    echo "   Copy .env.example to .env and add your GROQ_API_KEY"
    exit 1
fi

# Check if GROQ_API_KEY is set
if ! grep -q "GROQ_API_KEY=gsk_" .env; then
    echo "⚠️  Warning: GROQ_API_KEY may not be set correctly in .env"
    echo "   Get your free key at: https://console.groq.com/"
fi

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "📦 Activating virtual environment..."
    source venv/bin/activate
fi

# Start the server
echo "🚀 Starting FastAPI server..."
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
RUN_EOF

chmod +x "$PROJECT_DIR/run.sh"

echo -e "${GREEN}✓ run.sh created${NC}"

# ============================================================
# Create install script
# ============================================================
echo -e "${YELLOW}📥 Creating install script...${NC}"

cat > "$PROJECT_DIR/install.sh" << 'INSTALL_EOF'
#!/bin/bash

# ESports Strategy AI - Installation Script

echo "============================================================"
echo "    ESports Strategy AI - Installation"
echo "============================================================"

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "📦 Python version: $python_version"

# Create virtual environment
echo ""
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📦 Installing requirements..."
pip install -r requirements.txt

echo ""
echo "============================================================"
echo "    Installation Complete!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "1. Edit .env file and add your GROQ_API_KEY"
echo "   Get your free key at: https://console.groq.com/"
echo ""
echo "2. Run the server:"
echo "   ./run.sh"
echo ""
echo "3. Open http://localhost:8000/docs in your browser"
echo ""
INSTALL_EOF

chmod +x "$PROJECT_DIR/install.sh"

echo -e "${GREEN}✓ install.sh created${NC}"

# ============================================================
# Final Summary
# ============================================================
echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}    Setup Complete!${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo "Files created:"
echo "  📁 backend/"
echo "     ├── main.py           (FastAPI server)"
echo "     ├── llm_engine.py     (Groq LLM integration)"
echo "     └── prompt_templates.py (Prompt engineering)"
echo "  📁 sample_matches/"
echo "     └── sample_match.json (Sample data)"
echo "  📄 requirements.txt"
echo "  📄 .env.example"
echo "  📄 .env"
echo "  📄 test_api.py"
echo "  📄 run.sh"
echo "  📄 install.sh"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo ""
echo "1. Add your Groq API key to .env file:"
echo "   ${BLUE}nano .env${NC}"
echo "   (Get free key at https://console.groq.com/)"
echo ""
echo "2. Install dependencies:"
echo "   ${BLUE}./install.sh${NC}"
echo "   OR"
echo "   ${BLUE}pip install -r requirements.txt${NC}"
echo ""
echo "3. Start the server:"
echo "   ${BLUE}./run.sh${NC}"
echo "   OR"
echo "   ${BLUE}cd backend && python main.py${NC}"
echo ""
echo "4. Test the API:"
echo "   ${BLUE}python test_api.py${NC}"
echo ""
echo "5. Open API docs:"
echo "   ${BLUE}http://localhost:8000/docs${NC}"
echo ""
echo -e "${GREEN}============================================================${NC}"