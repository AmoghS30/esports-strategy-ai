# backend/main.py
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

@app.post("/upload-match")
async def upload_match_file(file: UploadFile = File(...)):
    """Upload match data JSON file"""
    try:
        content = await file.read()
        match_data = json.loads(content)
        
        return {
            "status": "success",
            "match_id": match_data.get("match_id", "unknown"),
            "teams": match_data.get("teams", {})
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
        "style_options": ["analytical", "narrative", "technical", "casual"]
    }

@app.get("/")
async def root():
    return {
        "message": "ESports Strategy AI API - Comprehensive Match Analysis",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)