# ESports Strategy Summarizer - Flask Backend

A Flask-based REST API for analyzing esports matches, generating summaries, and providing tactical recommendations using fine-tuned language models.

## Features

- **Match Summarization**: Generate customizable match summaries with different styles (analytical, narrative, bullet, tactical)
- **Tactical Recommendations**: Get strategic suggestions for improving team performance
- **Turning Points Analysis**: Identify key moments that determined match outcomes
- **Complete Match Analysis**: All-in-one endpoint for comprehensive match insights
- **GPU Support**: Optimized for both CPU and GPU inference
- **Flexible Prompting**: Control summary focus, length, and style through API parameters

## Architecture

```
┌─────────────────┐
│   Flask API     │
│  (Port 5000)    │
└────────┬────────┘
         │
         ├─ Model Manager
         │  └─ Fine-tuned LLM (TinyLlama/Phi/Qwen)
         │
         ├─ Prompt Builder
         │  └─ Template-based prompt generation
         │
         └─ API Endpoints
            ├─ /api/summarize
            ├─ /api/recommendations
            ├─ /api/turning-points
            └─ /api/analyze
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, for faster inference)
- Fine-tuned model from training script

### Setup

1. **Clone/Download the backend files**

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set environment variables** (optional):
```bash
export MODEL_DIR="./models/esports-gpu-fast_final"
export MAX_LENGTH="512"
export TEMPERATURE="0.7"
```

4. **Ensure your trained model is available**:
```
models/
└── esports-gpu-fast_final/
    ├── config.json
    ├── adapter_config.json
    ├── adapter_model.bin
    ├── tokenizer_config.json
    └── ...
```

## Usage

### Starting the Server

**Simple start**:
```bash
python esports_backend.py
```

**Production deployment with Gunicorn**:
```bash
gunicorn -w 4 -b 0.0.0.0:5000 esports_backend:app
```

**With custom model path**:
```bash
MODEL_DIR="/path/to/your/model" python esports_backend.py
```

### API Endpoints

#### 1. Health Check
```bash
GET /health
```

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "gpu_available": true,
  "gpu_name": "NVIDIA A10G",
  "timestamp": "2024-01-01T12:00:00"
}
```

#### 2. Get Configuration
```bash
GET /api/config
```

**Response**:
```json
{
  "summary_styles": ["analytical", "narrative", "bullet", "tactical"],
  "focus_areas": ["team_fights", "economy", "objectives", "composition", "macro"],
  "length_options": ["short", "medium", "long"],
  "default_temperature": 0.7,
  "device": "cuda"
}
```

#### 3. Summarize Match
```bash
POST /api/summarize
Content-Type: application/json
```

**Request Body**:
```json
{
  "match_data": {
    "team_a": "Team Liquid",
    "team_b": "Evil Geniuses",
    "winner": "Team Liquid",
    "duration": "42:15",
    "team_a_composition": "Invoker, Anti-Mage, Lion, Earthshaker, Crystal Maiden",
    "team_b_composition": "PA, QoP, Rubick, Tidehunter, AA",
    "events": [
      "5:30 - Team Liquid secures first blood mid lane",
      "18:45 - Major team fight at Roshan pit, Team Liquid wins 4-1",
      "32:00 - Team Liquid wins decisive team fight"
    ],
    "commentary": "An intense match between two top-tier teams...",
    "statistics": {
      "Team Liquid Kills": "45",
      "Evil Geniuses Kills": "32"
    }
  },
  "style": "analytical",
  "focus": "team_fights",
  "length": "medium",
  "temperature": 0.7,
  "max_length": 512
}
```

**Response**:
```json
{
  "success": true,
  "summary": "Team Liquid dominated this match with superior team fighting...",
  "metadata": {
    "style": "analytical",
    "focus": "team_fights",
    "length": "medium",
    "temperature": 0.7,
    "timestamp": "2024-01-01T12:00:00"
  }
}
```

**Parameters**:
- `style`: `analytical` | `narrative` | `bullet` | `tactical`
- `focus`: `team_fights` | `economy` | `objectives` | `composition` | `macro`
- `length`: `short` | `medium` | `long`
- `temperature`: 0.1-2.0 (default: 0.7)
- `max_length`: 128-2048 (default: 512)

#### 4. Get Tactical Recommendations
```bash
POST /api/recommendations
Content-Type: application/json
```

**Request Body**:
```json
{
  "match_data": {
    "team_a": "Team Liquid",
    "team_b": "Evil Geniuses",
    "winner": "Evil Geniuses",
    "team_a_composition": "Spectre, Shadow Fiend, Disruptor, Bounty Hunter, Oracle",
    "events": [
      "Early game dominated by Evil Geniuses",
      "Team Liquid struggled to protect Spectre"
    ],
    "statistics": {
      "Team Liquid Kills": "28",
      "Evil Geniuses Kills": "41"
    }
  },
  "team": "team_a",
  "recommendation_depth": 3,
  "temperature": 0.7
}
```

**Response**:
```json
{
  "success": true,
  "recommendations": "1. Draft tankier heroes to protect Spectre...\n2. Focus on warding...\n3. Adjust timing windows...",
  "metadata": {
    "team": "team_a",
    "recommendation_depth": 3,
    "timestamp": "2024-01-01T12:00:00"
  }
}
```

#### 5. Identify Turning Points
```bash
POST /api/turning-points
Content-Type: application/json
```

**Request Body**:
```json
{
  "match_data": {
    "team_a": "OG",
    "team_b": "Team Secret",
    "winner": "OG",
    "events": [
      "15:30 - OG loses mid tower, down 8k gold",
      "22:00 - OG wins major team fight 5-0, momentum shifts",
      "28:15 - OG secures Roshan and Aegis"
    ]
  },
  "temperature": 0.7
}
```

**Response**:
```json
{
  "success": true,
  "turning_points": "1. The 5-0 team wipe at 22 minutes...\n2. Roshan secure at 28:15...\n3. Final push with Aegis...",
  "metadata": {
    "timestamp": "2024-01-01T12:00:00"
  }
}
```

#### 6. Complete Match Analysis
```bash
POST /api/analyze
Content-Type: application/json
```

**Request Body**:
```json
{
  "match_data": { /* full match data */ },
  "style": "tactical",
  "focus": "objectives",
  "team": "team_a",
  "recommendation_depth": 3
}
```

**Response**:
```json
{
  "success": true,
  "analysis": {
    "summary": "Match summary...",
    "recommendations": "Tactical recommendations...",
    "turning_points": "Key turning points..."
  },
  "metadata": {
    "timestamp": "2024-01-01T12:00:00",
    "match": {
      "team_a": "Fnatic",
      "team_b": "PSG.LGD",
      "winner": "Fnatic"
    }
  }
}
```

## Testing

Run the test client to verify all endpoints:

```bash
python test_api.py
```

This will test:
- Health check
- Configuration retrieval
- Match summarization
- Tactical recommendations
- Turning points analysis
- Complete match analysis

## Match Data Format

The `match_data` object supports the following fields:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `team_a` | string | Yes | First team name |
| `team_b` | string | Yes | Second team name |
| `winner` | string | Yes | Winning team name |
| `duration` | string | No | Match duration (e.g., "42:15") |
| `team_a_composition` | string | No | Team A hero/champion lineup |
| `team_b_composition` | string | No | Team B hero/champion lineup |
| `events` | array[string] | No | Key match events with timestamps |
| `commentary` | string | No | Match commentary transcript |
| `statistics` | object | No | Match statistics (kills, gold, etc.) |

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_DIR` | `./models/esports-gpu-fast_final` | Path to fine-tuned model |
| `MAX_LENGTH` | `512` | Maximum generation length |
| `TEMPERATURE` | `0.7` | Default sampling temperature |
| `TOP_P` | `0.9` | Nucleus sampling parameter |
| `TOP_K` | `50` | Top-k sampling parameter |

### Model Configuration

The backend automatically detects and uses:
- **GPU (CUDA)**: If available, uses FP16 for faster inference
- **CPU**: Falls back to FP32 on CPU

## Performance

### Inference Speed

On **g5.2xlarge** (NVIDIA A10G):
- TinyLlama (1.1B): ~80 samples/second
- Qwen-0.5B: ~120 samples/second
- Phi-2 (2.7B): ~40 samples/second

On **CPU** (4 cores):
- TinyLlama: ~5-10 samples/second
- Qwen-0.5B: ~8-15 samples/second

### Memory Requirements

| Model | GPU VRAM | CPU RAM |
|-------|----------|---------|
| TinyLlama-1.1B | ~3 GB | ~4 GB |
| Qwen-0.5B | ~2 GB | ~3 GB |
| Phi-2-2.7B | ~6 GB | ~8 GB |

## Error Handling

The API returns appropriate HTTP status codes:

| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Bad Request (missing/invalid data) |
| 500 | Internal Server Error |
| 503 | Service Unavailable (model not loaded) |

**Error Response Format**:
```json
{
  "error": "Error message describing what went wrong"
}
```

## Production Deployment

### Using Gunicorn

```bash
gunicorn -w 4 -b 0.0.0.0:5000 --timeout 120 esports_backend:app
```

### Docker Deployment

Create a `Dockerfile`:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY esports_backend.py .
COPY models/ models/

EXPOSE 5000

CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5000", "--timeout", "120", "esports_backend:app"]
```

Build and run:
```bash
docker build -t esports-api .
docker run -p 5000:5000 -v $(pwd)/models:/app/models esports-api
```

### NGINX Reverse Proxy

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 120s;
    }
}
```

## Troubleshooting

### Model Not Loading

**Problem**: `Model not loaded` error

**Solutions**:
1. Check MODEL_DIR path: `echo $MODEL_DIR`
2. Verify model files exist: `ls -la $MODEL_DIR`
3. Check logs for detailed error messages

### CUDA Out of Memory

**Problem**: GPU memory errors

**Solutions**:
1. Reduce batch size (not applicable for inference)
2. Use smaller model (Qwen-0.5B instead of Phi-2)
3. Enable model quantization (add to future versions)

### Slow Inference

**Problem**: Slow generation times

**Solutions**:
1. Use GPU instance instead of CPU
2. Reduce `max_length` parameter
3. Use smaller/faster model
4. Enable FP16 precision (auto-enabled on GPU)

## License

This project is for educational purposes as part of the ESports Strategy Summarizer academic project.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review test client examples
3. Verify model training completed successfully
4. Check Flask logs for detailed error messages
