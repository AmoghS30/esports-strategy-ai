# Quick Reference Guide

## 🚀 Common Commands

### Starting the Server
```bash
# Simple start
python esports_backend.py

# Using startup script
./start_server.sh

# Production mode
gunicorn -w 4 -b 0.0.0.0:5000 esports_backend:app
```

### Testing
```bash
# Run all tests
python test_api.py

# Quick health check
curl http://localhost:5000/health

# Get config
curl http://localhost:5000/api/config
```

## 📝 Quick API Examples

### 1. Simple Summary
```bash
curl -X POST http://localhost:5000/api/summarize \
  -H "Content-Type: application/json" \
  -d '{
    "match_data": {
      "team_a": "Team A",
      "team_b": "Team B",
      "winner": "Team A",
      "events": ["Event 1", "Event 2"]
    },
    "style": "analytical"
  }'
```

### 2. Get Recommendations
```bash
curl -X POST http://localhost:5000/api/recommendations \
  -H "Content-Type: application/json" \
  -d '{
    "match_data": {
      "team_a": "Team A",
      "team_b": "Team B",
      "winner": "Team B"
    },
    "team": "team_a",
    "recommendation_depth": 3
  }'
```

### 3. Complete Analysis
```bash
curl -X POST http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "match_data": {
      "team_a": "Team A",
      "team_b": "Team B",
      "winner": "Team A"
    },
    "style": "tactical",
    "focus": "objectives"
  }'
```

## 🔧 Configuration Quick Reference

### Environment Variables
```bash
export MODEL_DIR="./models/esports-gpu-fast_final"
export TEMPERATURE="0.7"
export MAX_LENGTH="512"
```

### Summary Styles
- `analytical` - Detailed analysis
- `narrative` - Story-like
- `bullet` - Quick points
- `tactical` - Strategic depth

### Focus Areas
- `team_fights` - Combat focus
- `economy` - Resources
- `objectives` - Map control
- `composition` - Draft
- `macro` - Strategy

### Length Options
- `short` - 2-3 sentences
- `medium` - 1-2 paragraphs
- `long` - 3-4 paragraphs

## 🐛 Quick Troubleshooting

### Model Won't Load
```bash
# Check path
ls -la ./models/esports-gpu-fast_final

# Set environment
export MODEL_DIR="./models/esports-gpu-fast_final"
```

### GPU Not Working
```bash
# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### Port Already in Use
```bash
# Kill existing process
lsof -ti:5000 | xargs kill -9

# Or use different port
PORT=5001 python esports_backend.py
```

## 📦 Installation Shortcuts

### One-Line Install
```bash
pip install flask flask-cors torch transformers peft datasets accelerate
```

### Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

## 🎯 Python Quick Examples

### Minimal Example
```python
import requests

response = requests.post(
    "http://localhost:5000/api/summarize",
    json={
        "match_data": {
            "team_a": "A",
            "team_b": "B",
            "winner": "A"
        }
    }
)
print(response.json()["summary"])
```

### With Error Handling
```python
try:
    response = requests.post(url, json=data)
    response.raise_for_status()
    result = response.json()
    print(result["summary"])
except requests.exceptions.RequestException as e:
    print(f"Error: {e}")
```

## 📊 Response Formats

### Success Response
```json
{
  "success": true,
  "summary": "Match analysis...",
  "metadata": {
    "style": "analytical",
    "timestamp": "2024-01-01T12:00:00"
  }
}
```

### Error Response
```json
{
  "error": "Error message"
}
```

## 🔑 Key Parameters

| Parameter | Type | Default | Options |
|-----------|------|---------|---------|
| style | string | analytical | analytical, narrative, bullet, tactical |
| focus | string | team_fights | team_fights, economy, objectives, composition, macro |
| length | string | medium | short, medium, long |
| temperature | float | 0.7 | 0.1 - 2.0 |
| max_length | int | 512 | 128 - 2048 |

## 💡 Pro Tips

1. **Better Results**: Include more events and commentary
2. **Faster Inference**: Use GPU and lower max_length
3. **Creative Output**: Increase temperature to 0.9-1.0
4. **Consistent Output**: Lower temperature to 0.3-0.5
5. **Best Performance**: Use Qwen-0.5B on GPU (~120 samples/sec)

## 📱 Testing Tools

### cURL
```bash
curl -X POST http://localhost:5000/api/summarize \
  -H "Content-Type: application/json" \
  -d @match_data.json
```

### Postman
Import: `ESports_API.postman_collection.json`

### Python
Run: `python test_api.py`

## 🚀 Deployment Checklist

- [ ] Model trained and saved
- [ ] Dependencies installed
- [ ] Environment variables set
- [ ] Server starts without errors
- [ ] Health endpoint responds
- [ ] Test client passes all tests
- [ ] GPU detected (if using)
- [ ] Production server configured (Gunicorn)
- [ ] Reverse proxy setup (NGINX)
- [ ] SSL certificate installed

## 📞 Quick Links

- **Main README**: See `README.md` for full documentation
- **API Examples**: See `API_EXAMPLES.md` for detailed examples
- **Project Summary**: See `PROJECT_SUMMARY.md` for overview
- **Test Client**: Run `python test_api.py`
- **Postman**: Import `ESports_API.postman_collection.json`

---

**Need Help?** Check the full README.md for detailed troubleshooting!
