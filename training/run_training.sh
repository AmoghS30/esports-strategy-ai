#!/bin/bash

# ESports Strategy AI - Training Runner
# This script runs the complete training pipeline

echo "============================================================"
echo "    ESports Strategy AI - Training Pipeline"
echo "============================================================"
echo ""

# Activate virtual environment if exists
if [ -d "../venv" ]; then
    source ../venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

# Step 1: Prepare Data
echo "📊 Step 1: Preparing training data..."
python prepare_data.py --samples 500

if [ $? -ne 0 ]; then
    echo "❌ Data preparation failed!"
    exit 1
fi

echo ""
echo "✅ Data preparation complete!"
echo ""

# Step 2: Train Model
echo "🏋️ Step 2: Training model..."
echo "   This will take a while depending on your GPU..."
echo ""

python train.py \
    --model gemma-2b \
    --epochs 3 \
    --batch-size 2 \
    --lr 2e-4 \
    --lora-r 16

if [ $? -ne 0 ]; then
    echo "❌ Training failed!"
    exit 1
fi

echo ""
echo "✅ Training complete!"
echo ""

# Step 3: Evaluate
echo "📊 Step 3: Evaluating model..."

# Find the latest model
LATEST_MODEL=$(ls -td ./models/esports-lora_final 2>/dev/null | head -1)

if [ -z "$LATEST_MODEL" ]; then
    LATEST_MODEL=$(ls -td ./models/esports-lora_* 2>/dev/null | head -1)
fi

if [ -n "$LATEST_MODEL" ]; then
    python evaluate.py --model "$LATEST_MODEL" --samples 20
else
    echo "⚠️  No trained model found to evaluate"
fi

echo ""
echo "============================================================"
echo "    Training Pipeline Complete!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "1. Check evaluation results in evaluation_results.json"
echo "2. Update .env with LOCAL_MODEL_PATH=$LATEST_MODEL"
echo "3. Set LLM_BACKEND=local in .env"
echo "4. Restart the backend server"
echo ""
