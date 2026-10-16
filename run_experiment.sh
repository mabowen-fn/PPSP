#!/bin/bash

# Setup and run PPSP experiment with local Ollama

echo "Setting up PPSP experiment environment..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Check connectivity to local services
echo ""
echo "Checking local services..."
echo -n "Ollama (10.50.2.40:11434): "
curl -s -o /dev/null -w "%{http_code}\n" http://10.50.2.40:11434/ || echo "FAILED"

echo -n "HarmBench Judge (10.50.2.39:18082): "
curl -s -o /dev/null -w "%{http_code}\n" http://10.50.2.39:18082/health || echo "FAILED"

# Create necessary directories
mkdir -p data/harmbench results

echo ""
echo "Running Experiment 1: PPSP vs Response Attack"
python experiments/exp1_ppsp_vs_ra.py --num-behaviors 5

echo ""
echo "Experiment complete!"
