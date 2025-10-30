#!/bin/bash
# Environment Setup for Model Welfare Experiments
# Usage: ./setup.sh

set -e  # Exit on error

echo "=== Model Welfare Experiments Setup ==="
echo ""

# Check Python version
if ! command -v python3.11 &> /dev/null; then
    echo "Error: Python 3.11 is required but not found."
    echo "Please install Python 3.11 before running this script."
    exit 1
fi

echo "✓ Python 3.11 found"

# Create virtual environment
echo "Creating virtual environment..."
python3.11 -m venv reflexion_env

# Activate virtual environment
echo "Activating virtual environment..."
source reflexion_env/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

# Download spacy model
echo "Downloading spacy language model..."
python -m spacy download en_core_web_sm

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cat > .env << EOF
# API Keys - Replace with your actual keys
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here

# Model Configuration
MODEL_NAME=gpt-4
TEMPERATURE=0.7
MAX_TOKENS=500

# Logging
LOG_LEVEL=DEBUG
LOG_FILE=welfare_signals.log

# Experiment Settings
MAX_ITERATIONS=3
BATCH_SIZE=10
EOF
    echo "✓ Created .env file - please update with your API keys"
else
    echo "✓ .env file already exists"
fi

# Create necessary directories
echo "Creating directory structure..."
mkdir -p results/{baseline,modified,analysis}
mkdir -p logs
mkdir -p data/benchmarks

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "1. Edit .env file with your API keys"
echo "2. Activate the virtual environment: source reflexion_env/bin/activate"
echo "3. Run the pipeline: python run_pipeline.py --help"
echo ""
