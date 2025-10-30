@echo off
REM Environment Setup for Model Welfare Experiments (Windows)
REM Usage: setup.bat

echo === Model Welfare Experiments Setup ===
echo.

REM Check Python version
python --version 2>nul | findstr /C:"3.11" >nul
if errorlevel 1 (
    echo Error: Python 3.11 is required but not found.
    echo Please install Python 3.11 before running this script.
    exit /b 1
)

echo [OK] Python 3.11 found

REM Create virtual environment
echo Creating virtual environment...
python -m venv reflexion_env

REM Activate virtual environment
echo Activating virtual environment...
call reflexion_env\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo Installing dependencies...
pip install -r requirements.txt

REM Download spacy model
echo Downloading spacy language model...
python -m spacy download en_core_web_sm

REM Create .env file if it doesn't exist
if not exist .env (
    echo Creating .env file...
    (
        echo # API Keys - Replace with your actual keys
        echo OPENAI_API_KEY=your_openai_key_here
        echo ANTHROPIC_API_KEY=your_anthropic_key_here
        echo.
        echo # Model Configuration
        echo MODEL_NAME=gpt-4
        echo TEMPERATURE=0.7
        echo MAX_TOKENS=500
        echo.
        echo # Logging
        echo LOG_LEVEL=DEBUG
        echo LOG_FILE=welfare_signals.log
        echo.
        echo # Experiment Settings
        echo MAX_ITERATIONS=3
        echo BATCH_SIZE=10
    ) > .env
    echo [OK] Created .env file - please update with your API keys
) else (
    echo [OK] .env file already exists
)

REM Create necessary directories
echo Creating directory structure...
if not exist results\baseline mkdir results\baseline
if not exist results\modified mkdir results\modified
if not exist results\analysis mkdir results\analysis
if not exist logs mkdir logs
if not exist data\benchmarks mkdir data\benchmarks

echo.
echo === Setup Complete ===
echo.
echo Next steps:
echo 1. Edit .env file with your API keys
echo 2. Activate the virtual environment: reflexion_env\Scripts\activate.bat
echo 3. Run the pipeline: python run_pipeline.py --help
echo.
pause
