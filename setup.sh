#!/bin/bash
# Self-Evolving Game Agent - Terminal Setup Script

# 1. Set your OpenAI API Key here if it's not already in your ~/.zshrc or ~/.bash_profile
# export OPENAI_API_KEY="your-api-key-here"

# 2. Activate the virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "✅ Virtual environment (.venv) activated."
else
    echo "⚠️  .venv directory not found. Creating it now..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    echo "✅ .venv created and activated."
fi

# 3. Verify API Key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ WARNING: OPENAI_API_KEY is not set. Evolution mode will fail."
    echo "   Please run: export OPENAI_API_KEY='your-key'"
else
    echo "✅ OPENAI_API_KEY is set."
fi

# 4. Add alias for convenience (Mac/Zsh focus)
alias py='python'

echo "--------------------------------------------------------"
echo "🚀 Setup complete! You can now run commands like:"
echo "   python main.py --game 2048 --mode evolve"
echo "--------------------------------------------------------"
