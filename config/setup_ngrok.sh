#!/bin/bash
# Setup ngrok with custom domain for UVU GB10 Chatbot

echo "╔═══════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                           ║"
echo "║              NGROK SETUP - UVU CHATBOT CUSTOM DOMAIN                     ║"
echo "║                                                                           ║"
echo "╚═══════════════════════════════════════════════════════════════════════════╝"

# Get ngrok token from environment variable (NEVER commit tokens to git!)
NGROK_TOKEN="${NGROK_AUTHTOKEN:-}"
CUSTOM_DOMAIN="uvuchatbot.ngrok.app"

if [ -z "$NGROK_TOKEN" ]; then
    echo "❌ ERROR: NGROK_AUTHTOKEN environment variable not set"
    echo ""
    echo "Please set your ngrok authtoken:"
    echo "  export NGROK_AUTHTOKEN='your_token_here'"
    echo ""
    echo "Get your token from: https://dashboard.ngrok.com/get-started/your-authtoken"
    exit 1
fi

echo ""
echo "1️⃣  Installing ngrok..."

# Check if ngrok is installed
if command -v ngrok &> /dev/null; then
    echo "✅ ngrok already installed"
else
    echo "  Downloading ngrok for ARM64..."
    wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-arm64.tgz -O /tmp/ngrok.tgz
    tar xvzf /tmp/ngrok.tgz -C /tmp
    sudo mv /tmp/ngrok /usr/local/bin/
    rm /tmp/ngrok.tgz
    echo "✅ ngrok installed to /usr/local/bin/ngrok"
fi

echo ""
echo "2️⃣  Configuring ngrok authtoken..."

# Add authtoken
ngrok config add-authtoken $NGROK_TOKEN

if [ $? -eq 0 ]; then
    echo "✅ Authtoken configured"
else
    echo "❌ Failed to configure authtoken"
    exit 1
fi

echo ""
echo "3️⃣  Verifying configuration..."

# Check config
ngrok config check

if [ $? -eq 0 ]; then
    echo "✅ Configuration valid"
else
    echo "⚠️  Configuration may have issues"
fi

echo ""
echo "4️⃣  Testing ngrok connection..."

# Test ngrok
timeout 5 ngrok diagnose || true

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "✅ NGROK SETUP COMPLETE"
echo ""
echo "Your ngrok is configured with:"
echo "  Authtoken: ${NGROK_TOKEN:0:20}..."
echo "  Custom domain: $CUSTOM_DOMAIN"
echo ""
echo "To start chatbot with custom domain:"
echo ""
echo "  Option 1: Using ngrok CLI (recommended for custom domain)"
echo "    # Terminal 1: Start chatbot"
echo "    cd /home/majid/Downloads/jetson-ml-assessment"
echo "    source venv/bin/activate"
echo "    python3 scripts/uvu_chatbot_pro.py"
echo ""
echo "    # Terminal 2: Start ngrok tunnel"
echo "    ngrok http --url=$CUSTOM_DOMAIN 7860"
echo ""
echo "  Option 2: Auto-launch (generates random domain)"
echo "    cd /home/majid/Downloads/jetson-ml-assessment"
echo "    source venv/bin/activate"
echo "    python3 scripts/uvu_chatbot_pro.py"
echo "    (Gradio will create automatic share link)"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "⚠️  IMPORTANT: For custom domain (uvuchatbot.ngrok.app) to work:"
echo "  1. Your ngrok account must have a paid plan"
echo "  2. Domain must be configured in ngrok dashboard"
echo "  3. Use separate ngrok CLI command (Option 1 above)"
echo ""
echo "🚀 Ready to deploy!"
echo ""

