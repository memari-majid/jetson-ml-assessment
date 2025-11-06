#!/bin/bash
# UVU GB10 Chatbot Deployment Script
# Deploys chatbot with custom ngrok domain

echo "╔═══════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                           ║"
echo "║              UVU GB10 CHATBOT - DEPLOYMENT SCRIPT                        ║"
echo "║                                                                           ║"
echo "╚═══════════════════════════════════════════════════════════════════════════╝"

cd /home/majid/Downloads/jetson-ml-assessment

# Activate virtual environment
echo ""
echo "1️⃣  Activating virtual environment..."
source venv/bin/activate
echo "✅ Virtual environment activated"

# Check GPU
echo ""
echo "2️⃣  Checking GPU availability..."
python3 -c "import torch; print(f'  GPU: {torch.cuda.get_device_name(0)}'); print(f'  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ GPU ready"
else
    echo "⚠️  GPU not available, will use CPU"
fi

# Start chatbot in background
echo ""
echo "3️⃣  Starting chatbot application..."
echo "  This will download the model on first run (~14 GB)"
echo "  Model: Mistral-7B-Instruct-v0.2 (state-of-the-art)"
echo ""

# Start the chatbot
python3 gb10_chatbot.py > chatbot.log 2>&1 &
CHATBOT_PID=$!

echo "✅ Chatbot started (PID: $CHATBOT_PID)"
echo "  Waiting for server to be ready..."
sleep 10

# Check if chatbot is running
if ps -p $CHATBOT_PID > /dev/null; then
    echo "✅ Chatbot server is running"
else
    echo "❌ Chatbot failed to start. Check chatbot.log"
    cat chatbot.log
    exit 1
fi

# Get the local URL
echo ""
echo "4️⃣  Chatbot URLs:"
echo "  Local:  http://localhost:7860"
echo "  LAN:    http://$(hostname -I | awk '{print $1}'):7860"

# Check for gradio share URL in log
sleep 5
if grep -q "Running on public URL" chatbot.log; then
    PUBLIC_URL=$(grep "Running on public URL" chatbot.log | grep -oP 'https://[^ ]+')
    echo "  Public: $PUBLIC_URL"
fi

echo ""
echo "5️⃣  Setting up ngrok with custom domain..."
echo ""
echo "⚠️  IMPORTANT: Custom ngrok domains require:"
echo "  1. Paid ngrok account (free tier doesn't support custom domains)"
echo "  2. Domain configured in ngrok dashboard: uvuchatbot.ngrok.app"
echo "  3. ngrok authtoken set"
echo ""

# Check for ngrok
if ! command -v ngrok &> /dev/null; then
    echo "❌ ngrok not found. Installing..."
    
    # Download and install ngrok
    wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-arm64.tgz -O /tmp/ngrok.tgz
    tar xvzf /tmp/ngrok.tgz -C /tmp
    sudo mv /tmp/ngrok /usr/local/bin/
    rm /tmp/ngrok.tgz
    
    echo "✅ ngrok installed"
fi

# Check for authtoken
if [ -z "$NGROK_AUTHTOKEN" ]; then
    echo ""
    echo "⚠️  NGROK_AUTHTOKEN not set."
    echo ""
    echo "To use custom domain, run:"
    echo "  1. Get your authtoken from: https://dashboard.ngrok.com/get-started/your-authtoken"
    echo "  2. Set it: export NGROK_AUTHTOKEN='your_token_here'"
    echo "  3. Configure domain 'uvuchatbot.ngrok.app' in ngrok dashboard"
    echo "  4. Re-run this script"
    echo ""
    echo "For now, starting with random ngrok URL..."
    
    # Start ngrok without custom domain
    ngrok http 7860 > ngrok.log 2>&1 &
    NGROK_PID=$!
    
else
    echo "✅ NGROK_AUTHTOKEN found"
    ngrok config add-authtoken $NGROK_AUTHTOKEN
    
    echo ""
    echo "🚀 Starting ngrok with custom domain: uvuchatbot.ngrok.app"
    
    # Start ngrok with custom domain
    # Note: Custom domains use different syntax
    ngrok http --url=uvuchatbot.ngrok.app 7860 > ngrok.log 2>&1 &
    NGROK_PID=$!
fi

echo "  ngrok PID: $NGROK_PID"
sleep 5

# Get ngrok URL
echo ""
echo "6️⃣  ngrok tunnel established!"

# Try to get URL from ngrok API
NGROK_URL=$(curl -s http://localhost:4040/api/tunnels 2>/dev/null | grep -oP '"public_url":"https://[^"]+' | head -1 | cut -d'"' -f4)

if [ -n "$NGROK_URL" ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "  🎉 CHATBOT IS LIVE! 🎉"
    echo ""
    echo "  Public URL:  $NGROK_URL"
    echo "  Local URL:   http://localhost:7860"
    echo ""
    echo "  Share this URL with students and users!"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
else
    echo "  Check ngrok.log for URL or visit http://localhost:4040 (ngrok dashboard)"
fi

echo ""
echo "📊 System Information:"
echo "  Chatbot PID: $CHATBOT_PID"
echo "  ngrok PID: $NGROK_PID"
echo "  Logs: chatbot.log, ngrok.log"
echo ""
echo "To stop:"
echo "  kill $CHATBOT_PID $NGROK_PID"
echo ""
echo "Or use: pkill -f gb10_chatbot"
echo ""
echo "✅ Deployment complete!"
echo ""

