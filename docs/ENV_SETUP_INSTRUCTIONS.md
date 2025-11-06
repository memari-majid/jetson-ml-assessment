# Environment Setup Instructions

**IMPORTANT:** This chatbot requires API tokens that must NEVER be committed to git.

---

## 🔐 Required Tokens

### 1. HuggingFace API Token

**Purpose:** Access LLM models (Llama, Mistral, etc.)

**Get your token:**
1. Go to: https://huggingface.co/settings/tokens
2. Create new token (read permission is enough)
3. Copy the token (starts with `hf_`)

---

### 2. ngrok Authtoken (Optional)

**Purpose:** Public URL with custom domain

**Get your token:**
1. Go to: https://dashboard.ngrok.com/get-started/your-authtoken
2. Copy your authtoken
3. For custom domains, you need paid plan

---

## ⚙️ Setup Methods

### Method 1: Use .env File (Recommended)

**Step 1: Create .env file**
```bash
cd /home/majid/Downloads/jetson-ml-assessment

# Create .env file with your tokens
cat > .env << 'EOF'
# HuggingFace Token
HF_TOKEN=your_actual_hf_token_here

# ngrok Token (optional, for custom domain)
NGROK_AUTHTOKEN=your_actual_ngrok_token_here
EOF
```

**Step 2: Load environment**
```bash
# Load .env before running
source .env

# Run chatbot
python3 uvu_chatbot_pro.py
```

---

### Method 2: Export in Terminal

```bash
# Set tokens for current session
export HF_TOKEN='your_hf_token_here'
export NGROK_AUTHTOKEN='your_ngrok_token_here'

# Run chatbot
python3 uvu_chatbot_pro.py
```

---

### Method 3: Add to Shell Profile (Permanent)

```bash
# Edit your shell profile
nano ~/.bashrc  # or ~/.zshrc

# Add these lines:
export HF_TOKEN='your_hf_token_here'
export NGROK_AUTHTOKEN='your_ngrok_token_here'

# Save and reload
source ~/.bashrc
```

---

## 🚀 Quick Start Script

Create a startup script:

```bash
cat > start_chatbot.sh << 'EOF'
#!/bin/bash
cd /home/majid/Downloads/jetson-ml-assessment

# Load environment
if [ -f .env ]; then
    source .env
    echo "✅ Environment loaded"
else
    echo "❌ .env file not found! Create it first."
    exit 1
fi

# Verify tokens are set
if [ -z "$HF_TOKEN" ]; then
    echo "❌ HF_TOKEN not set in .env"
    exit 1
fi

# Activate venv
source venv/bin/activate

# Start chatbot
python3 uvu_chatbot_pro.py
EOF

chmod +x start_chatbot.sh
```

**Then run:**
```bash
./start_chatbot.sh
```

---

## ✅ Verification

**Check tokens are set:**
```bash
echo "HF_TOKEN: ${HF_TOKEN:0:10}..."  # Shows first 10 chars
echo "NGROK_AUTHTOKEN: ${NGROK_AUTHTOKEN:0:10}..."
```

**Should show:**
```
HF_TOKEN: hf_GCJV...
NGROK_AUTHTOKEN: 31yUo8D...
```

---

## 🛡️ Security Notes

**Your .env file is automatically ignored by git** (see `.gitignore`)

**NEVER:**
- ❌ Commit .env to git
- ❌ Share tokens in chat/email
- ❌ Hardcode tokens in code
- ❌ Include tokens in documentation
- ❌ Screenshot tokens
- ❌ Post tokens online

**ALWAYS:**
- ✅ Use environment variables
- ✅ Keep tokens in .env (git-ignored)
- ✅ Revoke if exposed
- ✅ Use read-only permissions when possible
- ✅ Rotate tokens periodically

---

## 📋 Current Setup Status

After following these instructions:

```bash
# Your setup should look like:
/home/majid/Downloads/jetson-ml-assessment/
├── .env                    # Your tokens (NOT in git)
├── .env.example           # Template (IN git)  
├── .gitignore             # Protects .env (IN git)
├── uvu_chatbot_pro.py     # Uses env vars (IN git)
└── start_chatbot.sh       # Startup script (IN git)
```

**git status should NOT show .env** ✅

---

**Security Status:** ✅ Properly configured  
**Next Step:** Set your actual tokens in .env file  
**Ready:** Yes, once tokens are configured

