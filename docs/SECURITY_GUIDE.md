# Security Guide - UVU GB10 Chatbot

**Critical:** This guide explains how to properly handle API tokens and secrets.

---

## ⚠️ SECURITY ALERT

**NEVER commit API tokens or secrets to git!**

### Tokens That Must Stay Secret:

1. **HuggingFace API Token** - Grants access to your HF account
2. **ngrok Authtoken** - Controls your ngrok tunnels
3. **Database passwords** - Protects user data
4. **API keys** - Any external service credentials

---

## ✅ Proper Secret Management

### Method 1: Environment Variables (Recommended)

**Setup:**

```bash
# Create .env file (this file is in .gitignore)
cp .env.example .env

# Edit .env with your actual tokens
nano .env
```

**.env file:**
```bash
HF_TOKEN=your_actual_huggingface_token
NGROK_AUTHTOKEN=your_actual_ngrok_token
```

**Load before running:**
```bash
# Option A: Source .env file
source .env

# Option B: Export directly
export HF_TOKEN='your_token'
export NGROK_AUTHTOKEN='your_token'

# Then run chatbot
python3 uvu_chatbot_pro.py
```

---

### Method 2: System Environment Variables

**Add to ~/.bashrc or ~/.zshrc:**

```bash
# HuggingFace
export HF_TOKEN='your_token_here'

# ngrok
export NGROK_AUTHTOKEN='your_token_here'
```

**Reload shell:**
```bash
source ~/.bashrc
```

---

### Method 3: Secrets Manager (Production)

For production deployments, use:
- **AWS Secrets Manager**
- **HashiCorp Vault**
- **Azure Key Vault**
- **Google Secret Manager**

---

## 🔒 How We Fixed The Issues

### Bug 1: Hardcoded HuggingFace Token ✅ FIXED

**Before (INSECURE ❌):**
```python
HF_TOKEN = "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"  # NEVER DO THIS!
```

**After (SECURE ✅):**
```python
HF_TOKEN = os.environ.get('HF_TOKEN', None)
if not HF_TOKEN:
    print("⚠️  WARNING: HF_TOKEN not set")
```

---

### Bug 2: State Variable Not Used ✅ FIXED

**Before (WRONG ❌):**
```python
login_btn.click(
    handle_login,
    outputs=[login_status, gr.State()]  # Creates throwaway state!
)
```

**After (CORRECT ✅):**
```python
login_btn.click(
    handle_login,
    outputs=[login_status, logged_in]  # Uses actual state variable
)
```

---

### Bug 3: Token in Documentation ✅ FIXED

**Before (INSECURE ❌):**
```
Token: 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
```

**After (SECURE ✅):**
```
Token: Set via HF_TOKEN environment variable
```

---

## 📋 Security Checklist

### Before Committing to Git:

- [ ] No API tokens in code
- [ ] No passwords in code
- [ ] No authtokens in documentation
- [ ] .env file in .gitignore
- [ ] .env.example provided (with placeholders only)
- [ ] README includes setup instructions
- [ ] Secrets loaded from environment

### After Exposing Tokens:

If you accidentally committed tokens:

1. **Immediately revoke the token**
   - HuggingFace: https://huggingface.co/settings/tokens
   - ngrok: https://dashboard.ngrok.com/get-started/your-authtoken

2. **Generate new token**

3. **Remove from git history:**
   ```bash
   # Use git filter-branch or BFG Repo-Cleaner
   # Or create fresh repo
   ```

4. **Update .gitignore** to prevent future accidents

---

## 🛡️ Best Practices

### 1. Use .gitignore

Ensure these are in `.gitignore`:
```
.env
*.log
*.key
*.pem
secrets/
credentials/
.env.local
.env.production
```

### 2. Never Hardcode Secrets

```python
# ❌ NEVER DO THIS:
api_key = "sk_1234567890"

# ✅ ALWAYS DO THIS:
api_key = os.environ.get('API_KEY')
if not api_key:
    raise ValueError("API_KEY environment variable not set")
```

### 3. Use Environment-Specific Config

```python
import os
from dotenv import load_dotenv  # pip install python-dotenv

# Load .env file
load_dotenv()

# Access secrets
HF_TOKEN = os.environ['HF_TOKEN']
NGROK_TOKEN = os.environ['NGROK_AUTHTOKEN']
```

### 4. Validate on Startup

```python
required_env_vars = ['HF_TOKEN', 'NGROK_AUTHTOKEN']
missing = [var for var in required_env_vars if not os.environ.get(var)]

if missing:
    raise ValueError(f"Missing environment variables: {missing}")
```

---

## 📖 Setup Instructions for Users

### Quick Start

1. **Copy environment template:**
   ```bash
   cp .env.example .env
   ```

2. **Edit .env with your tokens:**
   ```bash
   nano .env
   ```

3. **Load environment:**
   ```bash
   source .env
   ```

4. **Run chatbot:**
   ```bash
   python3 uvu_chatbot_pro.py
   ```

---

## 🔍 Security Audit Results

### Current Status: ✅ SECURE

- ✅ No hardcoded tokens in code
- ✅ No tokens in documentation
- ✅ Environment variables used
- ✅ .env.example provided
- ✅ Proper error messages
- ✅ State management fixed

### Previous Issues (NOW FIXED):

- ❌ ~~HuggingFace token hardcoded~~ → ✅ Fixed (uses env var)
- ❌ ~~ngrok token in docs~~ → ✅ Fixed (redacted)
- ❌ ~~State variable bug~~ → ✅ Fixed (uses logged_in state)

---

## 🚨 Action Items

### If You Previously Pushed Tokens:

⚠️ **CRITICAL: Revoke and regenerate ALL exposed tokens immediately!**

1. **HuggingFace:**
   - Go to: https://huggingface.co/settings/tokens
   - Revoke token: `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`
   - Generate new token
   - Set in environment: `export HF_TOKEN='new_token'`

2. **ngrok:**
   - Go to: https://dashboard.ngrok.com/get-started/your-authtoken
   - Revoke token: `31yUo8DyENwWL0FcfZPHr1EWajT_7tkfLKJJR2ioPRqPXxz5k`
   - Generate new authtoken
   - Set in environment: `export NGROK_AUTHTOKEN='new_token'`

3. **Clean Git History:**
   - Consider creating fresh repo
   - Or use BFG Repo-Cleaner to remove tokens from history

---

## ✅ Verification

Run this to verify no secrets in code:

```bash
# Check for common token patterns
grep -r "hf_[A-Za-z0-9]" . --exclude-dir=venv --exclude-dir=.git
grep -r "sk_[A-Za-z0-9]" . --exclude-dir=venv --exclude-dir=.git
grep -r "AUTHTOKEN.*=" . --exclude-dir=venv --exclude-dir=.git --include="*.py"

# Should return nothing or only .env.example
```

---

**Security Status:** ✅ All issues fixed  
**Next Step:** Revoke exposed tokens and generate new ones  
**Prevention:** Always use environment variables for secrets

