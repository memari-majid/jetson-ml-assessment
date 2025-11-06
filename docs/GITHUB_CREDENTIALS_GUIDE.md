# GitHub Credentials Setup Guide

This document shows where your GitHub credentials are stored and how to use them across all projects on this machine.

---

## 📍 Where Your Credentials Are Stored

### 1. **Personal Access Token (PAT)**

**Location 1: Cursor MCP Configuration**
- **File**: `~/.cursor/mcp.json`
- **Token**: `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`
- **Purpose**: Used by Cursor's AI to interact with GitHub API
- **Scope**: Cursor IDE only

**Location 2: Git Credential Store (HTTPS)**
- **File**: `~/.git-credentials` (will be created on first use)
- **Configuration**: `git config --global credential.helper store`
- **Purpose**: Automatic authentication for HTTPS git operations
- **Scope**: All git commands system-wide

### 2. **SSH Key**

**Public Key**:
```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIHYtkd81LxFLP4w9RgiPVPXCuOW9Vcq3KqyLHpUKIrjU jetson-ml-assessment
```

**Configuration Files**:
- **Private Key**: `~/.ssh/jetson_ml_key`
- **Public Key**: `~/.ssh/jetson_ml_key.pub`
- **SSH Config**: `~/.ssh/config`

---

## 🔧 How to Use Your Credentials

### Method 1: SSH (Recommended - Most Secure)

#### Step 1: Add SSH Key to GitHub (One-time Setup)

1. **Copy your public key**:
```bash
cat ~/.ssh/jetson_ml_key.pub
```

2. **Add to GitHub**:
   - Go to: https://github.com/settings/ssh/new
   - Paste the public key
   - Give it a name: "Jetson ML Machine"
   - Click "Add SSH key"

#### Step 2: Test SSH Connection
```bash
ssh -T git@github.com
```

You should see: `Hi username! You've successfully authenticated...`

#### Step 3: Use SSH URLs for All Git Operations

**Clone a repository**:
```bash
git clone git@github.com:username/repo.git
```

**For existing repositories with HTTPS URLs, switch to SSH**:
```bash
cd /path/to/your/repo
git remote set-url origin git@github.com:username/repo.git
```

**Check current remote**:
```bash
git remote -v
```

---

### Method 2: HTTPS with Token

#### Setup (Already Done!)
```bash
git config --global credential.helper store
```

#### How to Use

1. **First time using HTTPS** - When you clone/push/pull, Git will ask:
   ```
   Username: your-github-username
   Password: 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
   ```

2. **After first use** - Credentials are saved in `~/.git-credentials` and used automatically

**Example**:
```bash
git clone https://github.com/username/repo.git
cd repo
git add .
git commit -m "Update"
git push  # No password needed after first time!
```

---

## 🎯 Quick Setup Script

To automatically store your HTTPS credentials without waiting for first push:

```bash
# Store credentials in git credential store
echo "https://YOUR_GITHUB_USERNAME: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@github.com" > ~/.git-credentials
chmod 600 ~/.git-credentials
```

**Replace `YOUR_GITHUB_USERNAME` with your actual GitHub username!**

---

## 🔍 View/Manage Your Credentials

### View Token in MCP Config
```bash
cat ~/.cursor/mcp.json | grep GITHUB_PERSONAL_ACCESS_TOKEN
```

### View Stored Git Credentials
```bash
cat ~/.git-credentials
```

### View SSH Configuration
```bash
cat ~/.ssh/config
cat ~/.ssh/jetson_ml_key.pub
```

### Check Git Configuration
```bash
git config --global --list
```

---

## 🌐 Using Across All Projects

### Option A: SSH (Automatic for all repos)
- ✅ Already configured via `~/.ssh/config`
- ✅ Works for ALL repositories once public key is added to GitHub
- ✅ No need to enter credentials ever
- 🔒 Most secure method

### Option B: HTTPS (Automatic after first use)
- ✅ Already configured via `git config --global credential.helper store`
- ✅ Works for ALL repositories
- ✅ Credentials stored once in `~/.git-credentials`
- ⚠️ Token stored in plain text (but file is readable only by you)

---

## 🔐 Security Best Practices

### Current Setup
- ✅ SSH private key has correct permissions (600)
- ✅ SSH config has correct permissions (600)
- ✅ Git credential store will have correct permissions (600)
- ✅ Token is stored in your home directory (not in project files)

### Recommendations
1. **Never commit** credentials to Git repositories
2. **Use SSH** for maximum security
3. **Rotate tokens** periodically on GitHub
4. **Keep your private key safe** - never share `~/.ssh/jetson_ml_key`

---

## 📝 Common Commands

### Clone Repository
```bash
# SSH (recommended)
git clone git@github.com:username/repo.git

# HTTPS
git clone https://github.com/username/repo.git
```

### Check Authentication
```bash
# SSH test
ssh -T git@github.com

# HTTPS test (will use stored credentials)
git ls-remote https://github.com/username/repo.git
```

### Convert Existing Repo from HTTPS to SSH
```bash
cd /path/to/repo
git remote set-url origin git@github.com:username/repo.git
```

---

## 🆘 Troubleshooting

### SSH Not Working?
```bash
# Test connection
ssh -T git@github.com

# Check SSH key is loaded
ssh-add -l

# Add key manually if needed
ssh-add ~/.ssh/jetson_ml_key
```

### HTTPS Not Working?
```bash
# Check credential helper is set
git config --global credential.helper

# Should output: store

# Check if credentials are saved
cat ~/.git-credentials
```

### Token Expired?
1. Generate new token at: https://github.com/settings/tokens
2. Update in `~/.cursor/mcp.json`
3. Update in `~/.git-credentials` (or delete file and re-enter on next push)

---

## 📊 Summary

Your GitHub credentials work **system-wide** for all projects through:

1. **SSH**: `~/.ssh/config` + `~/.ssh/jetson_ml_key`
2. **HTTPS**: `git credential.helper store` + `~/.git-credentials`
3. **Cursor AI**: `~/.cursor/mcp.json` with your PAT

**You can use GitHub in any project directory on this machine!** 🎉

---

*Last Updated: November 6, 2025*
*Machine: Jetson ML (Linux 6.11.0-1016-nvidia)*

