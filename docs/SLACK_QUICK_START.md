# Slack Integration - Quick Start

**Ready to use!** Just need a Slack Bot Token.

---

## 🚀 Setup (5 Minutes)

### Step 1: Get Your Slack Bot Token

1. Go to https://api.slack.com/apps
2. Click "Create New App" → "From scratch"
3. Name it (e.g., "ML Bot") and choose workspace
4. Go to "OAuth & Permissions"
5. Add Bot Token Scopes:
   - `channels:read`
   - `channels:write`  
   - `chat:write`
   - `users:read`
   - `groups:read`
   - `files:write`
6. Click "Install to Workspace"
7. Copy the "Bot User OAuth Token" (starts with `xoxb-`)

### Step 2: Set Environment Variable

```bash
export SLACK_BOT_TOKEN='xoxb-your-token-here'
export SLACK_CHANNEL='general'  # Optional, default is 'general'
```

### Step 3: Test Connection

```bash
cd /home/majid/Downloads/jetson-ml-assessment
source venv/bin/activate
python3 slack_test.py
```

✅ If successful, you'll see your bot info and channel list!

---

## 📦 Available Scripts

### 1. **slack_test.py** - Test Connection

**Purpose:** Test Slack connection and show capabilities

**Usage:**
```bash
python3 slack_test.py
```

**What it does:**
- ✅ Verifies authentication
- 📺 Lists all channels
- 👥 Shows workspace users
- ✨ Demonstrates API capabilities
- 💬 Optionally sends test message

---

### 2. **slack_chatbot_monitor.py** - Monitor Chatbot

**Purpose:** Continuously monitor chatbot health and send alerts

**Usage:**
```bash
# Continuous monitoring (checks every 5 minutes)
python3 slack_chatbot_monitor.py

# Send one-time test notification
python3 slack_chatbot_monitor.py test
```

**What it monitors:**
- ✅ Chatbot process status (running/stopped)
- 💻 CPU and memory usage
- 📊 User count and activity
- ⚠️ Resource warnings (>90% usage)
- 🔔 Status change notifications

**Example notifications:**
- "✅ Chatbot Status: Running"
- "❌ ALERT: Chatbot Down"
- "⚠️ Warning: Resource Usage High"

---

### 3. **slack_benchmark_reporter.py** - Post Benchmarks

**Purpose:** Send benchmark results to Slack with nice formatting

**Usage:**
```bash
# Report single benchmark
python3 slack_benchmark_reporter.py gb10_benchmark_results.json

# Compare two benchmarks
python3 slack_benchmark_reporter.py jetson_benchmark_results.json gb10_benchmark_results.json
```

**What it shows:**
- 🎯 Performance metrics (FPS, latency, throughput)
- 🖥️ System configuration
- 📊 Summary statistics
- 📈 Comparison with improvements
- 📎 Uploads full JSON file

**Example output:**
```
🎯 Benchmark Results

System Configuration
• GPU: NVIDIA GB10
• CUDA: 12.9
• PyTorch: 2.9.0

📊 Performance Results
• ResNet-18: 125.43 FPS • Latency: 7.97 ms
• MobileNet-v2: 189.67 FPS • Latency: 5.27 ms

🕐 2025-11-06 14:23:45
```

---

## 🎯 Example Workflows

### Monitor Chatbot 24/7

```bash
# Terminal 1: Run chatbot
cd /home/majid/Downloads/jetson-ml-assessment
source venv/bin/activate
python3 uvu_chatbot_pro.py

# Terminal 2: Run monitor
cd /home/majid/Downloads/jetson-ml-assessment
source venv/bin/activate
export SLACK_BOT_TOKEN='xoxb-your-token'
export SLACK_CHANNEL='ml-alerts'
python3 slack_chatbot_monitor.py
```

Now you'll get Slack notifications when:
- Chatbot starts/stops
- Resource usage is high
- Any status changes

---

### Share Benchmark Results

```bash
# Run benchmark
python3 gb10_ml_benchmark.py

# Post to Slack
python3 slack_benchmark_reporter.py gb10_benchmark_results.json
```

Your team will see a nicely formatted report in Slack!

---

### Compare Performance

```bash
# Compare Jetson vs GB10
python3 slack_benchmark_reporter.py \
    jetson_benchmark_results.json \
    gb10_benchmark_results.json
```

Shows improvement percentages for each model! 🚀

---

## 🔧 Configuration Options

### Environment Variables

```bash
# Required
export SLACK_BOT_TOKEN='xoxb-...'

# Optional
export SLACK_CHANNEL='general'  # Default channel
```

### Change Check Interval (Monitor)

Edit `slack_chatbot_monitor.py`:
```python
CHECK_INTERVAL = 300  # seconds (5 minutes)
```

Change to:
- `60` = 1 minute
- `300` = 5 minutes (default)
- `600` = 10 minutes
- `3600` = 1 hour

---

## 💡 Integration Ideas

### 1. Training Notifications

```python
from slack_sdk import WebClient

client = WebClient(token=SLACK_BOT_TOKEN)

# Before training
client.chat_postMessage(
    channel='#ml-training',
    text='🚀 Started training ResNet-50'
)

# After training
client.chat_postMessage(
    channel='#ml-training',
    text='✅ Training complete! Accuracy: 95.3%'
)
```

### 2. Error Alerts

```python
try:
    train_model()
except Exception as e:
    client.chat_postMessage(
        channel='#alerts',
        text=f'❌ Training failed: {str(e)}'
    )
```

### 3. Daily Reports

```python
import schedule

def send_daily_stats():
    stats = get_daily_stats()
    client.chat_postMessage(
        channel='#reports',
        text=f'📊 Daily Stats\nUsers: {stats["users"]}\nMessages: {stats["messages"]}'
    )

schedule.every().day.at("09:00").do(send_daily_stats)
```

---

## 🐛 Troubleshooting

### "Token not found"

```bash
# Check if set
echo $SLACK_BOT_TOKEN

# Set it
export SLACK_BOT_TOKEN='xoxb-your-token'
```

### "not_in_channel" error

The bot needs to be invited to the channel:
1. Open Slack channel
2. Type: `/invite @YourBotName`

### "missing_scope" error

Add the required scope:
1. Go to app settings → OAuth & Permissions
2. Add the missing scope
3. Reinstall app to workspace

### Rate limiting

Slack has rate limits:
- 1 message per second per channel
- 50 files per minute
- 100 API calls per minute

The scripts handle this automatically.

---

## 📚 More Examples

See **SLACK_INTEGRATION_GUIDE.md** for:
- Advanced features
- Interactive buttons
- Slash commands
- Modal forms
- File uploads
- And more!

---

## ✅ Checklist

Before using:
- [ ] Created Slack app
- [ ] Added bot token scopes
- [ ] Installed app to workspace
- [ ] Got bot token (xoxb-...)
- [ ] Set SLACK_BOT_TOKEN env var
- [ ] Tested connection with `slack_test.py`
- [ ] Invited bot to channels you want to use

---

## 🚀 Next Steps

1. **Test the connection:**
   ```bash
   python3 slack_test.py
   ```

2. **Try the monitor:**
   ```bash
   python3 slack_chatbot_monitor.py test
   ```

3. **Post a benchmark:**
   ```bash
   python3 slack_benchmark_reporter.py gb10_benchmark_results.json
   ```

4. **Run continuous monitoring:**
   ```bash
   python3 slack_chatbot_monitor.py
   ```

---

**Status:** ✅ Ready to use (just need Slack token)  
**Support:** See SLACK_INTEGRATION_GUIDE.md for detailed docs

