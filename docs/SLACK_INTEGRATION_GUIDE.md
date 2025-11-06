# Slack Integration Guide

**Project:** NVIDIA Jetson ML Assessment  
**Date:** November 6, 2025

---

## 🎯 Overview

This guide shows how to integrate Slack with your ML/AI projects for notifications, monitoring, and team collaboration.

---

## 🚀 Quick Start

### Step 1: Install Slack SDK

```bash
cd /home/majid/Downloads/jetson-ml-assessment
source venv/bin/activate
pip install slack-sdk
```

✅ Already installed!

---

### Step 2: Create a Slack App

1. **Go to Slack API:** https://api.slack.com/apps
2. **Click** "Create New App"
3. **Choose** "From scratch"
4. **Name your app:** (e.g., "ML Chatbot Monitor")
5. **Select workspace:** Choose your Slack workspace

---

### Step 3: Configure Bot Permissions

In your Slack app settings:

1. Go to **"OAuth & Permissions"**
2. Scroll to **"Bot Token Scopes"**
3. Add these scopes:

**Essential Scopes:**
- `channels:read` - View channels
- `channels:write` - Manage channels
- `chat:write` - Send messages
- `users:read` - View users
- `groups:read` - View private channels
- `im:write` - Send DMs
- `mpim:write` - Send group DMs

**Optional (for advanced features):**
- `files:write` - Upload files
- `reactions:write` - Add reactions
- `pins:write` - Pin messages
- `channels:history` - Read message history
- `commands` - Create slash commands
- `app_mentions:read` - Detect @mentions

---

### Step 4: Install App to Workspace

1. Click **"Install to Workspace"**
2. **Authorize** the permissions
3. **Copy** the "Bot User OAuth Token" (starts with `xoxb-`)

---

### Step 5: Set Environment Variable

```bash
# Temporary (current session)
export SLACK_BOT_TOKEN='xoxb-your-token-here'

# Permanent (add to ~/.bashrc)
echo 'export SLACK_BOT_TOKEN="xoxb-your-token-here"' >> ~/.bashrc
source ~/.bashrc
```

**Security Note:** NEVER commit tokens to git!

---

### Step 6: Test Connection

```bash
cd /home/majid/Downloads/jetson-ml-assessment
source venv/bin/activate
python3 slack_test.py
```

---

## 💡 What You Can Do with Slack

### 1. 💬 Messaging

**Send Messages:**
```python
from slack_sdk import WebClient

client = WebClient(token='xoxb-your-token')

# Send to channel
client.chat_postMessage(
    channel='#general',
    text='Hello from Python! 🤖'
)

# Send formatted message
client.chat_postMessage(
    channel='#general',
    text='Status Update',
    blocks=[
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*ML Model Training Complete!*\n✅ Accuracy: 95.3%"
            }
        }
    ]
)
```

**Send Direct Messages:**
```python
# Send DM to user
client.chat_postMessage(
    channel='@username',
    text='Private notification'
)
```

**Update Messages:**
```python
# Update existing message
client.chat_update(
    channel='C1234567890',
    ts='1234567890.123456',
    text='Updated message'
)
```

---

### 2. 📺 Channel Management

**List Channels:**
```python
response = client.conversations_list()
for channel in response['channels']:
    print(f"#{channel['name']}")
```

**Create Channel:**
```python
response = client.conversations_create(
    name='ml-notifications',
    is_private=False
)
```

**Join Channel:**
```python
client.conversations_join(channel='C1234567890')
```

**Invite Users:**
```python
client.conversations_invite(
    channel='C1234567890',
    users='U1234567890'
)
```

---

### 3. 👥 User Management

**List Users:**
```python
response = client.users_list()
for user in response['members']:
    if not user['is_bot'] and not user['deleted']:
        print(f"@{user['name']}: {user['profile']['real_name']}")
```

**Get User Info:**
```python
user_info = client.users_info(user='U1234567890')
print(user_info['user']['profile']['real_name'])
```

**Lookup User by Email:**
```python
user = client.users_lookupByEmail(email='user@example.com')
```

---

### 4. 📁 File Sharing

**Upload File:**
```python
client.files_upload_v2(
    channel='#general',
    file='benchmark_results.json',
    title='ML Benchmark Results',
    initial_comment='Latest benchmark results 📊'
)
```

**Upload Multiple Files:**
```python
client.files_upload_v2(
    channels=['#general', '#ml-team'],
    file='model.pth',
    title='Trained Model v2.0'
)
```

---

### 5. 🔔 Interactive Features

**Add Reactions:**
```python
client.reactions_add(
    channel='C1234567890',
    timestamp='1234567890.123456',
    name='rocket'  # :rocket:
)
```

**Create Buttons:**
```python
client.chat_postMessage(
    channel='#general',
    text='Choose an action:',
    blocks=[
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Start Training"},
                    "action_id": "start_training",
                    "value": "start"
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Stop Training"},
                    "action_id": "stop_training",
                    "value": "stop",
                    "style": "danger"
                }
            ]
        }
    ]
)
```

**Create Modal (Pop-up Form):**
```python
client.views_open(
    trigger_id='trigger_id_from_interaction',
    view={
        "type": "modal",
        "title": {"type": "plain_text", "text": "Model Configuration"},
        "blocks": [
            {
                "type": "input",
                "label": {"type": "plain_text", "text": "Learning Rate"},
                "element": {"type": "plain_text_input", "action_id": "lr_input"}
            }
        ]
    }
)
```

---

### 6. 📊 Rich Message Formatting

**Using Blocks (Recommended):**
```python
client.chat_postMessage(
    channel='#general',
    text='System Status',
    blocks=[
        {
            "type": "header",
            "text": {"type": "plain_text", "text": "🤖 Chatbot Status"}
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": "*Status:*\n✅ Running"},
                {"type": "mrkdwn", "text": "*Uptime:*\n24 hours"},
                {"type": "mrkdwn", "text": "*GPU:*\n85% utilized"},
                {"type": "mrkdwn", "text": "*Memory:*\n45 GB / 120 GB"}
            ]
        },
        {
            "type": "divider"
        },
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": "Last updated: <!date^1699286400^{date_short_pretty} at {time}|Nov 6, 2025>"
                }
            ]
        }
    ]
)
```

---

### 7. 🔍 Search and History

**Search Messages:**
```python
response = client.search_messages(
    query='error OR failed',
    sort='timestamp',
    sort_dir='desc'
)
```

**Get Conversation History:**
```python
history = client.conversations_history(
    channel='C1234567890',
    limit=50
)

for message in history['messages']:
    print(f"{message.get('user', 'Bot')}: {message['text']}")
```

---

## 🎯 ML Project Integration Ideas

### 1. Training Notifications

```python
# Send training start notification
client.chat_postMessage(
    channel='#ml-training',
    text=f'🚀 Training started: ResNet-50 on GPU {gpu_id}'
)

# Send progress updates
for epoch in range(epochs):
    if epoch % 10 == 0:
        client.chat_postMessage(
            channel='#ml-training',
            text=f'📊 Epoch {epoch}/{epochs} - Loss: {loss:.4f}'
        )

# Send completion notification
client.chat_postMessage(
    channel='#ml-training',
    text='✅ Training complete! Accuracy: 95.3%',
    blocks=[
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*Training Complete!*\n✅ Final Accuracy: 95.3%\n⏱️ Time: 2h 15m"
            }
        }
    ]
)
```

---

### 2. System Monitoring

```python
import psutil
import GPUtil

def send_system_status():
    gpus = GPUtil.getGPUs()
    
    client.chat_postMessage(
        channel='#system-alerts',
        blocks=[
            {
                "type": "header",
                "text": {"type": "plain_text", "text": "🖥️ System Status"}
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*CPU:*\n{psutil.cpu_percent()}%"},
                    {"type": "mrkdwn", "text": f"*Memory:*\n{psutil.virtual_memory().percent}%"},
                    {"type": "mrkdwn", "text": f"*GPU:*\n{gpus[0].load*100:.1f}%"},
                    {"type": "mrkdwn", "text": f"*GPU Temp:*\n{gpus[0].temperature}°C"}
                ]
            }
        ]
    )

# Run every hour
import schedule
schedule.every().hour.do(send_system_status)
```

---

### 3. Error Alerts

```python
import traceback

def alert_on_error(error):
    """Send error alerts to Slack"""
    error_msg = traceback.format_exc()
    
    client.chat_postMessage(
        channel='#alerts',
        text='❌ System Error Detected',
        blocks=[
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*❌ Error Detected*\n```{error_msg}```"
                }
            }
        ]
    )

try:
    # Your code
    risky_operation()
except Exception as e:
    alert_on_error(e)
```

---

### 4. Benchmark Reports

```python
def send_benchmark_results(results):
    """Send benchmark results to Slack"""
    
    # Create formatted message
    message = "*🎯 Benchmark Results*\n\n"
    for model, fps in results.items():
        message += f"• {model}: {fps:.2f} FPS\n"
    
    # Upload detailed results as file
    client.files_upload_v2(
        channel='#benchmarks',
        file='benchmark_results.json',
        title='Full Benchmark Data',
        initial_comment=message
    )
```

---

### 5. Interactive Model Control

```python
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

app = App(token=SLACK_BOT_TOKEN)

@app.command("/train")
def handle_train_command(ack, command, client):
    """Handle /train slash command"""
    ack()
    
    # Show model selection modal
    client.views_open(
        trigger_id=command['trigger_id'],
        view={
            "type": "modal",
            "callback_id": "train_modal",
            "title": {"type": "plain_text", "text": "Train Model"},
            "blocks": [
                {
                    "type": "input",
                    "block_id": "model_block",
                    "label": {"type": "plain_text", "text": "Select Model"},
                    "element": {
                        "type": "static_select",
                        "action_id": "model_select",
                        "options": [
                            {"text": {"type": "plain_text", "text": "ResNet-18"}, "value": "resnet18"},
                            {"text": {"type": "plain_text", "text": "MobileNet-v2"}, "value": "mobilenet"}
                        ]
                    }
                }
            ]
        }
    )

@app.view("train_modal")
def handle_train_submission(ack, body, client):
    """Handle train modal submission"""
    ack()
    
    model = body['view']['state']['values']['model_block']['model_select']['selected_option']['value']
    
    # Start training
    start_training(model)
    
    # Notify user
    client.chat_postMessage(
        channel=body['user']['id'],
        text=f'🚀 Started training {model}'
    )

# Start app
SocketModeHandler(app, SLACK_APP_TOKEN).start()
```

---

### 6. Daily Reports

```python
import schedule
from datetime import datetime

def send_daily_report():
    """Send daily usage report"""
    
    stats = get_daily_stats()  # Your stats function
    
    client.chat_postMessage(
        channel='#daily-reports',
        blocks=[
            {
                "type": "header",
                "text": {"type": "plain_text", "text": f"📊 Daily Report - {datetime.now().strftime('%Y-%m-%d')}"}
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Total Requests:*\n{stats['requests']:,}"},
                    {"type": "mrkdwn", "text": f"*Active Users:*\n{stats['users']}"},
                    {"type": "mrkdwn", "text": f"*Avg Response Time:*\n{stats['avg_time']:.2f}s"},
                    {"type": "mrkdwn", "text": f"*Uptime:*\n{stats['uptime']}"}
                ]
            }
        ]
    )

# Schedule daily at 9 AM
schedule.every().day.at("09:00").do(send_daily_report)
```

---

## 📦 Example Integration Scripts

I've created these scripts for you:

1. **`slack_test.py`** - Test Slack connection and show capabilities
2. **`slack_ml_notifier.py`** - Send ML training notifications (TO BE CREATED)
3. **`slack_chatbot_monitor.py`** - Monitor chatbot health (TO BE CREATED)
4. **`slack_benchmark_reporter.py`** - Post benchmark results (TO BE CREATED)

---

## 🔒 Security Best Practices

### ✅ DO:
- Store tokens in environment variables
- Use `.env` files (added to `.gitignore`)
- Rotate tokens regularly
- Use minimal required scopes
- Implement rate limiting

### ❌ DON'T:
- Hardcode tokens in code
- Commit tokens to git
- Share tokens publicly
- Use tokens in URLs
- Log tokens in plaintext

---

## 🐛 Troubleshooting

### Token Not Found
```bash
echo $SLACK_BOT_TOKEN  # Should show your token
export SLACK_BOT_TOKEN='xoxb-your-token'
```

### Not in Channel Error
```python
# Join channel first
client.conversations_join(channel='C1234567890')
```

### Missing Permissions
- Go to app settings → OAuth & Permissions
- Add required scopes
- Reinstall app to workspace

### Rate Limits
```python
from slack_sdk.errors import SlackApiError
import time

try:
    client.chat_postMessage(...)
except SlackApiError as e:
    if e.response['error'] == 'rate_limited':
        # Wait and retry
        time.sleep(int(e.response.headers['Retry-After']))
```

---

## 📚 Resources

- **Slack API Documentation:** https://api.slack.com/
- **Python SDK:** https://github.com/slackapi/python-slack-sdk
- **Block Kit Builder:** https://app.slack.com/block-kit-builder
- **API Methods:** https://api.slack.com/methods

---

## 🚀 Next Steps

1. ✅ Install Slack SDK
2. ⏳ Create Slack App
3. ⏳ Get Bot Token
4. ⏳ Set Environment Variable
5. ⏳ Test Connection
6. ⏳ Create Integration Scripts
7. ⏳ Deploy to Production

---

**Status:** Ready to integrate once you have a Slack Bot Token!  
**Test Command:** `python3 slack_test.py`

