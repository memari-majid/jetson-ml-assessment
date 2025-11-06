#!/usr/bin/env python3
"""
Slack Chatbot Monitor
Monitors the UVU chatbot and sends status updates to Slack
"""

import os
import sys
import time
import psutil
import sqlite3
import json
from datetime import datetime
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

# Configuration
SLACK_TOKEN = os.environ.get('SLACK_BOT_TOKEN')
CHANNEL = os.environ.get('SLACK_CHANNEL', 'general')
CHECK_INTERVAL = 300  # 5 minutes
DB_PATH = 'chatbot_data/users.db'


class ChatbotMonitor:
    """Monitor chatbot and send Slack notifications"""
    
    def __init__(self, token, channel):
        if not token:
            raise ValueError("SLACK_BOT_TOKEN not set")
        
        self.client = WebClient(token=token)
        self.channel = channel
        self.last_status = None
        
    def check_chatbot_process(self):
        """Check if chatbot process is running"""
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = ' '.join(proc.info['cmdline'] or [])
                if 'uvu_chatbot_pro.py' in cmdline:
                    return {
                        'running': True,
                        'pid': proc.info['pid'],
                        'cpu': proc.cpu_percent(interval=0.1),
                        'memory': proc.memory_info().rss / 1024**3  # GB
                    }
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        return {'running': False}
    
    def get_database_stats(self):
        """Get statistics from chatbot database"""
        if not os.path.exists(DB_PATH):
            return None
        
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Get user count
            cursor.execute("SELECT COUNT(*) FROM users")
            user_count = cursor.fetchone()[0]
            
            # Get total messages (if table exists)
            try:
                cursor.execute("""
                    SELECT COUNT(*) FROM chat_history 
                    WHERE timestamp > datetime('now', '-24 hours')
                """)
                messages_24h = cursor.fetchone()[0]
            except:
                messages_24h = 0
            
            conn.close()
            
            return {
                'total_users': user_count,
                'messages_24h': messages_24h
            }
        except Exception as e:
            return None
    
    def get_system_resources(self):
        """Get system resource usage"""
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_available_gb': psutil.virtual_memory().available / 1024**3,
            'disk_usage_percent': psutil.disk_usage('/').percent
        }
    
    def send_status_message(self, status_type, details):
        """Send status message to Slack"""
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if status_type == 'running':
            emoji = "✅"
            title = "Chatbot Status: Running"
            color = "good"
        elif status_type == 'down':
            emoji = "❌"
            title = "⚠️ ALERT: Chatbot Down"
            color = "danger"
        elif status_type == 'warning':
            emoji = "⚠️"
            title = "Warning: Resource Usage High"
            color = "warning"
        else:
            emoji = "ℹ️"
            title = "Chatbot Status Update"
            color = None
        
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{emoji} {title}"
                }
            }
        ]
        
        # Add details as fields
        fields = []
        for key, value in details.items():
            if isinstance(value, float):
                formatted_value = f"{value:.2f}"
            else:
                formatted_value = str(value)
            
            fields.append({
                "type": "mrkdwn",
                "text": f"*{key}:*\n{formatted_value}"
            })
        
        if fields:
            # Split into chunks of 10 fields (Slack limit)
            for i in range(0, len(fields), 10):
                blocks.append({
                    "type": "section",
                    "fields": fields[i:i+10]
                })
        
        # Add timestamp
        blocks.append({
            "type": "context",
            "elements": [{
                "type": "mrkdwn",
                "text": f"🕐 {timestamp}"
            }]
        })
        
        try:
            self.client.chat_postMessage(
                channel=self.channel,
                text=title,
                blocks=blocks
            )
            print(f"✅ Sent {status_type} notification to Slack")
        except SlackApiError as e:
            print(f"❌ Failed to send message: {e.response['error']}")
    
    def check_and_notify(self):
        """Check chatbot status and send notification if changed"""
        
        process_status = self.check_chatbot_process()
        system_resources = self.get_system_resources()
        db_stats = self.get_database_stats()
        
        current_status = {
            'running': process_status['running'],
            'timestamp': datetime.now().isoformat()
        }
        
        # Chatbot down
        if not process_status['running']:
            if self.last_status is None or self.last_status['running']:
                details = {
                    'Status': '❌ Not Running',
                    'CPU Usage': f"{system_resources['cpu_percent']}%",
                    'Memory Available': f"{system_resources['memory_available_gb']:.1f} GB"
                }
                self.send_status_message('down', details)
        
        # Chatbot running
        else:
            details = {
                'Status': '✅ Running',
                'Process ID': process_status['pid'],
                'CPU Usage': f"{process_status['cpu']:.1f}%",
                'Memory Usage': f"{process_status['memory']:.1f} GB",
                'System CPU': f"{system_resources['cpu_percent']:.1f}%",
                'System Memory': f"{system_resources['memory_percent']:.1f}%"
            }
            
            # Add database stats if available
            if db_stats:
                details['Total Users'] = db_stats['total_users']
                details['Messages (24h)'] = db_stats['messages_24h']
            
            # Send notification if status changed
            if self.last_status is None or not self.last_status['running']:
                self.send_status_message('running', details)
            
            # Check for resource warnings
            elif (system_resources['cpu_percent'] > 90 or 
                  system_resources['memory_percent'] > 90):
                warning_details = {
                    'Status': '⚠️ High Resource Usage',
                    'CPU': f"{system_resources['cpu_percent']:.1f}%",
                    'Memory': f"{system_resources['memory_percent']:.1f}%"
                }
                self.send_status_message('warning', warning_details)
        
        self.last_status = current_status
    
    def run_monitoring(self):
        """Run continuous monitoring"""
        print(f"🔍 Starting chatbot monitoring...")
        print(f"📺 Sending notifications to #{self.channel}")
        print(f"⏱️  Check interval: {CHECK_INTERVAL} seconds")
        print("\nPress Ctrl+C to stop\n")
        
        try:
            while True:
                self.check_and_notify()
                time.sleep(CHECK_INTERVAL)
        except KeyboardInterrupt:
            print("\n\n✋ Monitoring stopped")
    
    def send_test_notification(self):
        """Send a test notification"""
        process_status = self.check_chatbot_process()
        system_resources = self.get_system_resources()
        db_stats = self.get_database_stats()
        
        details = {
            'Test': '✅ Connection Working',
            'Chatbot Status': '✅ Running' if process_status['running'] else '❌ Not Running',
            'System CPU': f"{system_resources['cpu_percent']:.1f}%",
            'System Memory': f"{system_resources['memory_percent']:.1f}%"
        }
        
        if process_status['running']:
            details['Process PID'] = process_status['pid']
            details['Process CPU'] = f"{process_status['cpu']:.1f}%"
            details['Process Memory'] = f"{process_status['memory']:.1f} GB"
        
        if db_stats:
            details['Total Users'] = db_stats['total_users']
            details['Messages (24h)'] = db_stats['messages_24h']
        
        self.send_status_message('info', details)


def main():
    """Main function"""
    
    if not SLACK_TOKEN:
        print("❌ SLACK_BOT_TOKEN environment variable not set")
        print("\nSet it with:")
        print("  export SLACK_BOT_TOKEN='xoxb-your-token-here'")
        sys.exit(1)
    
    monitor = ChatbotMonitor(SLACK_TOKEN, CHANNEL)
    
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        print("🧪 Sending test notification...")
        monitor.send_test_notification()
        print("✅ Test complete!")
    else:
        monitor.run_monitoring()


if __name__ == "__main__":
    main()

