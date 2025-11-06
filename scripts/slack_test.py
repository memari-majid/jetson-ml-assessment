#!/usr/bin/env python3
"""
Slack Integration Test Script
Tests Slack API connection and demonstrates available capabilities
"""

import os
import sys
from datetime import datetime

try:
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError
except ImportError:
    print("❌ Slack SDK not installed")
    print("\nInstalling slack_sdk...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "slack-sdk"])
    print("✅ Slack SDK installed! Please run the script again.")
    sys.exit(0)


class SlackTester:
    """Test and demonstrate Slack API capabilities"""
    
    def __init__(self, token=None):
        """Initialize Slack client with token"""
        self.token = token or os.environ.get('SLACK_BOT_TOKEN')
        if not self.token:
            raise ValueError("No Slack token provided. Set SLACK_BOT_TOKEN environment variable.")
        
        self.client = WebClient(token=self.token)
        self.bot_info = None
        
    def test_authentication(self):
        """Test if the token is valid"""
        print("\n" + "="*60)
        print("🔐 TESTING AUTHENTICATION")
        print("="*60)
        
        try:
            response = self.client.auth_test()
            self.bot_info = response
            
            print("✅ Authentication successful!")
            print(f"\n📋 Bot Information:")
            print(f"   • Bot Name: {response['user']}")
            print(f"   • Bot ID: {response['user_id']}")
            print(f"   • Team: {response['team']}")
            print(f"   • Team ID: {response['team_id']}")
            print(f"   • URL: {response['url']}")
            
            return True
            
        except SlackApiError as e:
            print(f"❌ Authentication failed: {e.response['error']}")
            return False
    
    def list_channels(self):
        """List all channels the bot has access to"""
        print("\n" + "="*60)
        print("📺 LISTING CHANNELS")
        print("="*60)
        
        try:
            # List public channels
            response = self.client.conversations_list(
                types="public_channel,private_channel",
                limit=50
            )
            
            channels = response['channels']
            print(f"\n✅ Found {len(channels)} channels:")
            
            for channel in channels:
                channel_type = "🔒 Private" if channel['is_private'] else "🌐 Public"
                member = "✓ Member" if channel.get('is_member', False) else "✗ Not member"
                print(f"\n   {channel_type} - #{channel['name']}")
                print(f"      ID: {channel['id']}")
                print(f"      {member}")
                if channel.get('topic', {}).get('value'):
                    print(f"      Topic: {channel['topic']['value']}")
            
            return channels
            
        except SlackApiError as e:
            print(f"❌ Failed to list channels: {e.response['error']}")
            return []
    
    def list_users(self):
        """List users in the workspace"""
        print("\n" + "="*60)
        print("👥 LISTING USERS")
        print("="*60)
        
        try:
            response = self.client.users_list()
            users = response['members']
            
            print(f"\n✅ Found {len(users)} users:")
            
            # Filter to show only active, non-bot users (first 10)
            active_users = [u for u in users if not u['deleted'] and not u['is_bot']][:10]
            
            for user in active_users:
                status = "🟢 Active" if user.get('is_active', False) else "⚫ Inactive"
                print(f"\n   {status} - {user['profile'].get('real_name', user['name'])}")
                print(f"      Username: @{user['name']}")
                print(f"      ID: {user['id']}")
            
            if len(active_users) < len([u for u in users if not u['deleted'] and not u['is_bot']]):
                print(f"\n   ... and {len([u for u in users if not u['deleted'] and not u['is_bot']]) - len(active_users)} more users")
            
            return users
            
        except SlackApiError as e:
            print(f"❌ Failed to list users: {e.response['error']}")
            return []
    
    def send_test_message(self, channel_name="general"):
        """Send a test message to a channel"""
        print("\n" + "="*60)
        print(f"💬 SENDING TEST MESSAGE TO #{channel_name}")
        print("="*60)
        
        try:
            # Find the channel ID
            channels = self.client.conversations_list(types="public_channel,private_channel")
            channel_id = None
            
            for channel in channels['channels']:
                if channel['name'] == channel_name:
                    channel_id = channel['id']
                    break
            
            if not channel_id:
                print(f"❌ Channel #{channel_name} not found")
                print("\nAvailable channels:")
                for channel in channels['channels']:
                    if channel.get('is_member', False):
                        print(f"   • #{channel['name']}")
                return False
            
            # Send message
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            message = f"🤖 *Slack Connection Test*\n\nSuccessfully connected at {timestamp}\n\n_This is an automated test message from the jetson-ml-assessment project._"
            
            response = self.client.chat_postMessage(
                channel=channel_id,
                text=message,
                mrkdwn=True
            )
            
            print(f"✅ Message sent successfully!")
            print(f"   • Channel: #{channel_name}")
            print(f"   • Message ID: {response['ts']}")
            
            return True
            
        except SlackApiError as e:
            print(f"❌ Failed to send message: {e.response['error']}")
            if e.response['error'] == 'not_in_channel':
                print(f"\n💡 Tip: Add the bot to #{channel_name} channel first!")
            return False
    
    def demonstrate_capabilities(self):
        """Show what we can do with Slack API"""
        print("\n" + "="*60)
        print("✨ SLACK API CAPABILITIES")
        print("="*60)
        
        capabilities = {
            "Messaging": [
                "✅ Send messages to channels",
                "✅ Send direct messages to users",
                "✅ Send rich formatted messages (blocks)",
                "✅ Update existing messages",
                "✅ Delete messages",
                "✅ Add reactions to messages",
                "✅ Pin/unpin messages"
            ],
            "Channels": [
                "✅ List all channels",
                "✅ Create new channels",
                "✅ Archive/unarchive channels",
                "✅ Join/leave channels",
                "✅ Invite users to channels",
                "✅ Set channel topic/purpose"
            ],
            "Users": [
                "✅ List workspace users",
                "✅ Get user information",
                "✅ Set bot presence",
                "✅ Lookup users by email"
            ],
            "Files": [
                "✅ Upload files",
                "✅ Share files to channels",
                "✅ Download files",
                "✅ Delete files"
            ],
            "Interactive": [
                "✅ Listen for events (mentions, messages)",
                "✅ Respond to slash commands",
                "✅ Create interactive buttons",
                "✅ Build modals (pop-up forms)",
                "✅ Real-time messaging (Socket Mode)"
            ],
            "Advanced": [
                "✅ Get conversation history",
                "✅ Search messages",
                "✅ Create reminders",
                "✅ Get workspace info",
                "✅ Manage user groups"
            ]
        }
        
        for category, features in capabilities.items():
            print(f"\n🔹 {category}:")
            for feature in features:
                print(f"   {feature}")
        
        print("\n" + "="*60)
        print("💡 INTEGRATION IDEAS")
        print("="*60)
        
        ideas = [
            "Send ML model training notifications",
            "Post GPU benchmark results to Slack",
            "Alert team when chatbot goes down",
            "Share inference results and analytics",
            "Create slash commands for system status",
            "Build interactive model selection bot",
            "Send daily usage reports",
            "Alert on performance anomalies"
        ]
        
        for i, idea in enumerate(ideas, 1):
            print(f"   {i}. {idea}")


def main():
    """Main test function"""
    print("\n" + "="*60)
    print("🚀 SLACK INTEGRATION TESTER")
    print("="*60)
    print("\nThis script tests the Slack API connection and shows capabilities.")
    
    # Check for token
    token = os.environ.get('SLACK_BOT_TOKEN')
    
    if not token:
        print("\n❌ SLACK_BOT_TOKEN environment variable not set")
        print("\n📋 Setup Instructions:")
        print("="*60)
        print("\n1. Create a Slack App:")
        print("   • Go to https://api.slack.com/apps")
        print("   • Click 'Create New App'")
        print("   • Choose 'From scratch'")
        print("   • Name your app and select workspace")
        
        print("\n2. Configure Bot Token Scopes:")
        print("   • Go to 'OAuth & Permissions'")
        print("   • Add these Bot Token Scopes:")
        print("     - channels:read")
        print("     - channels:write")
        print("     - chat:write")
        print("     - users:read")
        print("     - groups:read")
        print("     - im:write")
        print("     - mpim:write")
        
        print("\n3. Install App to Workspace:")
        print("   • Click 'Install to Workspace'")
        print("   • Authorize the app")
        
        print("\n4. Copy Bot Token:")
        print("   • Copy the 'Bot User OAuth Token' (starts with xoxb-)")
        
        print("\n5. Set Environment Variable:")
        print("   • export SLACK_BOT_TOKEN='xoxb-your-token-here'")
        
        print("\n6. Run this script again:")
        print("   • python3 slack_test.py")
        
        print("\n" + "="*60)
        return
    
    try:
        # Initialize tester
        tester = SlackTester(token)
        
        # Run tests
        if not tester.test_authentication():
            print("\n❌ Authentication failed. Please check your token.")
            return
        
        tester.list_channels()
        tester.list_users()
        tester.demonstrate_capabilities()
        
        # Ask if user wants to send test message
        print("\n" + "="*60)
        print("📤 TEST MESSAGE")
        print("="*60)
        response = input("\nWould you like to send a test message? (y/n): ").strip().lower()
        
        if response == 'y':
            channel = input("Enter channel name (default: general): ").strip() or "general"
            tester.send_test_message(channel)
        
        print("\n" + "="*60)
        print("✅ SLACK TEST COMPLETE!")
        print("="*60)
        print("\nYou can now integrate Slack into your applications!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

