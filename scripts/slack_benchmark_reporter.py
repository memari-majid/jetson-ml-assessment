#!/usr/bin/env python3
"""
Slack Benchmark Reporter
Posts ML benchmark results to Slack with beautiful formatting
"""

import os
import sys
import json
from datetime import datetime
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

# Configuration
SLACK_TOKEN = os.environ.get('SLACK_BOT_TOKEN')
CHANNEL = os.environ.get('SLACK_CHANNEL', 'general')


class BenchmarkReporter:
    """Report benchmark results to Slack"""
    
    def __init__(self, token, channel):
        if not token:
            raise ValueError("SLACK_BOT_TOKEN not set")
        
        self.client = WebClient(token=token)
        self.channel = channel
    
    def read_benchmark_file(self, filepath):
        """Read benchmark results from JSON file"""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"❌ File not found: {filepath}")
            return None
        except json.JSONDecodeError:
            print(f"❌ Invalid JSON in: {filepath}")
            return None
    
    def format_benchmark_message(self, results, title="Benchmark Results"):
        """Format benchmark results as Slack blocks"""
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"🎯 {title}"
                }
            }
        ]
        
        # Add system info if available
        if 'system_info' in results:
            sys_info = results['system_info']
            fields = []
            
            if 'gpu' in sys_info:
                fields.append({
                    "type": "mrkdwn",
                    "text": f"*GPU:*\n{sys_info['gpu']}"
                })
            if 'cuda' in sys_info:
                fields.append({
                    "type": "mrkdwn",
                    "text": f"*CUDA:*\n{sys_info['cuda']}"
                })
            if 'pytorch' in sys_info:
                fields.append({
                    "type": "mrkdwn",
                    "text": f"*PyTorch:*\n{sys_info['pytorch']}"
                })
            
            if fields:
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "*System Configuration*"
                    },
                    "fields": fields
                })
                blocks.append({"type": "divider"})
        
        # Add benchmark results
        if 'benchmarks' in results:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*📊 Performance Results*"
                }
            })
            
            for bench in results['benchmarks']:
                name = bench.get('name', 'Unknown')
                
                # Format metrics
                metrics = []
                if 'fps' in bench:
                    metrics.append(f"*FPS:* {bench['fps']:.2f}")
                if 'latency_ms' in bench:
                    metrics.append(f"*Latency:* {bench['latency_ms']:.2f} ms")
                if 'throughput' in bench:
                    metrics.append(f"*Throughput:* {bench['throughput']:.2f}")
                
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*{name}*\n" + " • ".join(metrics)
                    }
                })
        
        # Add summary if available
        if 'summary' in results:
            blocks.append({"type": "divider"})
            summary = results['summary']
            
            summary_fields = []
            for key, value in summary.items():
                if isinstance(value, float):
                    formatted_value = f"{value:.2f}"
                else:
                    formatted_value = str(value)
                
                summary_fields.append({
                    "type": "mrkdwn",
                    "text": f"*{key}:*\n{formatted_value}"
                })
            
            if summary_fields:
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "*Summary*"
                    },
                    "fields": summary_fields
                })
        
        # Add timestamp
        blocks.append({
            "type": "context",
            "elements": [{
                "type": "mrkdwn",
                "text": f"🕐 {timestamp}"
            }]
        })
        
        return blocks
    
    def send_benchmark_report(self, results, title="Benchmark Results", upload_file=None):
        """Send benchmark report to Slack"""
        
        blocks = self.format_benchmark_message(results, title)
        
        try:
            # Send message
            response = self.client.chat_postMessage(
                channel=self.channel,
                text=title,
                blocks=blocks
            )
            
            print(f"✅ Benchmark report sent to #{self.channel}")
            
            # Upload detailed results as file if requested
            if upload_file and os.path.exists(upload_file):
                self.client.files_upload_v2(
                    channel=self.channel,
                    file=upload_file,
                    title="Detailed Benchmark Results",
                    initial_comment="📎 Full benchmark data attached"
                )
                print(f"✅ Uploaded file: {upload_file}")
            
            return True
            
        except SlackApiError as e:
            print(f"❌ Failed to send report: {e.response['error']}")
            return False
    
    def send_comparison_report(self, results1, results2, label1="Before", label2="After"):
        """Send comparison report between two benchmark results"""
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "📊 Benchmark Comparison"
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*{label1}* vs *{label2}*"
                }
            },
            {"type": "divider"}
        ]
        
        # Compare benchmarks
        if 'benchmarks' in results1 and 'benchmarks' in results2:
            bench1_dict = {b['name']: b for b in results1['benchmarks']}
            bench2_dict = {b['name']: b for b in results2['benchmarks']}
            
            for name in bench1_dict.keys():
                if name in bench2_dict:
                    b1 = bench1_dict[name]
                    b2 = bench2_dict[name]
                    
                    # Calculate improvement
                    if 'fps' in b1 and 'fps' in b2:
                        fps1 = b1['fps']
                        fps2 = b2['fps']
                        improvement = ((fps2 - fps1) / fps1) * 100
                        
                        emoji = "🚀" if improvement > 0 else "📉"
                        sign = "+" if improvement > 0 else ""
                        
                        blocks.append({
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": f"{emoji} *{name}*\n{fps1:.2f} FPS → {fps2:.2f} FPS ({sign}{improvement:.1f}%)"
                            }
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
                text="Benchmark Comparison",
                blocks=blocks
            )
            print(f"✅ Comparison report sent to #{self.channel}")
            return True
        except SlackApiError as e:
            print(f"❌ Failed to send comparison: {e.response['error']}")
            return False


def main():
    """Main function"""
    
    if not SLACK_TOKEN:
        print("❌ SLACK_BOT_TOKEN environment variable not set")
        print("\nSet it with:")
        print("  export SLACK_BOT_TOKEN='xoxb-your-token-here'")
        sys.exit(1)
    
    reporter = BenchmarkReporter(SLACK_TOKEN, CHANNEL)
    
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 slack_benchmark_reporter.py <results.json>")
        print("  python3 slack_benchmark_reporter.py <before.json> <after.json>")
        print("\nExamples:")
        print("  python3 slack_benchmark_reporter.py gb10_benchmark_results.json")
        print("  python3 slack_benchmark_reporter.py jetson_benchmark_results.json gb10_benchmark_results.json")
        sys.exit(1)
    
    if len(sys.argv) == 2:
        # Single benchmark report
        results_file = sys.argv[1]
        results = reporter.read_benchmark_file(results_file)
        
        if results:
            reporter.send_benchmark_report(
                results,
                title=f"Benchmark Results - {os.path.basename(results_file)}",
                upload_file=results_file
            )
    
    elif len(sys.argv) >= 3:
        # Comparison report
        file1 = sys.argv[1]
        file2 = sys.argv[2]
        
        results1 = reporter.read_benchmark_file(file1)
        results2 = reporter.read_benchmark_file(file2)
        
        if results1 and results2:
            label1 = os.path.basename(file1).replace('_benchmark_results.json', '')
            label2 = os.path.basename(file2).replace('_benchmark_results.json', '')
            
            reporter.send_comparison_report(results1, results2, label1, label2)


if __name__ == "__main__":
    main()

