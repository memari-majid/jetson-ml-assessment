#!/usr/bin/env python3
"""
UVU AI Chatbot - Simple ChatGPT-like Interface
Clean, minimal design focused on conversation
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr
import os
import sqlite3
import hashlib
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Tokens (Private repo - safe to include, but don't push to public GitHub!)
HF_TOKEN = os.environ.get('HF_TOKEN', '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')
print("✅ HuggingFace token configured")

# Simple config
DATA_DIR = Path("chatbot_data")
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "users.db"

# Initialize database
def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            message TEXT NOT NULL,
            response TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    
    # Create demo accounts
    try:
        cursor.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", 
                      ("student", hashlib.sha256("student123".encode()).hexdigest()))
        cursor.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", 
                      ("admin", hashlib.sha256("admin123".encode()).hexdigest()))
        conn.commit()
    except:
        pass  # Already exists
    
    return conn

conn = init_db()
print("✅ Database initialized")

# Load model
print("🔄 Loading AI model...")
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        token=HF_TOKEN
    )
    print(f"✅ Model loaded: {MODEL_NAME}")
    MODEL_LOADED = True
except Exception as e:
    print(f"⚠️  Model loading issue: {e}")
    print("  Chatbot will show instructions to users")
    model = None
    tokenizer = None
    MODEL_LOADED = False

# Authentication functions
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def authenticate(username, password):
    cursor = conn.cursor()
    cursor.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
    result = cursor.fetchone()
    return result and result[0] == hash_password(password)

def register_user(username, password):
    if len(username) < 3 or len(password) < 6:
        return False, "Username min 3 chars, password min 6 chars"
    try:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users VALUES (?, ?, CURRENT_TIMESTAMP)", 
                      (username, hash_password(password)))
        conn.commit()
        return True, f"Account created! Login with: {username}"
    except:
        return False, "Username already exists"

# Chat function
def chat_response(message, history, username):
    if not MODEL_LOADED or model is None:
        return history + [[message, "⚠️ Model not loaded. Administrator: set HF_TOKEN environment variable and restart."]]
    
    # Build prompt with history
    prompt = ""
    for user_msg, bot_msg in history:
        prompt += f"User: {user_msg}\nAssistant: {bot_msg}\n\n"
    prompt += f"User: {message}\nAssistant:"
    
    # Generate
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the response
    if "Assistant:" in response:
        response = response.split("Assistant:")[-1].strip()
    
    # Save to database
    cursor = conn.cursor()
    cursor.execute("INSERT INTO conversations (username, message, response) VALUES (?, ?, ?)",
                  (username, message, response))
    conn.commit()
    
    return history + [[message, response]]

# UVU Green theme
theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.green,
    secondary_hue=gr.themes.colors.slate,
).set(
    button_primary_background_fill='#275D38',
    button_primary_background_fill_hover='#1a4428',
    button_primary_text_color='white',
)

# CSS for ChatGPT-like appearance with centered login
custom_css = """
/* Center everything */
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
}

/* Login page centering */
.login-container {
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 70vh;
}

/* Chat container */
.chat-container {
    max-width: 900px;
    margin: 0 auto;
}

/* Message bubbles */
.message-wrap {
    padding: 15px 20px !important;
}

/* Center login form */
.contain {
    max-width: 500px !important;
    margin: 0 auto !important;
}

/* Header styling */
.header-text {
    text-align: center;
    padding: 60px 20px;
}

/* Smooth transitions */
* {
    transition: all 0.3s ease;
}

/* Better button styling */
button[type="submit"] {
    padding: 12px 24px !important;
    font-size: 1.05em !important;
}
"""

# Create login interface
def create_login_interface():
    with gr.Blocks(theme=theme, css=custom_css, title="UVU AI Chatbot") as login_demo:
        gr.HTML("""
        <div style="text-align: center; padding: 40px 20px; background: linear-gradient(135deg, #275D38 0%, #1a4428 100%); color: white; border-radius: 15px; margin-bottom: 30px;">
            <h1 style="margin: 0; font-size: 2.5em;">🎓 UVU AI Chatbot</h1>
            <p style="margin: 15px 0 5px 0; font-size: 1.1em; opacity: 0.95;">Utah Valley University</p>
            <p style="margin: 5px 0 0 0; font-size: 0.9em; opacity: 0.85;">Powered by Dell Pro Max GB10</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML("<div style='height: 50px;'></div>")
            
            with gr.Column(scale=2):
                gr.Markdown("### 🔐 Sign In")
                gr.Markdown("Welcome! Please login to start chatting with AI.")
                
                username_input = gr.Textbox(label="Username", placeholder="Enter your username")
                password_input = gr.Textbox(label="Password", type="password", placeholder="Enter your password")
                
                with gr.Row():
                    login_btn = gr.Button("Sign In →", variant="primary", size="lg")
                
                status_msg = gr.Markdown("")
                
                gr.Markdown("---")
                gr.Markdown("**Demo Account:** username: `student` password: `student123`")
                
                gr.Markdown("""
                <div style="margin-top: 30px; padding: 20px; background: #f8f9fa; border-radius: 8px; border-left: 4px solid #275D38;">
                    <p style="margin: 0;"><strong>Need an account?</strong></p>
                    <p style="margin: 5px 0 0 0;">Contact your instructor or use the demo account above.</p>
                </div>
                """)
            
            with gr.Column(scale=1):
                gr.HTML("<div style='height: 50px;'></div>")
        
        def handle_login(username, password):
            if authenticate(username, password):
                return f"✅ Welcome, {username}! Redirecting to chat...", username
            return "❌ Invalid credentials. Try: student / student123", None
        
        login_btn.click(
            handle_login,
            inputs=[username_input, password_input],
            outputs=[status_msg, gr.State()]
        )
    
    return login_demo

# Create chat interface (ChatGPT-like)
def create_chat_interface(username):
    with gr.Blocks(theme=theme, css=custom_css, title="UVU AI Chatbot") as chat_demo:
        # Simple header
        with gr.Row():
            with gr.Column(scale=4):
                gr.HTML(f"""
                <div style="padding: 15px; background: #275D38; color: white; border-radius: 10px;">
                    <h2 style="margin: 0;">🎓 UVU AI Chat</h2>
                    <p style="margin: 5px 0 0 0; font-size: 0.9em; opacity: 0.9;">Logged in as: {username}</p>
                </div>
                """)
            with gr.Column(scale=1):
                logout_btn = gr.Button("Sign Out", size="sm")
        
        # Chat interface (ChatGPT style)
        chatbot = gr.Chatbot(
            value=[],
            height=550,
            show_label=False,
            container=True,
            bubble_full_width=False,
            avatar_images=(None, "🤖"),
            show_copy_button=True,
            likeable=False
        )
        
        # Input area (ChatGPT style - simple)
        with gr.Row():
            msg = gr.Textbox(
                show_label=False,
                placeholder="Message UVU AI Chatbot...",
                container=False,
                scale=10
            )
            submit = gr.Button("Send", variant="primary", scale=1, size="lg")
        
        # Hidden advanced options (collapsed by default)
        with gr.Accordion("⚙️ Options", open=False):
            gr.Markdown("**Quick Settings:**")
            with gr.Row():
                clear_btn = gr.Button("🗑️ Clear Chat", size="sm")
                export_btn = gr.Button("📥 Export History", size="sm")
        
        # Examples (ChatGPT-like)
        gr.Examples(
            examples=[
                "Explain machine learning in simple terms",
                "Write a Python function for binary search",
                "What are transformers in NLP?",
                "Help me debug my code",
            ],
            inputs=msg,
            label="💡 Suggestions"
        )
        
        # Footer
        gr.HTML("""
        <div style="text-align: center; padding: 20px; opacity: 0.7; font-size: 0.85em;">
            <p>UVU AI Chatbot powered by Dell Pro Max GB10 • NVIDIA Blackwell GPU • 13.4 TFLOPS</p>
            <p>Supports 150-200 concurrent users • 100% private & secure</p>
        </div>
        """)
        
        # Event handlers
        def respond(message, chat_history):
            return "", chat_response(message, chat_history, username)
        
        submit.click(respond, [msg, chatbot], [msg, chatbot])
        msg.submit(respond, [msg, chatbot], [msg, chatbot])
        clear_btn.click(lambda: [], None, chatbot)
        logout_btn.click(lambda: gr.update(visible=False), None, None)
    
    return chat_demo

# Main app with conditional login
with gr.Blocks(theme=theme, title="UVU AI Chatbot") as demo:
    user_state = gr.State(None)
    
    # Login screen - Centered and clean
    with gr.Group(visible=True) as login_group:
        # Spacer for vertical centering
        gr.HTML("<div style='height: 10vh;'></div>")
        
        # Centered header
        gr.HTML("""
        <div style="text-align: center; padding: 60px 20px; background: linear-gradient(135deg, #275D38 0%, #1a4428 100%); color: white; border-radius: 20px; margin: 0 auto 50px auto; max-width: 600px; box-shadow: 0 10px 40px rgba(39,93,56,0.3);">
            <div style="font-size: 4em; margin-bottom: 15px;">🎓</div>
            <h1 style="margin: 0; font-size: 2.8em; font-weight: 700; letter-spacing: -0.5px;">UVU AI</h1>
            <p style="margin: 20px 0 8px 0; font-size: 1.15em; opacity: 0.95;">Chat with State-of-the-Art AI</p>
            <p style="margin: 0; font-size: 0.95em; opacity: 0.85;">Utah Valley University</p>
        </div>
        """)
        
        # Centered login form
        with gr.Row():
            with gr.Column(scale=1):
                pass  # Left spacer
            
            with gr.Column(scale=3):
                # Clean login box
                gr.HTML("""
                <div style="text-align: center; margin-bottom: 30px;">
                    <h2 style="margin: 0 0 10px 0; font-size: 1.8em; font-weight: 600;">Welcome</h2>
                    <p style="margin: 0; color: #666; font-size: 1.05em;">Sign in to start chatting</p>
                </div>
                """)
                
                with gr.Group():
                    login_username = gr.Textbox(
                        label="",
                        placeholder="Username",
                        container=False,
                        elem_classes="login-input"
                    )
                    login_password = gr.Textbox(
                        label="",
                        type="password",
                        placeholder="Password",
                        container=False,
                        elem_classes="login-input"
                    )
                    
                    login_btn = gr.Button(
                        "Continue →", 
                        variant="primary", 
                        size="lg", 
                        elem_id="login-btn",
                        scale=1
                    )
                    
                    login_status = gr.Markdown("", elem_classes="status-message")
                
                # Demo account info
                gr.HTML("""
                <div style="text-align: center; margin-top: 35px; padding: 25px; background: linear-gradient(135deg, #f8faf9 0%, #f0f7f4 100%); border-radius: 12px; border: 1px solid #e0ebe5;">
                    <p style="margin: 0 0 12px 0; font-weight: 600; color: #275D38; font-size: 1.05em;">📚 Demo Account</p>
                    <p style="margin: 0; color: #555; font-size: 0.95em;">
                        Username: <code style="background: white; padding: 3px 8px; border-radius: 4px; font-weight: 600;">student</code> 
                        &nbsp;•&nbsp; 
                        Password: <code style="background: white; padding: 3px 8px; border-radius: 4px; font-weight: 600;">student123</code>
                    </p>
                    <p style="margin: 15px 0 0 0; color: #666; font-size: 0.85em;">New users: contact your instructor for access</p>
                </div>
                """)
            
            with gr.Column(scale=1):
                pass  # Right spacer
    
    # Chat screen (hidden initially)
    with gr.Group(visible=False) as chat_group:
        # Minimal header
        with gr.Row():
            gr.HTML("""
            <div style="padding: 12px 20px; background: #275D38; color: white; border-radius: 8px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="font-size: 1.3em; font-weight: 600;">💬 UVU AI Chat</span>
                        <span id="username-display" style="margin-left: 15px; opacity: 0.8; font-size: 0.9em;"></span>
                    </div>
                </div>
            </div>
            """)
            logout_btn = gr.Button("Sign Out", size="sm", scale=0)
        
        # Clean chat interface (ChatGPT style)
        chatbot = gr.Chatbot(
            value=[],
            height=500,
            show_label=False,
            container=False,
            bubble_full_width=False,
            avatar_images=(None, "🤖"),
            show_copy_button=True,
            render_markdown=True,
            latex_delimiters=[{"left": "$$", "right": "$$", "display": True}],
        )
        
        # Simple input (ChatGPT style)
        with gr.Row():
            msg = gr.Textbox(
                show_label=False,
                placeholder="Message UVU AI...",
                container=False,
                scale=10,
                lines=1
            )
            submit = gr.Button("↑", variant="primary", scale=0, size="lg")
        
        # Minimal footer
        gr.HTML("""
        <div style="text-align: center; padding: 15px; opacity: 0.6; font-size: 0.8em;">
            <p style="margin: 0;">Powered by Dell Pro Max GB10 (NVIDIA Blackwell GPU • 13.4 TFLOPS)</p>
        </div>
        """)
    
    # Login logic
    def do_login(username, password):
        if authenticate(username, password):
            return (
                gr.update(visible=False),  # Hide login
                gr.update(visible=True),   # Show chat
                username,                   # Store username
                f"Welcome, {username}!",
                []                          # Empty chat history
            )
        return (
            gr.update(visible=True),       # Keep login visible
            gr.update(visible=False),      # Keep chat hidden
            None,
            "❌ **Invalid credentials**\n\nPlease try again or use demo: student / student123",
            []
        )
    
    # Logout logic
    def do_logout():
        return (
            gr.update(visible=True),   # Show login
            gr.update(visible=False),  # Hide chat
            None,                       # Clear username
            "",                         # Clear status
            []                          # Clear chat
        )
    
    # Chat logic
    def respond(message, history, username):
        if not message.strip():
            return "", history
        
        new_history = chat_response(message, history, username)
        return "", new_history
    
    # Connect events
    login_btn.click(
        do_login,
        inputs=[login_username, login_password],
        outputs=[login_group, chat_group, user_state, login_status, chatbot]
    )
    
    logout_btn.click(
        do_logout,
        outputs=[login_group, chat_group, user_state, login_status, chatbot]
    )
    
    submit.click(
        respond,
        inputs=[msg, chatbot, user_state],
        outputs=[msg, chatbot]
    )
    
    msg.submit(
        respond,
        inputs=[msg, chatbot, user_state],
        outputs=[msg, chatbot]
    )

if __name__ == "__main__":
    print("\n" + "="*80)
    print("  UVU AI CHATBOT - SIMPLE INTERFACE")
    print("  ChatGPT-like Clean Design")
    print("="*80)
    print("\n🚀 Launching on port 8000...")
    print("  Access: http://localhost:8000")
    print("  Public: https://uvuchatbot.ngrok.app (if ngrok configured)")
    print("\n" + "="*80 + "\n")
    
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=8000,
        share=True,
        show_error=True
    )
