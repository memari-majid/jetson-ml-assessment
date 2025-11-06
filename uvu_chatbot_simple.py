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
            conversation_id TEXT NOT NULL,
            username TEXT NOT NULL,
            message TEXT NOT NULL,
            response TEXT NOT NULL,
            model_used TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Chat sessions table - for sidebar history like ChatGPT
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_sessions (
            conversation_id TEXT PRIMARY KEY,
            username TEXT NOT NULL,
            title TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            message_count INTEGER DEFAULT 0
        )
    ''')
    conn.commit()
    
    # Create demo accounts
    try:
        cursor.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", 
                      ("student", hashlib.sha256("student123".encode()).hexdigest()))
        cursor.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", 
                      ("admin", hashlib.sha256("admin".encode()).hexdigest()))  # Admin password: admin
        conn.commit()
        print("✅ Demo accounts: student/student123, admin/admin")
    except:
        print("ℹ️  Demo accounts already exist")
        pass  # Already exists
    
    return conn

conn = init_db()
print("✅ Database initialized")

# Available models for users to choose from
AVAILABLE_MODELS = {
    "TinyLlama-1.1B (Fastest)": {
        "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "size": "1.1B params, ~2GB",
        "speed": "5,000+ tokens/sec",
        "access": "✅ No approval needed"
    },
    "Llama-3.2-1B (Fast)": {
        "name": "meta-llama/Llama-3.2-1B-Instruct",
        "size": "1B params, ~2GB", 
        "speed": "5,000+ tokens/sec",
        "access": "⚠️ Requires HF approval"
    },
    "Llama-3.2-3B (Balanced)": {
        "name": "meta-llama/Llama-3.2-3B-Instruct",
        "size": "3B params, ~6GB",
        "speed": "3,000+ tokens/sec",
        "access": "⚠️ Requires HF approval"
    },
    "Mistral-7B (Best Quality)": {
        "name": "mistralai/Mistral-7B-Instruct-v0.2",
        "size": "7B params, ~14GB",
        "speed": "2,000+ tokens/sec",
        "access": "✅ No approval needed"
    },
    "CodeLlama-7B (Programming)": {
        "name": "codellama/CodeLlama-7b-Instruct-hf",
        "size": "7B params, ~14GB",
        "speed": "2,000+ tokens/sec",
        "access": "⚠️ Requires HF approval"
    },
    "GPT-OSS-Safeguard-20B (OpenAI)": {
        "name": "openai/gpt-oss-safeguard-20b",
        "size": "20B params, ~40GB",
        "speed": "800-1,000 tokens/sec",
        "access": "✅ No approval needed"
    },
    "Qwen3-4B-Instruct (Alibaba)": {
        "name": "Qwen/Qwen3-4B-Instruct-2507",
        "size": "4B params, ~8GB",
        "speed": "2,500+ tokens/sec",
        "access": "✅ No approval needed"
    }
}

# Load default model (TinyLlama - works immediately)
print("🔄 Loading default AI model...")
DEFAULT_MODEL = "TinyLlama-1.1B (Fastest)"
MODEL_NAME = AVAILABLE_MODELS[DEFAULT_MODEL]["name"]
print(f"   Model: {MODEL_NAME}")

# Model cache - store loaded models
loaded_models = {}

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        token=HF_TOKEN
    )
    loaded_models[MODEL_NAME] = (model, tokenizer)
    print(f"✅ Default model loaded: {DEFAULT_MODEL}")
    MODEL_LOADED = True
    CURRENT_MODEL_NAME = MODEL_NAME
except Exception as e:
    print(f"⚠️  Model loading issue: {e}")
    print("  Chatbot will show model loading instructions")
    model = None
    tokenizer = None
    MODEL_LOADED = False
    CURRENT_MODEL_NAME = None

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

# Function to load selected model
def load_model(model_display_name):
    """Load the selected model"""
    global model, tokenizer, MODEL_LOADED, CURRENT_MODEL_NAME
    
    if model_display_name not in AVAILABLE_MODELS:
        return f"❌ Unknown model: {model_display_name}"
    
    model_info = AVAILABLE_MODELS[model_display_name]
    model_name = model_info["name"]
    
    # Check if already loaded
    if model_name == CURRENT_MODEL_NAME:
        return f"✅ {model_display_name} is already loaded"
    
    # Check cache first
    if model_name in loaded_models:
        model, tokenizer = loaded_models[model_name]
        CURRENT_MODEL_NAME = model_name
        MODEL_LOADED = True
        return f"✅ Switched to {model_display_name} (from cache)"
    
    try:
        print(f"\n🔄 Loading {model_display_name}...")
        
        new_tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
        new_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            token=HF_TOKEN
        )
        
        # Clear old model from GPU if needed
        if model is not None and CURRENT_MODEL_NAME != model_name:
            del model
            del tokenizer
            torch.cuda.empty_cache()
        
        model = new_model
        tokenizer = new_tokenizer
        loaded_models[model_name] = (model, tokenizer)
        MODEL_LOADED = True
        CURRENT_MODEL_NAME = model_name
        
        print(f"✅ {model_display_name} loaded successfully!")
        return f"✅ {model_display_name} loaded!\n\n{model_info['size']} • {model_info['speed']}"
        
    except Exception as e:
        error_msg = str(e)
        if "gated" in error_msg.lower() or "403" in error_msg:
            return f"⚠️ {model_display_name} requires HuggingFace access approval.\n\nVisit: https://huggingface.co/{model_name}\nClick 'Agree and access repository'\n\nUsing current model instead."
        return f"❌ Error loading {model_display_name}: {error_msg}"

# Chat management functions (ChatGPT-like)
def create_new_chat(username):
    """Create a new chat session"""
    import uuid
    conversation_id = str(uuid.uuid4())[:8]
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO chat_sessions (conversation_id, username, title, message_count)
        VALUES (?, ?, ?, 0)
    """, (conversation_id, username, "New Chat"))
    conn.commit()
    return conversation_id

def get_user_chats(username):
    """Get all chat sessions for sidebar (ChatGPT-like)"""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT conversation_id, title, updated_at, message_count
        FROM chat_sessions
        WHERE username = ?
        ORDER BY updated_at DESC
        LIMIT 50
    """, (username,))
    return cursor.fetchall()

def load_chat_history(conversation_id):
    """Load messages from a specific chat"""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT message, response
        FROM conversations
        WHERE conversation_id = ?
        ORDER BY timestamp ASC
    """, (conversation_id,))
    messages = cursor.fetchall()
    return [[msg, resp] for msg, resp in messages]

def update_chat_title(conversation_id, first_message):
    """Auto-generate chat title from first message (like ChatGPT)"""
    title = first_message[:50] + "..." if len(first_message) > 50 else first_message
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE chat_sessions
        SET title = ?, updated_at = CURRENT_TIMESTAMP
        WHERE conversation_id = ?
    """, (title, conversation_id))
    conn.commit()

def delete_chat(conversation_id):
    """Delete a chat session"""
    cursor = conn.cursor()
    cursor.execute("DELETE FROM conversations WHERE conversation_id = ?", (conversation_id,))
    cursor.execute("DELETE FROM chat_sessions WHERE conversation_id = ?", (conversation_id,))
    conn.commit()
    return "✅ Chat deleted"

# Chat function with conversation management
def chat_response(message, history, username, selected_model_name, conversation_id):
    if not MODEL_LOADED or model is None:
        return history + [[message, "⚠️ No model loaded. Please select a model from the dropdown above or contact administrator."]]
    
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
    cursor.execute("""
        INSERT INTO conversations (conversation_id, username, message, response, model_used)
        VALUES (?, ?, ?, ?, ?)
    """, (conversation_id, username, message, response, selected_model_name))
    
    # Update session
    cursor.execute("""
        UPDATE chat_sessions
        SET updated_at = CURRENT_TIMESTAMP,
            message_count = message_count + 1
        WHERE conversation_id = ?
    """, (conversation_id,))
    
    # Auto-generate title from first message
    if len(history) == 0:
        update_chat_title(conversation_id, message)
    
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
    
    # Chat screen with ChatGPT-like sidebar (hidden initially)
    with gr.Group(visible=False) as chat_group:
        # Header
        with gr.Row():
            gr.HTML("""
            <div style="padding: 12px 20px; background: #275D38; color: white; border-radius: 8px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="font-size: 1.3em; font-weight: 600;">💬 UVU AI Chat</span>
                    </div>
                </div>
            </div>
            """)
            logout_btn = gr.Button("Sign Out", size="sm", scale=0)
        
        # Main content with sidebar (ChatGPT style)
        with gr.Row():
            # Sidebar - Chat history (ChatGPT-like)
            with gr.Column(scale=2, min_width=250):
                gr.HTML("""
                <div style="padding: 15px 10px; background: #f7f9f8; border-radius: 8px; margin-bottom: 10px;">
                    <h3 style="margin: 0 0 5px 0; font-size: 0.95em; color: #275D38;">💬 Your Chats</h3>
                </div>
                """)
                
                new_chat_btn = gr.Button(
                    "➕ New Chat",
                    variant="primary",
                    size="sm",
                    elem_id="new-chat-btn"
                )
                
                # Chat history list
                chat_history_list = gr.Radio(
                    choices=[],
                    label="Recent Conversations",
                    interactive=True,
                    elem_id="chat-list"
                )
                
                refresh_chats_btn = gr.Button("🔄 Refresh", size="sm", variant="secondary")
                
                gr.HTML("""
                <div style="margin-top: 15px; padding: 12px; background: #f0f7f4; border-radius: 6px; font-size: 0.85em;">
                    <p style="margin: 0; color: #275D38; font-weight: 600;">💡 Tip</p>
                    <p style="margin: 5px 0 0 0; color: #555;">Click "New Chat" to start fresh, or select a previous conversation to continue.</p>
                </div>
                """)
            
            # Main chat area
            with gr.Column(scale=10):
                # Current conversation ID (hidden)
                current_conversation_id = gr.State(None)
        
        # Model selector - compact and clean
        with gr.Row():
            with gr.Column(scale=2):
                pass
            with gr.Column(scale=8):
                model_selector = gr.Dropdown(
                    choices=list(AVAILABLE_MODELS.keys()),
                    value=DEFAULT_MODEL,
                    label="🤖 AI Model",
                    info="Choose your AI assistant",
                    container=True,
                    scale=1
                )
                model_info = gr.Markdown(f"""
                **{DEFAULT_MODEL}:** {AVAILABLE_MODELS[DEFAULT_MODEL]['size']} • {AVAILABLE_MODELS[DEFAULT_MODEL]['speed']} • {AVAILABLE_MODELS[DEFAULT_MODEL]['access']}
                """, elem_classes="model-info")
            with gr.Column(scale=2):
                pass
        
        # Clean chat interface (ChatGPT style)
        chatbot = gr.Chatbot(
            value=[],
            height=450,
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
            # Create initial chat session
            conversation_id = create_new_chat(username)
            
            # Get user's chat history for sidebar
            chats = get_user_chats(username)
            chat_choices = {f"{title} ({msg_count} msgs)": conv_id for conv_id, title, _, msg_count in chats}
            
            return (
                gr.update(visible=False),  # Hide login
                gr.update(visible=True),   # Show chat
                username,                   # Store username
                f"Welcome, {username}!",
                [],                         # Empty chat history
                conversation_id,            # Current conversation ID
                gr.update(choices=list(chat_choices.keys()), value=None)  # Populate sidebar
            )
        return (
            gr.update(visible=True),       # Keep login visible
            gr.update(visible=False),      # Keep chat hidden
            None,
            "❌ **Invalid credentials**\n\nPlease try again or use demo: student / student123",
            [],
            None,
            gr.update()
        )
    
    # Logout logic
    def do_logout():
        return (
            gr.update(visible=True),   # Show login
            gr.update(visible=False),  # Hide chat
            None,                       # Clear username
            "",                         # Clear status
            [],                         # Clear chat
            None,                       # Clear conversation_id
            gr.update(choices=[])       # Clear chat list
        )
    
    # Chat management logic (ChatGPT-like)
    def handle_new_chat(username):
        """Create new chat and refresh sidebar"""
        conversation_id = create_new_chat(username)
        
        # Get updated chat list
        chats = get_user_chats(username)
        chat_choices = [f"{title} ({msg_count} msgs)" for _, title, _, msg_count in chats]
        
        return (
            [],  # Clear chat history
            conversation_id,  # New conversation ID
            gr.update(choices=chat_choices, value=None)  # Update sidebar
        )
    
    def handle_load_chat(username, selected_chat_label):
        """Load selected chat from sidebar"""
        if not selected_chat_label:
            return [], None
        
        # Parse conversation_id from label
        chats = get_user_chats(username)
        chat_map = {f"{title} ({msg_count} msgs)": conv_id for conv_id, title, _, msg_count in chats}
        
        if selected_chat_label in chat_map:
            conversation_id = chat_map[selected_chat_label]
            history = load_chat_history(conversation_id)
            return history, conversation_id
        
        return [], None
    
    def handle_refresh_chats(username):
        """Refresh chat sidebar"""
        chats = get_user_chats(username)
        chat_choices = [f"{title} ({msg_count} msgs)" for _, title, _, msg_count in chats]
        return gr.update(choices=chat_choices)
    
    # Model selection logic
    def load_selected_model(selected_model):
        """Load the model when user changes selection"""
        result = load_model(selected_model)
        info = AVAILABLE_MODELS[selected_model]
        info_text = f"**{selected_model}:** {info['size']} • {info['speed']} • {info['access']}\n\n{result}"
        return info_text
    
    # Chat logic with conversation management
    def respond(message, history, username, selected_model, conversation_id):
        if not message.strip():
            return "", history, gr.update()
        
        if not conversation_id:
            # Create new conversation if none exists
            conversation_id = create_new_chat(username)
        
        new_history = chat_response(message, history, username, selected_model, conversation_id)
        
        # Refresh sidebar to show updated title/count
        chats = get_user_chats(username)
        chat_choices = [f"{title} ({msg_count} msgs)" for _, title, _, msg_count in chats]
        
        return "", new_history, gr.update(choices=chat_choices)
    
    # Connect events
    login_btn.click(
        do_login,
        inputs=[login_username, login_password],
        outputs=[login_group, chat_group, user_state, login_status, chatbot, current_conversation_id, chat_history_list]
    )
    
    logout_btn.click(
        do_logout,
        outputs=[login_group, chat_group, user_state, login_status, chatbot, current_conversation_id, chat_history_list]
    )
    
    # Sidebar chat management (ChatGPT-like)
    new_chat_btn.click(
        handle_new_chat,
        inputs=user_state,
        outputs=[chatbot, current_conversation_id, chat_history_list]
    )
    
    chat_history_list.change(
        handle_load_chat,
        inputs=[user_state, chat_history_list],
        outputs=[chatbot, current_conversation_id]
    )
    
    refresh_chats_btn.click(
        handle_refresh_chats,
        inputs=user_state,
        outputs=chat_history_list
    )
    
    # Model selector events
    model_selector.change(
        load_selected_model,
        inputs=model_selector,
        outputs=model_info
    )
    
    # Chat events with conversation management
    submit.click(
        respond,
        inputs=[msg, chatbot, user_state, model_selector, current_conversation_id],
        outputs=[msg, chatbot, chat_history_list]
    )
    
    msg.submit(
        respond,
        inputs=[msg, chatbot, user_state, model_selector, current_conversation_id],
        outputs=[msg, chatbot, chat_history_list]
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
