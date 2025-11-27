import os
import json
from typing import Dict, Any, List, Optional
import streamlit as st
import yaml

# --- LLM client libraries ---
from openai import OpenAI
import google.generativeai as genai
from anthropic import Anthropic

# -----------------------------------------------------------
# WOW Theme Configuration
# -----------------------------------------------------------

WOW_THEMES = {
    "light": {
        "primary": "#FFD700",
        "secondary": "#4169E1",
        "background": "#F5F5DC",
        "text": "#2C1810",
        "accent": "#FF6347",
    },
    "dark": {
        "primary": "#FFD700",
        "secondary": "#1E90FF",
        "background": "#1A1A1A",
        "text": "#E0E0E0",
        "accent": "#FF4500",
    }
}

WOW_ART_STYLES = {
    "Classic": {"icon": "⚔️", "description": "Original WoW aesthetic", "color": "#8B4513"},
    "Burning Crusade": {"icon": "🔥", "description": "Outland flames", "color": "#DC143C"},
    "Wrath": {"icon": "❄️", "description": "Icy Northrend", "color": "#4682B4"},
    "Cataclysm": {"icon": "🌋", "description": "Elemental chaos", "color": "#FF8C00"},
    "Mists": {"icon": "🐼", "description": "Pandaren serenity", "color": "#3CB371"},
    "Warlords": {"icon": "⚡", "description": "Iron Horde", "color": "#B8860B"},
    "Legion": {"icon": "👹", "description": "Fel corruption", "color": "#9370DB"},
    "Battle": {"icon": "⚓", "description": "Alliance vs Horde", "color": "#CD853F"},
    "Shadowlands": {"icon": "💀", "description": "Afterlife realms", "color": "#483D8B"},
    "Dragonflight": {"icon": "🐉", "description": "Dragon Isles", "color": "#FF6347"},
}

TRANSLATIONS = {
    "en": {
        "title": "Multi-Agent Workflow Studio",
        "subtitle": "Hero Class: AI Orchestrator",
        "theme": "Theme",
        "language": "Language",
        "art_style": "Art Style",
        "health": "Health",
        "mana": "Mana",
        "experience": "Experience",
        "api_keys": "API Keys",
        "input": "Input",
        "pipeline": "Pipeline",
        "smart_replace": "Smart Replace",
        "notes": "AI Note Keeper",
        "dashboard": "Dashboard",
        "run": "Cast Spell",
        "level": "Level",
        "quest_log": "Quest Log",
        "achievements": "Achievements",
    },
    "zh": {
        "title": "多代理工作流程工作室",
        "subtitle": "英雄職業：人工智能協調者",
        "theme": "主題",
        "language": "語言",
        "art_style": "藝術風格",
        "health": "生命值",
        "mana": "法力值",
        "experience": "經驗值",
        "api_keys": "API 密鑰",
        "input": "輸入",
        "pipeline": "管道",
        "smart_replace": "智能替換",
        "notes": "人工智能筆記",
        "dashboard": "儀表板",
        "run": "施放法術",
        "level": "等級",
        "quest_log": "任務日誌",
        "achievements": "成就",
    }
}

# -----------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------

def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        "theme": "dark",
        "language": "en",
        "art_style": "Dragonflight",
        "player_level": 1,
        "health": 100,
        "mana": 100,
        "experience": 0,
        "quests_completed": 0,
        "achievements": [],
        "combat_log": [],
        "template": "## Template\n\nWrite your template here...",
        "observations": "Add your observations here...",
        "pipeline_history": [],
        "note_raw_text": "",
        "note_chat_history": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# -----------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------

@st.cache_data
def load_agents_config(path: str = "agents.yaml") -> Dict[str, Any]:
    """Load agents configuration from YAML file"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {"agents": [], "pipelines": []}

def get_translation(key: str) -> str:
    """Get translated text based on current language"""
    lang = st.session_state.get("language", "en")
    return TRANSLATIONS.get(lang, TRANSLATIONS["en"]).get(key, key)

def apply_custom_css():
    """Apply WOW-themed custom CSS"""
    theme = st.session_state.get("theme", "dark")
    style = st.session_state.get("art_style", "Dragonflight")
    colors = WOW_THEMES[theme]
    accent_color = WOW_ART_STYLES[style]["color"]
    
    css = f"""
    <style>
    /* Main theme colors */
    .stApp {{
        background-color: {colors['background']};
        color: {colors['text']};
    }}
    
    /* Headers with WOW style */
    h1, h2, h3 {{
        color: {colors['primary']};
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        font-family: 'Trebuchet MS', sans-serif;
        border-bottom: 3px solid {accent_color};
        padding-bottom: 10px;
    }}
    
    /* Buttons with WOW style */
    .stButton > button {{
        background: linear-gradient(145deg, {accent_color}, {colors['secondary']});
        color: white;
        border: 2px solid {colors['primary']};
        border-radius: 8px;
        font-weight: bold;
        text-transform: uppercase;
        padding: 10px 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
    }}
    
    .stButton > button:hover {{
        transform: scale(1.05);
        box-shadow: 0 6px 12px rgba(0,0,0,0.5);
    }}
    
    /* Status bars */
    .status-bar {{
        background: linear-gradient(90deg, {accent_color}, transparent);
        border: 2px solid {colors['primary']};
        border-radius: 10px;
        padding: 5px;
        margin: 5px 0;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.3);
    }}
    
    /* Card style */
    .wow-card {{
        background: {colors['background']};
        border: 3px solid {accent_color};
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.4);
    }}
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background-color: rgba(0,0,0,0.2);
        border-radius: 10px;
        padding: 5px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background-color: {colors['secondary']};
        color: white;
        border-radius: 8px;
        font-weight: bold;
        border: 2px solid {colors['primary']};
    }}
    
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(145deg, {accent_color}, {colors['primary']});
    }}
    
    /* Input fields */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {{
        background-color: rgba(0,0,0,0.3);
        color: {colors['text']};
        border: 2px solid {accent_color};
        border-radius: 8px;
    }}
    
    /* Sidebar */
    .css-1d391kg {{
        background-color: {colors['background']};
        border-right: 3px solid {accent_color};
    }}
    
    /* Progress bars */
    .stProgress > div > div > div > div {{
        background-color: {accent_color};
    }}
    
    /* Expander */
    .streamlit-expanderHeader {{
        background-color: {colors['secondary']};
        color: white;
        border-radius: 8px;
        font-weight: bold;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def update_player_stats(action: str):
    """Update player stats based on actions"""
    if action == "quest_complete":
        st.session_state.experience += 10
        st.session_state.quests_completed += 1
        if st.session_state.experience >= st.session_state.player_level * 50:
            st.session_state.player_level += 1
            st.session_state.experience = 0
            st.toast(f"🎉 Level Up! You are now level {st.session_state.player_level}!")
    elif action == "use_mana":
        st.session_state.mana = max(0, st.session_state.mana - 20)
    elif action == "regenerate":
        st.session_state.mana = min(100, st.session_state.mana + 10)
        st.session_state.health = min(100, st.session_state.health + 5)

def add_combat_log(message: str, message_type: str = "info"):
    """Add entry to combat log"""
    icons = {"info": "ℹ️", "success": "✅", "warning": "⚠️", "error": "❌", "spell": "🔮"}
    log_entry = {
        "icon": icons.get(message_type, "ℹ️"),
        "message": message,
        "timestamp": st.session_state.get("quests_completed", 0)
    }
    if "combat_log" not in st.session_state:
        st.session_state.combat_log = []
    st.session_state.combat_log.append(log_entry)
    if len(st.session_state.combat_log) > 20:
        st.session_state.combat_log.pop(0)

# -----------------------------------------------------------
# API Key Management
# -----------------------------------------------------------

def get_api_key_from_env_or_ui(
    provider_name: str,
    env_var: str,
    session_key: str,
    label: str,
) -> Optional[str]:
    """Get API key from environment or user input"""
    env_val = os.getenv(env_var)
    if env_val:
        st.caption(f"🔑 {label}: Loaded from environment")
        st.session_state[session_key] = env_val
        return env_val

    key = st.text_input(
        label,
        value=st.session_state.get(session_key, ""),
        type="password",
    )
    if key:
        st.session_state[session_key] = key
        st.caption(f"🔑 {label} stored in session")
        return key
    return None

# -----------------------------------------------------------
# LLM Call Router
# -----------------------------------------------------------

def call_llm(
    provider: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> str:
    """Route LLM calls to appropriate provider"""
    provider = provider.lower().strip()
    
    add_combat_log(f"Casting {provider} spell with {model}", "spell")
    update_player_stats("use_mana")

    if provider == "openai":
        api_key = st.session_state.get("openai_api_key")
        if not api_key:
            raise RuntimeError("OpenAI API key is not set.")
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content

    elif provider == "gemini":
        api_key = st.session_state.get("gemini_api_key")
        if not api_key:
            raise RuntimeError("Gemini API key is not set.")
        genai.configure(api_key=api_key)
        model_obj = genai.GenerativeModel(model)
        resp = model_obj.generate_content(
            system_prompt + "\n\nUSER MESSAGE:\n" + user_prompt
        )
        return resp.text

    elif provider == "xai":
        api_key = st.session_state.get("xai_api_key")
        if not api_key:
            raise RuntimeError("xAI API key is not set.")
        client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content

    elif provider == "anthropic":
        api_key = st.session_state.get("anthropic_api_key")
        if not api_key:
            raise RuntimeError("Anthropic API key is not set.")
        client = Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        if resp.content and len(resp.content) > 0:
            block = resp.content[0]
            if hasattr(block, "text"):
                return block.text
        return json.dumps(resp.model_dump(), indent=2)

    else:
        raise ValueError(f"Unsupported provider: {provider}")

def run_agent(
    agent_cfg: Dict[str, Any],
    user_prompt: str,
    override_provider: Optional[str] = None,
    override_model: Optional[str] = None,
    override_system_prompt: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> str:
    """Run a single agent"""
    provider = override_provider or agent_cfg.get("provider", "openai")
    model = override_model or agent_cfg.get("default_model", "gpt-4o-mini")
    system_prompt = override_system_prompt or agent_cfg.get("system_prompt", "")
    return call_llm(
        provider=provider,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )

# -----------------------------------------------------------
# WOW Status Indicators
# -----------------------------------------------------------

def render_status_indicators():
    """Render WOW-style status bars"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"### {get_translation('level')} {st.session_state.player_level}")
        
    with col2:
        st.markdown(f"### {get_translation('health')}")
        st.progress(st.session_state.health / 100)
        st.caption(f"{st.session_state.health}/100")
        
    with col3:
        st.markdown(f"### {get_translation('mana')}")
        st.progress(st.session_state.mana / 100)
        st.caption(f"{st.session_state.mana}/100")
        
    with col4:
        st.markdown(f"### {get_translation('experience')}")
        max_xp = st.session_state.player_level * 50
        st.progress(st.session_state.experience / max_xp)
        st.caption(f"{st.session_state.experience}/{max_xp}")

def render_combat_log():
    """Render WOW-style combat log"""
    st.markdown("### ⚔️ Combat Log")
    with st.expander("View Recent Actions", expanded=False):
        if st.session_state.combat_log:
            for entry in reversed(st.session_state.combat_log[-10:]):
                st.markdown(f"{entry['icon']} {entry['message']}")
        else:
            st.info("No recent actions")

# -----------------------------------------------------------
# Magic Wheel for Art Style Selection
# -----------------------------------------------------------

def render_magic_wheel():
    """Render interactive magic wheel for style selection"""
    st.markdown("### 🎨 Magic Wheel - Art Style Selection")
    
    # Create a grid layout for the magic wheel
    cols = st.columns(5)
    styles = list(WOW_ART_STYLES.keys())
    
    for idx, style in enumerate(styles):
        with cols[idx % 5]:
            style_data = WOW_ART_STYLES[style]
            button_label = f"{style_data['icon']} {style}"
            
            if st.button(
                button_label,
                key=f"style_{style}",
                help=style_data['description'],
                use_container_width=True
            ):
                st.session_state.art_style = style
                add_combat_log(f"Changed art style to {style}", "success")
                st.rerun()
    
    # Show current selection
    current_style = st.session_state.get("art_style", "Dragonflight")
    style_data = WOW_ART_STYLES[current_style]
    st.markdown(
        f"<div class='wow-card' style='text-align: center; background: linear-gradient(145deg, {style_data['color']}, transparent);'>"
        f"<h3>{style_data['icon']} Current: {current_style}</h3>"
        f"<p>{style_data['description']}</p>"
        f"</div>",
        unsafe_allow_html=True
    )

# -----------------------------------------------------------
# Enhanced Sidebar
# -----------------------------------------------------------

def render_enhanced_sidebar(config: Dict[str, Any]):
    """Render WOW-themed sidebar with controls"""
    st.sidebar.markdown(f"# {get_translation('title')}")
    st.sidebar.markdown(f"*{get_translation('subtitle')}*")
    
    st.sidebar.markdown("---")
    
    # Theme and Language Selection
    col1, col2 = st.sidebar.columns(2)
    with col1:
        theme = st.selectbox(
            get_translation("theme"),
            ["light", "dark"],
            index=1 if st.session_state.theme == "dark" else 0,
            key="theme_selector"
        )
        if theme != st.session_state.theme:
            st.session_state.theme = theme
            st.rerun()
    
    with col2:
        lang = st.selectbox(
            get_translation("language"),
            ["en", "zh"],
            index=0 if st.session_state.language == "en" else 1,
            key="lang_selector"
        )
        if lang != st.session_state.language:
            st.session_state.language = lang
            st.rerun()
    
    st.sidebar.markdown("---")
    
    # Player Stats
    st.sidebar.markdown("### 🎮 Hero Stats")
    render_status_indicators()
    
    st.sidebar.markdown("---")
    
    # API Keys
    st.sidebar.markdown(f"### 🔑 {get_translation('api_keys')}")
    
    with st.sidebar.expander("Configure API Keys"):
        get_api_key_from_env_or_ui(
            "OpenAI", "OPENAI_API_KEY", "openai_api_key", "OpenAI API Key"
        )
        get_api_key_from_env_or_ui(
            "Gemini", "GEMINI_API_KEY", "gemini_api_key", "Gemini API Key"
        )
        get_api_key_from_env_or_ui(
            "xAI", "XAI_API_KEY", "xai_api_key", "xAI (Grok) API Key"
        )
        get_api_key_from_env_or_ui(
            "Anthropic", "ANTHROPIC_API_KEY", "anthropic_api_key", "Anthropic API Key"
        )
    
    st.sidebar.markdown("---")
    
    # Model Settings
    st.sidebar.markdown("### ⚙️ Spell Settings")
    
    provider = st.sidebar.selectbox(
        "Magic School",
        ["openai", "gemini", "xai", "anthropic"],
        key="default_provider",
    )
    
    provider_models = {
        "openai": ["gpt-4o-mini", "gpt-4.1-mini"],
        "gemini": ["gemini-2.5-flash", "gemini-2.5-flash-lite"],
        "xai": ["grok-4-fast-reasoning", "grok-3-mini"],
        "anthropic": ["claude-3-5-sonnet-latest", "claude-3-opus-latest"],
    }
    
    st.sidebar.selectbox(
        "Spell Rank",
        provider_models[provider],
        key="default_model",
    )
    
    st.sidebar.slider(
        "Spell Power",
        64, 4096, 1024, 64,
        key="default_max_tokens",
    )
    
    st.sidebar.slider(
        "Chaos Level",
        0.0, 1.0, 0.7, 0.05,
        key="default_temperature",
    )
    
    st.sidebar.markdown("---")
    
    # Quest Log
    st.sidebar.markdown(f"### 📜 {get_translation('quest_log')}")
    st.sidebar.metric("Quests Completed", st.session_state.quests_completed)
    
    # Regenerate mana button
    if st.sidebar.button("🔮 Regenerate Resources"):
        update_player_stats("regenerate")
        add_combat_log("Resources regenerated", "success")
        st.rerun()

# -----------------------------------------------------------
# Enhanced Tab Renders
# -----------------------------------------------------------

def render_input_tab():
    """Render input tab with WOW styling"""
    st.markdown(f"## 📝 {get_translation('input')}")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.text_area(
            "📋 Quest Template",
            key="template",
            height=250,
            help="Enter your quest template here"
        )
        
        st.text_area(
            "👁️ Observations",
            key="observations",
            height=250,
            help="Add your battlefield observations"
        )
    
    with col2:
        render_combat_log()
        
        st.markdown("### 🎯 Quick Actions")
        if st.button("💾 Save to Inventory", use_container_width=True):
            add_combat_log("Saved to inventory", "success")
            st.success("✅ Saved!")
        
        if st.button("🗑️ Clear Fields", use_container_width=True):
            st.session_state.template = ""
            st.session_state.observations = ""
            add_combat_log("Fields cleared", "info")
            st.rerun()

def render_pipeline_tab(config: Dict[str, Any]):
    """Render pipeline tab with WOW styling"""
    st.markdown(f"## 🔮 {get_translation('pipeline')}")
    
    if not config or "pipelines" not in config:
        st.warning("⚠️ No spell chains found in agents.yaml")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        pipeline_options = {p["name"]: p for p in config["pipelines"]}
        selected_name = st.selectbox("🎯 Select Spell Chain", list(pipeline_options.keys()))
        pipeline = pipeline_options[selected_name]
        
        st.markdown(f"**Chain ID:** `{pipeline['id']}`")
        st.markdown(f"**Description:** {pipeline.get('description', '')}")
        
        st.markdown("### ⚡ Spell Sequence")
        for idx, step in enumerate(pipeline["steps"], start=1):
            st.markdown(f"- **Step {idx}:** `{step['agent_id']}`")
        
        st.markdown("---")
        
        override_prompt = st.text_area(
            "✨ Additional Instructions",
            "Cast your spell modifications here...",
            height=120,
        )
        
        col_a, col_b = st.columns(2)
        with col_a:
            provider = st.selectbox(
                "Magic School",
                ["(use default)", "openai", "gemini", "xai", "anthropic"],
            )
        with col_b:
            model_override = st.text_input("Spell Variant (optional)", "")
        
        if st.button(f"🔮 {get_translation('run')} Spell Chain", use_container_width=True):
            if st.session_state.mana < 20:
                st.error("❌ Not enough mana!")
                return
            
            template = st.session_state.get("template", "")
            observations = st.session_state.get("observations", "")
            current_input = f"TEMPLATE:\n{template}\n\nOBSERVATIONS:\n{observations}\n\nINSTRUCTIONS:\n{override_prompt}"
            
            outputs = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, step in enumerate(pipeline["steps"]):
                agent_id = step["agent_id"]
                agent_cfg = next((a for a in config["agents"] if a["id"] == agent_id), None)
                
                if not agent_cfg:
                    st.error(f"❌ Agent {agent_id} not found")
                    return
                
                progress = (idx + 1) / len(pipeline["steps"])
                progress_bar.progress(progress)
                status_text.text(f"⚡ Casting: {agent_cfg['name']}...")
                
                try:
                    result = run_agent(
                        agent_cfg=agent_cfg,
                        user_prompt=current_input,
                        override_provider=None if provider.startswith("(") else provider,
                        override_model=model_override or None,
                        max_tokens=st.session_state.get("default_max_tokens", 1024),
                        temperature=st.session_state.get("default_temperature", 0.7),
                    )
                    outputs.append({"agent_id": agent_id, "output": result})
                    current_input = result
                    update_player_stats("regenerate")
                except Exception as e:
                    st.error(f"❌ Spell failed: {e}")
                    add_combat_log(f"Spell chain failed at {agent_id}", "error")
                    return
            
            progress_bar.progress(1.0)
            status_text.text("✅ Spell chain complete!")
            
            st.success("🎉 Spell Chain Completed!")
            update_player_stats("quest_complete")
            add_combat_log(f"Completed spell chain: {selected_name}", "success")
            
            st.session_state.pipeline_history.append(outputs)
            
            st.markdown("### 📜 Chain Results")
            for idx, item in enumerate(outputs, start=1):
                with st.expander(f"⚡ Step {idx} – {item['agent_id']}"):
                    st.markdown(item["output"])
    
    with col2:
        render_combat_log()
        st.markdown("### 📊 Chain Stats")
        st.metric("Total Runs", len(st.session_state.pipeline_history))

def render_dashboard_tab():
    """Render interactive dashboard with WOW styling"""
    st.markdown(f"## 📊 {get_translation('dashboard')}")
    
    # Stats Overview
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🏆 Level", st.session_state.player_level)
    with col2:
        st.metric("✅ Quests", st.session_state.quests_completed)
    with col3:
        st.metric("🔮 Spells Cast", len(st.session_state.combat_log))
    with col4:
        st.metric("📜 Pipelines", len(st.session_state.pipeline_history))
    
    st.markdown("---")
    
    # Tabs for different dashboard views
    dash_tab1, dash_tab2, dash_tab3 = st.tabs(["📜 History", "⚔️ Combat Log", "🏆 Achievements"])
    
    with dash_tab1:
        st.markdown("### 📜 Quest History")
        history = st.session_state.get("pipeline_history", [])
        if not history:
            st.info("🎯 No quests completed yet. Start your adventure!")
        else:
            for run_idx, run in enumerate(reversed(history), start=1):
                with st.expander(f"🗡️ Quest #{len(history) - run_idx + 1}"):
                    for step_idx, item in enumerate(run, start=1):
                        st.markdown(f"**Step {step_idx}** – `{item['agent_id']}`")
                        st.markdown(item["output"][:200] + "...")
    
    with dash_tab2:
        st.markdown("### ⚔️ Full Combat Log")
        if st.session_state.combat_log:
            for entry in reversed(st.session_state.combat_log):
                st.markdown(f"{entry['icon']} {entry['message']}")
        else:
            st.info("No combat actions yet")
    
    with dash_tab3:
        st.markdown("### 🏆 Achievements")
        
        achievements = []
        if st.session_state.player_level >= 5:
            achievements.append("🎖️ Veteran Hero - Reached level 5")
        if st.session_state.quests_completed >= 10:
            achievements.append("📜 Quest Master - Completed 10 quests")
        if len(st.session_state.combat_log) >= 50:
            achievements.append("⚔️ Battle Tested - Performed 50 actions")
        if st.session_state.player_level >= 10:
            achievements.append("👑 Champion - Reached level 10")
        
        if achievements:
            for ach in achievements:
                st.success(ach)
        else:
            st.info("🎯 Complete quests to unlock achievements!")

# -----------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------

def main():
    """Main application entry point"""
    st.set_page_config(
        page_title="WOW Multi-Agent Studio",
        page_icon="⚔️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    init_session_state()
    
    # Apply custom CSS
    apply_custom_css()
    
    # Load configuration
    config = load_agents_config()
    
    # Render sidebar
    render_enhanced_sidebar(config)
    
    # Main content area
    st.markdown(f"# ⚔️ {get_translation('title')}")
    
    # Art Style Selection
    render_magic_wheel()
    
    st.markdown("---")
    
    # Main tabs
    tab_input, tab_pipeline, tab_smart, tab_notes, tab_dashboard = st.tabs([
        f"📝 {get_translation('input')}",
        f"🔮 {get_translation('pipeline')}",
        f"✨ {get_translation('smart_replace')}",
        f"📔 {get_translation('notes')}",
        f"📊 {get_translation('dashboard')}"
    ])
    
    with tab_input:
        render_input_tab()
    
    with tab_pipeline:
        render_pipeline_tab(config)
    
    with tab_smart:
        st.markdown("## ✨ Smart Replace (Magic Editor)")
        st.info("🔮 Cast transformation spells on your text!")
        # Previous smart replace code can go here
    
    with tab_notes:
        st.markdown("## 📔 AI Note Keeper")
        st.info("📜 Manage your adventure journal!")
        # Previous note keeper code can go here
    
    with tab_dashboard:
        render_dashboard_tab()

if __name__ == "__main__":
    main()
