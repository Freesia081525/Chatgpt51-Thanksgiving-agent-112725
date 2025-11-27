import os
import json
from typing import Dict, Any, List, Optional

import streamlit as st
import yaml

# --- LLM client libraries (install via requirements.txt) ---
from openai import OpenAI
import google.generativeai as genai
from anthropic import Anthropic

# -----------------------------------------------------------
# Utility: Load agents.yaml
# -----------------------------------------------------------

@st.cache_data
def load_agents_config(path: str = "agents.yaml") -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# -----------------------------------------------------------
# Utility: manage API keys (env + UI)
# -----------------------------------------------------------

def get_api_key_from_env_or_ui(
    provider_name: str,
    env_var: str,
    session_key: str,
    label: str,
) -> Optional[str]:
    """
    If an environment variable exists, use it and do NOT show the key.
    Otherwise, allow user to input the key (password field) and store it
    only in session_state.
    """
    env_val = os.getenv(env_var)
    if env_val:
        st.caption(f"{label}: loaded from environment variable `{env_var}`.")
        st.session_state[session_key] = env_val
        return env_val

    # Only show input if not in env
    key = st.text_input(
        label,
        value=st.session_state.get(session_key, ""),
        type="password",
    )
    if key:
        st.session_state[session_key] = key
        st.caption(f"{label} stored in session only for this browser session.")
        return key
    return None


# -----------------------------------------------------------
# Utility: LLM call router
# -----------------------------------------------------------

def call_llm(
    provider: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> str:
    """
    Route calls to the selected provider/model.
    This function assumes relevant API keys are already in st.session_state.
    """
    provider = provider.lower().strip()

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
        # xAI currently offers an OpenAI-compatible API surface.
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
        # Anthropic returns content as a list of blocks; use the first text block
        if resp.content and len(resp.content) > 0:
            block = resp.content[0]
            if hasattr(block, "text"):
                return block.text
        return json.dumps(resp.model_dump(), indent=2)

    else:
        raise ValueError(f"Unsupported provider: {provider}")


# -----------------------------------------------------------
# Helpers: run a single agent from agents.yaml
# -----------------------------------------------------------

def run_agent(
    agent_cfg: Dict[str, Any],
    user_prompt: str,
    override_provider: Optional[str] = None,
    override_model: Optional[str] = None,
    override_system_prompt: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> str:
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
# UI Components
# -----------------------------------------------------------

def sidebar_global_controls(config: Dict[str, Any]):
    st.sidebar.title("Global Controls")

    # API keys
    st.sidebar.subheader("API Keys")

    get_api_key_from_env_or_ui(
        provider_name="OpenAI",
        env_var="OPENAI_API_KEY",
        session_key="openai_api_key",
        label="OpenAI API Key",
    )
    get_api_key_from_env_or_ui(
        provider_name="Gemini",
        env_var="GEMINI_API_KEY",
        session_key="gemini_api_key",
        label="Gemini API Key",
    )
    get_api_key_from_env_or_ui(
        provider_name="xAI",
        env_var="XAI_API_KEY",
        session_key="xai_api_key",
        label="xAI (Grok) API Key",
    )
    get_api_key_from_env_or_ui(
        provider_name="Anthropic",
        env_var="ANTHROPIC_API_KEY",
        session_key="anthropic_api_key",
        label="Anthropic API Key",
    )

    st.sidebar.markdown("---")

    # Global model settings
    st.sidebar.subheader("Default Model Settings")

    provider = st.sidebar.selectbox(
        "Default Provider",
        ["openai", "gemini", "xai", "anthropic"],
        key="default_provider",
    )

    # Basic models per provider (can be expanded)
    provider_models = {
        "openai": ["gpt-4o-mini", "gpt-4.1-mini"],
        "gemini": ["gemini-2.5-flash", "gemini-2.5-flash-lite"],
        "xai": ["grok-4-fast-reasoning", "grok-3-mini"],
        "anthropic": ["claude-3-5-sonnet-latest", "claude-3-opus-latest"],
    }

    model = st.sidebar.selectbox(
        "Default Model",
        provider_models[provider],
        key="default_model",
    )

    max_tokens = st.sidebar.slider(
        "Default Max Tokens",
        min_value=64,
        max_value=4096,
        value=1024,
        step=64,
        key="default_max_tokens",
    )

    temperature = st.sidebar.slider(
        "Default Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
        key="default_temperature",
    )

    st.sidebar.markdown("---")

    st.sidebar.subheader("Agents Config Overview")
    if config and "agents" in config:
        for agent in config["agents"]:
            st.sidebar.caption(f"- **{agent['name']}** ({agent['id']})")


def render_input_tab():
    st.header("Input / Template")

    if "template" not in st.session_state:
        st.session_state.template = "## Template\n\nWrite your template here..."
    if "observations" not in st.session_state:
        st.session_state.observations = "Add your observations here..."

    st.text_area(
        "Template",
        key="template",
        height=200,
    )

    st.text_area(
        "Observations",
        key="observations",
        height=200,
    )

    st.info("These inputs will be available to the pipeline and agents.")


def render_pipeline_tab(config: Dict[str, Any]):
    st.header("Pipeline")

    if not config or "pipelines" not in config:
        st.warning("No pipelines found in agents.yaml.")
        return

    pipeline_options = {p["name"]: p for p in config["pipelines"]}
    selected_name = st.selectbox("Select Pipeline", list(pipeline_options.keys()))
    pipeline = pipeline_options[selected_name]

    st.markdown(f"**Pipeline ID:** `{pipeline['id']}`")
    st.markdown(f"**Description:** {pipeline.get('description', '')}")

    st.markdown("### Steps")
    for idx, step in enumerate(pipeline["steps"], start=1):
        st.markdown(f"- Step {idx}: `{step['agent_id']}`")

    st.markdown("---")

    st.subheader("Execution Settings")

    override_prompt = st.text_area(
        "Additional User Prompt (optional)",
        "Use the template and observations to produce an improved version.",
        height=120,
        key="pipeline_override_prompt",
    )

    provider = st.selectbox(
        "Provider (override for this pipeline run)",
        ["(use agent default)", "openai", "gemini", "xai", "anthropic"],
        key="pipeline_provider",
    )
    provider_override = None if provider.startswith("(") else provider

    model_override = st.text_input(
        "Model (override, optional)",
        value="",
        key="pipeline_model_override",
    ) or None

    max_tokens = st.slider(
        "Max Tokens",
        64,
        4096,
        st.session_state.get("default_max_tokens", 1024),
        step=64,
        key="pipeline_max_tokens",
    )

    temperature = st.slider(
        "Temperature",
        0.0,
        1.0,
        st.session_state.get("default_temperature", 0.7),
        step=0.05,
        key="pipeline_temperature",
    )

    if "pipeline_history" not in st.session_state:
        st.session_state.pipeline_history = []

    if st.button("Run Pipeline"):
        template = st.session_state.get("template", "")
        observations = st.session_state.get("observations", "")
        current_input = f"TEMPLATE:\n{template}\n\nOBSERVATIONS:\n{observations}\n\nUSER INSTRUCTION:\n{override_prompt}"

        outputs = []
        for step in pipeline["steps"]:
            agent_id = step["agent_id"]
            agent_cfg = next(
                (a for a in config["agents"] if a["id"] == agent_id), None
            )
            if not agent_cfg:
                st.error(f"Agent {agent_id} not found.")
                return

            with st.spinner(f"Running agent: {agent_cfg['name']} ({agent_id})"):
                try:
                    result = run_agent(
                        agent_cfg=agent_cfg,
                        user_prompt=current_input,
                        override_provider=provider_override,
                        override_model=model_override,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    outputs.append({"agent_id": agent_id, "output": result})
                    current_input = result  # feed forward
                except Exception as e:
                    st.error(f"Error running agent {agent_id}: {e}")
                    return

        st.success("Pipeline completed.")
        st.session_state.pipeline_history.append(outputs)

        st.markdown("### Pipeline Outputs")
        for idx, item in enumerate(outputs, start=1):
            st.markdown(f"#### Step {idx} – Agent `{item['agent_id']}`")
            st.markdown(item["output"])


def render_smart_replace_tab(config: Dict[str, Any]):
    st.header("Smart Replace")

    text = st.text_area(
        "Input Text",
        "Paste your text here...",
        height=200,
        key="smart_replace_text",
    )
    instructions = st.text_area(
        "Replacement Instructions",
        "Example: Replace all mentions of 'beta' with 'v1.0' and adjust the tone to be more formal.",
        height=120,
        key="smart_replace_instructions",
    )

    # Agent selection (optional)
    agent_ids = [a["id"] for a in config.get("agents", [])]
    selected_agent_id = st.selectbox(
        "Agent (from agents.yaml)",
        ["smart_replacer"] + agent_ids,
        index=0,
        key="smart_replace_agent",
    )

    override_system_prompt = st.text_area(
        "System Prompt (override, optional)",
        "",
        height=100,
        key="smart_replace_system_prompt_override",
    ) or None

    provider = st.selectbox(
        "Provider",
        ["openai", "gemini", "xai", "anthropic"],
        key="smart_replace_provider",
    )

    provider_models = {
        "openai": ["gpt-4o-mini", "gpt-4.1-mini"],
        "gemini": ["gemini-2.5-flash", "gemini-2.5-flash-lite"],
        "xai": ["grok-4-fast-reasoning", "grok-3-mini"],
        "anthropic": ["claude-3-5-sonnet-latest", "claude-3-opus-latest"],
    }

    model = st.selectbox(
        "Model",
        provider_models[provider],
        key="smart_replace_model",
    )

    max_tokens = st.slider(
        "Max Tokens",
        64,
        4096,
        st.session_state.get("default_max_tokens", 1024),
        step=64,
        key="smart_replace_max_tokens",
    )

    temperature = st.slider(
        "Temperature",
        0.0,
        1.0,
        st.session_state.get("default_temperature", 0.7),
        step=0.05,
        key="smart_replace_temperature",
    )

    if st.button("Run Smart Replace"):
        if selected_agent_id == "smart_replacer":
            # Use default agent config if exists
            agent_cfg = next(
                (a for a in config.get("agents", []) if a["id"] == "smart_replacer"),
                {
                    "id": "smart_replacer",
                    "name": "Smart Replacer (inline)",
                    "provider": provider,
                    "default_model": model,
                    "system_prompt": (
                        "Perform smart replacements in the given text based on the user's instructions. "
                        "Preserve meaning and structure where possible."
                    ),
                },
            )
        else:
            agent_cfg = next(
                (a for a in config.get("agents", []) if a["id"] == selected_agent_id),
                None,
            )
            if not agent_cfg:
                st.error(f"Agent {selected_agent_id} not found.")
                return

        user_prompt = f"TEXT:\n{text}\n\nINSTRUCTIONS:\n{instructions}"
        try:
            result = run_agent(
                agent_cfg=agent_cfg,
                user_prompt=user_prompt,
                override_provider=provider,
                override_model=model,
                override_system_prompt=override_system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            st.markdown("### Updated Text")
            st.markdown(result)
        except Exception as e:
            st.error(f"Error running smart replace: {e}")


def render_note_keeper_tab(config: Dict[str, Any]):
    st.header("AI Note Keeper")

    if "note_raw_text" not in st.session_state:
        st.session_state.note_raw_text = ""

    st.subheader("Note Input")
    st.session_state.note_raw_text = st.text_area(
        "Paste your note text here",
        st.session_state.note_raw_text,
        height=200,
        key="note_input_text",
    )

    st.markdown("---")

    # Tabs for Note Keeper features
    tab_markdown, tab_format, tab_keywords, tab_entities, tab_chat, tab_mindmap = st.tabs(
        [
            "Text → Markdown",
            "AI Formatting",
            "AI Keywords",
            "AI Entities (20)",
            "AI Chat",
            "AI Mindmap",
        ]
    )

    # ---------- Text -> Markdown ----------
    with tab_markdown:
        st.subheader("Text → Markdown")

        default_prompt = (
            "Convert the following raw text into clean, well-structured markdown. "
            "Use headings, subheadings, bullet lists, and code blocks where appropriate. "
            "Do not omit content."
        )
        system_prompt = st.text_area(
            "System Prompt",
            default_prompt,
            height=120,
            key="note_markdown_system_prompt",
        )

        provider = st.selectbox(
            "Provider",
            ["openai", "gemini", "xai", "anthropic"],
            key="note_markdown_provider",
        )
        provider_models = {
            "openai": ["gpt-4o-mini", "gpt-4.1-mini"],
            "gemini": ["gemini-2.5-flash", "gemini-2.5-flash-lite"],
            "xai": ["grok-4-fast-reasoning", "grok-3-mini"],
            "anthropic": ["claude-3-5-sonnet-latest", "claude-3-opus-latest"],
        }
        model = st.selectbox(
            "Model",
            provider_models[provider],
            key="note_markdown_model",
        )

        max_tokens = st.slider(
            "Max Tokens",
            64,
            4096,
            st.session_state.get("default_max_tokens", 1024),
            step=64,
            key="note_markdown_max_tokens",
        )

        if st.button("Convert to Markdown"):
            try:
                md = call_llm(
                    provider=provider,
                    model=model,
                    system_prompt=system_prompt,
                    user_prompt=st.session_state.note_raw_text,
                    max_tokens=max_tokens,
                    temperature=0.3,
                )
                st.markdown("### Markdown Output")
                st.markdown(md)
                st.session_state.note_markdown = md
            except Exception as e:
                st.error(f"Error converting to markdown: {e}")

    # ---------- AI Formatting ----------
    with tab_format:
        st.subheader("AI Formatting (Reorganize Article)")

        base_text = st.text_area(
            "Source Text (defaults to last markdown if available)",
            st.session_state.get("note_markdown", st.session_state.note_raw_text),
            height=200,
            key="note_format_input_text",
        )

        default_format_prompt = (
            "Reorganize and format the article into a clear, well-structured markdown document. "
            "Preserve ALL information, but improve headings, ordering, and readability. "
            "Do not remove sections; you may add headings, bullet lists, and tables."
        )
        system_prompt = st.text_area(
            "System Prompt",
            default_format_prompt,
            height=120,
            key="note_format_system_prompt",
        )

        provider = st.selectbox(
            "Provider",
            ["openai", "gemini", "xai", "anthropic"],
            key="note_format_provider",
        )
        provider_models = {
            "openai": ["gpt-4o-mini", "gpt-4.1-mini"],
            "gemini": ["gemini-2.5-flash", "gemini-2.5-flash-lite"],
            "xai": ["grok-4-fast-reasoning", "grok-3-mini"],
            "anthropic": ["claude-3-5-sonnet-latest", "claude-3-opus-latest"],
        }
        model = st.selectbox(
            "Model",
            provider_models[provider],
            key="note_format_model",
        )

        max_tokens = st.slider(
            "Max Tokens",
            64,
            4096,
            st.session_state.get("default_max_tokens", 1024),
            step=64,
            key="note_format_max_tokens",
        )

        if st.button("AI Format Article"):
            try:
                formatted = call_llm(
                    provider=provider,
                    model=model,
                    system_prompt=system_prompt,
                    user_prompt=base_text,
                    max_tokens=max_tokens,
                    temperature=0.4,
                )
                st.markdown("### Formatted Markdown")
                st.markdown(formatted)
                st.session_state.note_formatted = formatted
            except Exception as e:
                st.error(f"Error formatting article: {e}")

    # ---------- AI Keywords ----------
    with tab_keywords:
        st.subheader("AI Keywords Highlighting")

        base_md = st.text_area(
            "Markdown to highlight",
            st.session_state.get("note_formatted", st.session_state.get("note_markdown", st.session_state.note_raw_text)),
            height=250,
            key="note_keywords_text",
        )

        keywords_input = st.text_input(
            "Keywords (comma-separated)",
            "Streamlit, agents, API, pipeline, model",
            key="note_keywords_input",
        )
        color = st.color_picker("Keyword Color", "#FF7F50", key="note_keywords_color")  # coral default

        if st.button("Highlight Keywords"):
            keywords = [k.strip() for k in keywords_input.split(",") if k.strip()]
            highlighted = base_md
            # Simple replacement (case-sensitive); in real use you may want regex with word boundaries
            for kw in keywords:
                if kw:
                    highlighted = highlighted.replace(
                        kw,
                        f"<span style='color:{color}; font-weight:bold;'>{kw}</span>",
                    )

            st.markdown("### Highlighted Markdown")
            st.markdown(highlighted, unsafe_allow_html=True)

    # ---------- AI Entities ----------
    with tab_entities:
        st.subheader("AI Entities (20)")

        base_text = st.text_area(
            "Text for entity extraction",
            st.session_state.get("note_formatted", st.session_state.note_raw_text),
            height=220,
            key="note_entities_text",
        )

        default_instr = (
            "Extract exactly 20 key entities from the note. For each, provide:\n"
            "- name\n- type\n- short_description\n- related_components (comma-separated)\n"
            "Return ONLY valid JSON as a list of 20 objects."
        )
        user_prompt = st.text_area(
            "Entity Extraction Instructions",
            default_instr,
            height=140,
            key="note_entities_prompt",
        )

        provider = st.selectbox(
            "Provider",
            ["openai", "gemini", "xai", "anthropic"],
            key="note_entities_provider",
        )
        provider_models = {
            "openai": ["gpt-4o-mini", "gpt-4.1-mini"],
            "gemini": ["gemini-2.5-flash", "gemini-2.5-flash-lite"],
            "xai": ["grok-4-fast-reasoning", "grok-3-mini"],
            "anthropic": ["claude-3-5-sonnet-latest", "claude-3-opus-latest"],
        }
        model = st.selectbox(
            "Model",
            provider_models[provider],
            key="note_entities_model",
        )

        max_tokens = st.slider(
            "Max Tokens",
            256,
            4096,
            2048,
            step=64,
            key="note_entities_max_tokens",
        )

        if st.button("Generate 20 Entities"):
            try:
                raw = call_llm(
                    provider=provider,
                    model=model,
                    system_prompt="You are an information extraction system.",
                    user_prompt=user_prompt + "\n\nNOTE:\n" + base_text,
                    max_tokens=max_tokens,
                    temperature=0.2,
                )
                # Try to parse JSON
                try:
                    entities = json.loads(raw)
                    st.session_state.note_entities = entities
                except json.JSONDecodeError:
                    st.warning("Model did not return valid JSON. Showing raw response.")
                    st.text(raw)
                    return

                st.markdown("### Entities Table (markdown)")
                if isinstance(entities, list):
                    # Build markdown table
                    header = "| # | Name | Type | Description | Related Components |\n"
                    header += "|---|------|------|-------------|--------------------|\n"
                    rows = []
                    for i, ent in enumerate(entities, start=1):
                        rows.append(
                            f"| {i} | {ent.get('name','')} | {ent.get('type','')} | "
                            f"{ent.get('short_description','')} | {ent.get('related_components','')} |"
                        )
                    st.markdown(header + "\n".join(rows))

                    st.markdown("### Entities JSON")
                    st.json(entities)
                else:
                    st.text(raw)
            except Exception as e:
                st.error(f"Error generating entities: {e}")

    # ---------- AI Chat ----------
    with tab_chat:
        st.subheader("AI Chat (Note-Aware)")

        context_text = st.text_area(
            "Context Note",
            st.session_state.get("note_formatted", st.session_state.get("note_markdown", st.session_state.note_raw_text)),
            height=220,
            key="note_chat_context",
        )

        chat_system_prompt = st.text_area(
            "System Prompt",
            "You are an assistant that answers questions based ONLY on the provided note. "
            "If something is not contained in the note, say you don't know.",
            height=120,
            key="note_chat_system_prompt",
        )

        provider = st.selectbox(
            "Provider",
            ["openai", "gemini", "xai", "anthropic"],
            key="note_chat_provider",
        )
        provider_models = {
            "openai": ["gpt-4o-mini", "gpt-4.1-mini"],
            "gemini": ["gemini-2.5-flash", "gemini-2.5-flash-lite"],
            "xai": ["grok-4-fast-reasoning", "grok-3-mini"],
            "anthropic": ["claude-3-5-sonnet-latest", "claude-3-opus-latest"],
        }
        model = st.selectbox(
            "Model",
            provider_models[provider],
            key="note_chat_model",
        )

        max_tokens = st.slider(
            "Max Tokens",
            64,
            4096,
            1024,
            step=64,
            key="note_chat_max_tokens",
        )

        if "note_chat_history" not in st.session_state:
            st.session_state.note_chat_history = []

        user_message = st.text_input("Your message", key="note_chat_user_message")

        if st.button("Send"):
            if not user_message.strip():
                st.warning("Please enter a message.")
            else:
                full_prompt = (
                    "NOTE (context):\n"
                    + context_text
                    + "\n\nUSER MESSAGE:\n"
                    + user_message
                )
                try:
                    reply = call_llm(
                        provider=provider,
                        model=model,
                        system_prompt=chat_system_prompt,
                        user_prompt=full_prompt,
                        max_tokens=max_tokens,
                        temperature=0.3,
                    )
                    st.session_state.note_chat_history.append(
                        {"role": "user", "content": user_message}
                    )
                    st.session_state.note_chat_history.append(
                        {"role": "assistant", "content": reply}
                    )
                except Exception as e:
                    st.error(f"Error during chat: {e}")

        st.markdown("### Chat History")
        for msg in st.session_state.note_chat_history:
            if msg["role"] == "user":
                st.markdown(f"**You:** {msg['content']}")
            else:
                st.markdown(f"**Assistant:** {msg['content']}")

    # ---------- AI Mindmap ----------
    with tab_mindmap:
        st.subheader("AI Mindmap (JSON for networkx)")

        base_text = st.text_area(
            "Text to map",
            st.session_state.get("note_formatted", st.session_state.note_raw_text),
            height=220,
            key="note_mindmap_text",
        )

        default_mindmap_prompt = (
            "Analyze the text and create a concept mindmap. "
            "Return JSON with this structure:\n\n"
            "{\n"
            '  "nodes": [ {"id": "id1", "label": "Node Label"}, ... ],\n'
            '  "edges": [ {"source": "id1", "target": "id2", "relation": "rel"}, ... ]\n'
            "}\n\n"
            "Use concise, meaningful node IDs. Do not include any explanation text."
        )
        system_prompt = st.text_area(
            "System Prompt",
            default_mindmap_prompt,
            height=180,
            key="note_mindmap_system_prompt",
        )

        provider = st.selectbox(
            "Provider",
            ["openai", "gemini", "xai", "anthropic"],
            key="note_mindmap_provider",
        )
        provider_models = {
            "openai": ["gpt-4o-mini", "gpt-4.1-mini"],
            "gemini": ["gemini-2.5-flash", "gemini-2.5-flash-lite"],
            "xai": ["grok-4-fast-reasoning", "grok-3-mini"],
            "anthropic": ["claude-3-5-sonnet-latest", "claude-3-opus-latest"],
        }
        model = st.selectbox(
            "Model",
            provider_models[provider],
            key="note_mindmap_model",
        )

        max_tokens = st.slider(
            "Max Tokens",
            256,
            4096,
            2048,
            step=64,
            key="note_mindmap_max_tokens",
        )

        if st.button("Generate Mindmap JSON"):
            try:
                raw = call_llm(
                    provider=provider,
                    model=model,
                    system_prompt=system_prompt,
                    user_prompt=base_text,
                    max_tokens=max_tokens,
                    temperature=0.2,
                )
                try:
                    graph = json.loads(raw)
                    st.session_state.note_mindmap = graph
                    st.markdown("### Mindmap JSON")
                    st.json(graph)
                    st.info(
                        "You can export this JSON and load it into Python using networkx, "
                        "e.g., by adding nodes and edges from this structure."
                    )
                except json.JSONDecodeError:
                    st.warning("Model did not return valid JSON. Showing raw response.")
                    st.text(raw)
            except Exception as e:
                st.error(f"Error generating mindmap: {e}")


def render_dashboard_tab():
    st.header("Dashboard / History")

    st.subheader("Pipeline Runs")
    history = st.session_state.get("pipeline_history", [])
    if not history:
        st.info("No pipeline runs yet.")
    else:
        for run_idx, run in enumerate(history, start=1):
            st.markdown(f"### Run {run_idx}")
            for step_idx, item in enumerate(run, start=1):
                st.markdown(f"#### Step {step_idx} – Agent `{item['agent_id']}`")
                with st.expander("Show output"):
                    st.markdown(item["output"])

    st.markdown("---")

    st.subheader("Note Entities & Mindmap (if generated)")
    if "note_entities" in st.session_state:
        st.markdown("#### Latest Entities")
        st.json(st.session_state.note_entities)

    if "note_mindmap" in st.session_state:
        st.markdown("#### Latest Mindmap JSON")
        st.json(st.session_state.note_mindmap)


# -----------------------------------------------------------
# Main entry point
# -----------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Multi-Agent Workflow Studio (Streamlit)",
        layout="wide",
    )

    config = load_agents_config()

    # Sidebar
    sidebar_global_controls(config)

    # Main tabs
    tab_input, tab_pipeline, tab_smart, tab_notes, tab_dashboard = st.tabs(
        ["Input", "Pipeline", "Smart Replace", "AI Note Keeper", "Dashboard"]
    )

    with tab_input:
        render_input_tab()

    with tab_pipeline:
        render_pipeline_tab(config)

    with tab_smart:
        render_smart_replace_tab(config)

    with tab_notes:
        render_note_keeper_tab(config)

    with tab_dashboard:
        render_dashboard_tab()


if __name__ == "__main__":
    main()
