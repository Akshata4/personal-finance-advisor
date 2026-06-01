import streamlit as st
from pydantic_ai.messages import ModelRequest, ModelResponse, ToolCallPart, ToolReturnPart

from finance_agent import (
    clear_db,
    db_exists,
    finance_agent,
    get_transaction_count,
    load_csv_to_db,
)

st.set_page_config(page_title="Finance Auditor", page_icon="💰", layout="wide")

# ── Session state ─────────────────────────────────────────────────────────────
if "display_messages" not in st.session_state:
    st.session_state.display_messages = []  # {"role": ..., "content": ...}
if "agent_messages" not in st.session_state:
    st.session_state.agent_messages = []    # PydanticAI ModelMessage history


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("💰 Finance Auditor")
    st.divider()

    # DB status
    if db_exists():
        count = get_transaction_count()
        st.success(f"✓ {count} transactions loaded")
        if st.button("Clear database"):
            clear_db()
            st.session_state.display_messages = []
            st.session_state.agent_messages = []
            st.rerun()
    else:
        st.info("No data loaded yet")

    st.subheader("Upload statement")
    uploaded = st.file_uploader("Bank CSV file", type=["csv"])

    if uploaded:
        if st.button("Load CSV", type="primary"):
            with st.spinner("Normalising transactions and spawning categorisation agents..."):
                try:
                    csv_text = uploaded.read().decode("utf-8")
                    count, ambiguous = load_csv_to_db(csv_text)
                    st.success(f"Loaded {count} transactions")
                    if ambiguous:
                        with st.expander(
                            f"⚠️ {len(ambiguous)} ambiguous categories — click to review"
                        ):
                            st.caption(
                                "These were assigned to the best-guess category. "
                                "You can ask the agent to re-examine any of them."
                            )
                            for desc in ambiguous:
                                st.write(f"• {desc}")
                    # reset conversation when new data is loaded
                    st.session_state.display_messages = []
                    st.session_state.agent_messages = []
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to load CSV: {e}")

    st.divider()
    st.toggle("Show agent reasoning", key="show_reasoning", value=False)
    st.divider()
    st.caption("Try asking:")
    st.caption("• How much did I spend last month?")
    st.caption("• Compare food spending in April vs May")
    st.caption("• What subscriptions am I paying?")
    st.caption("• Show my top 5 merchants this month")
    st.caption("• Which day did I spend the most?")


# ── Reasoning trace helper ────────────────────────────────────────────────────

def render_reasoning(result) -> None:
    """Render tool calls and results from the current run as a Streamlit expander."""
    steps = []
    for msg in result.all_messages():
        if isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, ToolCallPart):
                    steps.append({"type": "call", "tool": part.tool_name, "args": part.args})
        elif isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, ToolReturnPart):
                    steps.append({"type": "result", "content": str(part.content)})

    if not steps:
        return

    with st.expander("Agent reasoning — see how it answered", expanded=False):
        for i, step in enumerate(steps):
            if step["type"] == "call":
                st.markdown(f"**Step {i + 1} — Tool call: `{step['tool']}`**")
                st.code(str(step["args"]), language="json")
            else:
                st.markdown(f"**Step {i + 1} — Tool result**")
                st.code(step["content"], language="text")


# ── Main chat area ────────────────────────────────────────────────────────────
st.title("Ask about your finances")

if not db_exists():
    st.info("👈 Upload a bank statement CSV from the sidebar to get started.")
    st.stop()

# Render conversation history
for msg in st.session_state.display_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# New user input
if prompt := st.chat_input("e.g. How much did I spend on food last month?"):
    # Show user message immediately
    st.session_state.display_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run agent and stream response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = finance_agent.run_sync(
                prompt,
                message_history=st.session_state.agent_messages,
            )
        response = str(result.output)
        st.session_state.agent_messages = list(result.all_messages())
        st.session_state.display_messages.append(
            {"role": "assistant", "content": response}
        )
        st.markdown(response)
        if st.session_state.get("show_reasoning"):
            render_reasoning(result)
