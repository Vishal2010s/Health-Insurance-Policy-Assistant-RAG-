"""Streamlit chatbot for the Health Insurance RAG assistant.

Keep this file beside ``master.py``. The knowledge base should be stored at
``data/knowledge_base.json``. Start the app with: ``streamlit run app.py``.
"""

import json
import os
from pathlib import Path
from statistics import mean

import streamlit as st
from dotenv import load_dotenv

from master_HF import HealthInsuranceRAG


# ──────────────────────────────────────────────
# PAGE AND FILE CONFIGURATION
# ──────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_KB_PATH = BASE_DIR / "data" / "knowledge_base.json"

st.set_page_config(
    page_title="Health Insurance Policy Assistant",
    page_icon="🏥",
    layout="centered",
)


# ──────────────────────────────────────────────
# RAG PIPELINE INITIALIZATION
# ──────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_rag() -> HealthInsuranceRAG:
    """Create the RAG pipeline once and reuse it across Streamlit reruns."""
    # Load tracing settings before master.py initializes LangSmith. Support
    # both the current flag and the older LangChain v2 tracing flag.
    load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)
    current_tracing = (
        os.getenv("LANGSMITH_TRACING", "false").strip().lower() == "true"
    )
    legacy_tracing = (
        os.getenv("LANGCHAIN_TRACING_V2", "false").strip().lower() == "true"
    )
    if current_tracing or legacy_tracing:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGCHAIN_TRACING_V2"] = "true"

    kb_path = Path(os.getenv("SOURCE_PATH", str(DEFAULT_KB_PATH)))
    if not kb_path.is_file():
        raise FileNotFoundError(
            f"Knowledge base not found: {kb_path}. "
            "Place it at data/knowledge_base.json or set SOURCE_PATH."
        )

    rag = HealthInsuranceRAG(kb_path=str(kb_path))
    rag.setup()
    return rag


# ──────────────────────────────────────────────
# SESSION AND STATUS HELPERS
# ──────────────────────────────────────────────

def initialize_session() -> None:
    """Create the conversation list on the first Streamlit run."""
    if "messages" not in st.session_state:
        st.session_state.messages = []


def langsmith_is_enabled() -> bool:
    """Return True when either supported tracing flag and its key are set."""
    tracing_enabled = any(
        os.getenv(flag, "false").strip().lower() == "true"
        for flag in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2")
    )
    return tracing_enabled and bool(os.getenv("LANGSMITH_API_KEY"))


def assistant_messages() -> list:
    """Return only completed assistant messages from chat history."""
    return [
        message
        for message in st.session_state.messages
        if message["role"] == "assistant"
    ]


def transcript_json() -> str:
    """Create a readable JSON transcript for download."""
    return json.dumps(
        st.session_state.messages,
        indent=2,
        ensure_ascii=False,
    )


# ──────────────────────────────────────────────
# RESPONSE AND SOURCE DISPLAY
# ──────────────────────────────────────────────

def unique_policy_clauses(sources: list) -> list:
    """Return one display entry per retrieved policy clause."""
    unique_sources = []
    seen_clauses = set()
    for source in sources:
        doc_id = source.get("doc_id", "UNKNOWN")
        title = source.get("title", "Untitled clause")
        category = source.get("category", "general")
        clause_key = (doc_id, title, category)
        if clause_key not in seen_clauses:
            seen_clauses.add(clause_key)
            unique_sources.append((doc_id, title, category))
    return unique_sources


def show_sources(sources: list) -> None:
    """Display a concise, de-duplicated list of retrieved policy clauses."""
    unique_sources = unique_policy_clauses(sources)
    if not unique_sources:
        st.info("No source clauses were returned for this response.")
        return

    for number, (doc_id, title, category) in enumerate(unique_sources, start=1):
        st.markdown(
            f"{number}. **{doc_id} — {title}** — Category: {category}"
        )


def save_rating(message_index: int, rating: int) -> None:
    """Store a real user rating beside the relevant assistant message."""
    st.session_state.messages[message_index]["rating"] = rating


def show_response(result: dict, message_index: int) -> None:
    """Display one answer, its metadata, sources, and feedback control."""
    answer = str(result.get("answer", "No answer was returned."))
    if answer.startswith("Error:"):
        st.error(answer)
    else:
        st.markdown(answer)

    total_latency = float(result.get("total_latency", 0.0))
    source_count = len(unique_policy_clauses(result.get("sources", [])))
    model_name = result.get("model_name", "Gemini")
    st.caption(
        f"Response time: {total_latency:.2f}s · "
        f"Sources: {source_count} · Model: {model_name}"
    )

    with st.expander(
        f"View retrieved policy clauses ({source_count})",
        expanded=False,
    ):
        show_sources(result.get("sources", []))

    existing_rating = st.session_state.messages[message_index].get("rating")
    if existing_rating:
        st.caption(f"Your rating: {existing_rating}/5")
        return

    st.markdown("**🙏 Kindly provide your feedback.**")
    st.caption("Please rate this answer using the stars below.")
    feedback = st.feedback(
        "stars",
        key=f"response_rating_{message_index}",
    )
    if feedback is not None:
        # Streamlit returns 0–4 for star feedback; project reporting uses 1–5.
        save_rating(message_index, feedback + 1)
        st.rerun()


# ──────────────────────────────────────────────
# SIDEBAR AND PROJECT METRICS
# ──────────────────────────────────────────────

def show_sidebar(rag: HealthInsuranceRAG) -> None:
    """Display project information and current-session statistics."""
    with st.sidebar:
        st.subheader("System status")
        st.success("RAG pipeline ready")
        st.caption(f"Model: {rag.model_name}")
        st.caption(f"Embedding model:{rag.embedding_model_name}")
        messages = assistant_messages()
        successful = [
            message
            for message in messages
            if not str(message["result"].get("answer", "")).startswith("Error:")
        ]
        latencies = [
            float(message["result"].get("total_latency", 0.0))
            for message in successful
        ]
        ratings = [
            message["rating"]
            for message in messages
            if message.get("rating") is not None
        ]

        st.subheader("Current session")
        st.metric("Questions answered", len(successful))
        st.metric(
            "Average response time",
            f"{mean(latencies):.2f}s" if latencies else "—",
        )
        st.metric(
            "Average user rating",
            f"{mean(ratings):.1f}/5" if ratings else "Not rated",
        )

        st.download_button(
            "Download conversation",
            data=transcript_json(),
            file_name="insurance_chat_transcript.json",
            mime="application/json",
            use_container_width=True,
            disabled=not st.session_state.messages,
        )

        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


# ──────────────────────────────────────────────
# CHAT HISTORY AND EXAMPLE QUESTIONS
# ──────────────────────────────────────────────

def show_examples() -> str | None:
    """Show example questions inside a compact popover."""
    examples = [
        "Are pre and post hospitalization expenses covered?",
        "What is the waiting period for pre-existing diseases?",
        "How do I file a cashless claim?",
        "Is ambulance charge covered?",
        "Can I port the policy to another insurer?",
        "Which treatments are permanently excluded?",
        "How is the premium calculated?",
        "How can I find a network hospital?"
    ]

    selected_question = None
    with st.popover("💡 Try any of the Frequently asked query"):
        st.caption("Select a sample policy question:")
        for number, example in enumerate(examples):
            if st.button(
                example,
                key=f"example_{number}",
                use_container_width=True,
            ):
                selected_question = example
    return selected_question


def show_chat_history() -> None:
    """Render every user and assistant message in the current session."""
    for index, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                show_response(message["result"], index)


# ──────────────────────────────────────────────
# QUESTION PROCESSING
# ──────────────────────────────────────────────

def ask_question(rag: HealthInsuranceRAG, question: str) -> None:
    """Run one question through master.py and save the complete response."""
    clean_question = question.strip()
    if not clean_question:
        st.warning("Please enter a policy question.")
        return

    st.session_state.messages.append(
        {"role": "user", "content": clean_question}
    )
    with st.chat_message("user"):
        st.markdown(clean_question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving policy clauses and preparing the answer..."):
            result = rag.query_with_sources(clean_question)

        # Add UI-only metadata without changing master.py's result contract.
        result["model_name"] = rag.model_name
        st.session_state.messages.append(
            {"role": "assistant", "result": result}
        )
        message_index = len(st.session_state.messages) - 1
        show_response(result, message_index)


# ──────────────────────────────────────────────
# MAIN STREAMLIT APPLICATION
# ──────────────────────────────────────────────

def main() -> None:
    """Run the Streamlit chatbot."""
    st.title("🏥 Health Insurance Policy Assistant")
    st.markdown(
        """
        <style>
        @keyframes greetingFadeIn {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes greetingWave {
            0%, 60%, 100% { transform: rotate(0deg); }
            10% { transform: rotate(14deg); }
            20% { transform: rotate(-8deg); }
            30% { transform: rotate(14deg); }
            40% { transform: rotate(-4deg); }
            50% { transform: rotate(10deg); }
        }

        .welcome-message {
            font-size: 1.55rem;
            font-weight: 650;
            line-height: 1.35;
            margin: 0.35rem 0 1.4rem 0;
            animation: greetingFadeIn 0.8s ease-out;
        }

        .welcome-wave {
            display: inline-block;
            transform-origin: 70% 70%;
            animation: greetingWave 2.2s ease-in-out infinite;
        }

        .question-cue {
            font-size: 1.05rem;
            font-weight: 600;
            margin-top: 1rem;
            margin-bottom: 0.4rem;
        }
        </style>

        <div class="welcome-message">
           <br> <span class="welcome-wave">👋</span>
            Hello! How can I help you today?<br><br>
            💬 Raise your questions—we are here to clarify them as per our policy terms.
        </div>
        """,
        unsafe_allow_html=True,
    )

    initialize_session()

    try:
        with st.spinner("Initializing the RAG pipeline..."):
            rag = load_rag()
    except Exception as error:
        st.error(f"The RAG pipeline could not start: {error}")
        st.stop()

    show_sidebar(rag)

    _, example_column = st.columns([3, 1])
    with example_column:
        selected_question = show_examples()

    show_chat_history()

    typed_question = st.chat_input(
        "✍️ Type your question here..."
    )
    question = selected_question or typed_question
    if question:
        ask_question(rag, question)


if __name__ == "__main__":
    main()
