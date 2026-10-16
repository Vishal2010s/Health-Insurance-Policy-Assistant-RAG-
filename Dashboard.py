import logging
import warnings

# # FAISS currently remains in the archived langchain-community package.
# # Hide only its known sunset notice; do not hide unrelated warnings.
# warnings.filterwarnings(
#     "ignore",
#     message=r"`langchain-community` is being sunset.*",
#     category=DeprecationWarning,
# )

# # The Google SDK currently prints an AFC migration notice even though this
# # RAG pipeline does not define or call any tools. Hide only that SDK logger.
# logging.getLogger("google_genai.models").setLevel(logging.ERROR)
# logging.getLogger("google.genai.models").setLevel(logging.ERROR)
# 
import json
import os
import time
from pathlib import Path
from statistics import mean

import streamlit as st
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# -----------------------------------------------------------------------------
# PAGE AND FILE CONFIGURATION
# -----------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Health Insurance Policy Assistant",
    page_icon="🏥",
    layout="centered",
)

load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)

index_path = Path(
    os.getenv("FAISS_INDEX_PATH", str(BASE_DIR / "faiss_index"))
)

if not index_path.is_absolute():
    index_path = BASE_DIR / index_path

index_path = index_path.resolve()
config_path = index_path / "pipeline_config.json"

# -----------------------------------------------------------------------------
# READ SAVED PIPELINE CONFIGURATION
# -----------------------------------------------------------------------------

try:
    required_files = [
        index_path / "index.faiss",
        index_path / "index.pkl",
        config_path,
    ]

    missing = [str(path) for path in required_files if not path.is_file()]

    if missing:
        raise FileNotFoundError(
            "Run the master file to save the policy index first. "
            "Missing files: " + ", ".join(missing)
        )

    with config_path.open("r", encoding="utf-8") as file:
        saved_config = json.load(file)

    if not isinstance(saved_config, dict):
        raise ValueError("Pipeline configuration must be a JSON object.")

    for key in (
        "embedding_model_name",
        "normalize_embeddings",
        "model_name",
        "top_k",
        "system_prompt",
    ):
        if key not in saved_config:
            raise ValueError(f"Saved configuration is missing: {key}")

    for key in ("embedding_model_name", "model_name", "system_prompt"):
        if (
            not isinstance(saved_config[key], str)
            or not saved_config[key].strip()
        ):
            raise ValueError(f"{key} must contain non-empty text.")

    if not isinstance(saved_config["normalize_embeddings"], bool):
        raise ValueError("normalize_embeddings must be true or false.")

    if (
        isinstance(saved_config["top_k"], bool)
        or not isinstance(saved_config["top_k"], int)
        or saved_config["top_k"] < 1
    ):
        raise ValueError("top_k must be a positive integer.")

    top_k = saved_config["top_k"]

    # Reload on a Streamlit rerun if saved files have changed.
    pipeline_signature = (
        str(index_path),
        tuple(
            (path.stat().st_mtime_ns, path.stat().st_size)
            for path in required_files
        ),
    )

except Exception as error:
    st.error(f"The saved pipeline could not be loaded: {error}")
    st.stop()

# -----------------------------------------------------------------------------
# INITIALIZE FROM THE MASTER'S SAVED FAISS INDEX
# -----------------------------------------------------------------------------

if (
    "rag_pipeline" not in st.session_state
    or st.session_state.get("pipeline_signature") != pipeline_signature
):
    try:
        with st.spinner("Loading the saved policy index..."):
            api_key = os.getenv("GEMINI_API_KEY")

            if not api_key:
                raise ValueError("GEMINI_API_KEY not found in the environment.")

            tracing_requested = (
                os.getenv("LANGSMITH_TRACING", "false")
                .strip()
                .lower() == "true"
            )

            tracing_enabled = (
                tracing_requested and bool(os.getenv("LANGSMITH_API_KEY"))
            )

            os.environ["LANGSMITH_TRACING"] = (
                "true" if tracing_enabled else "false"
            )
            os.environ["LANGCHAIN_TRACING_V2"] = (
                "true" if tracing_enabled else "false"
            )

            if tracing_enabled:
                os.environ.setdefault("LANGSMITH_PROJECT", "insurance-rag")

            # MiniLM embeds new questions; document vectors are loaded from disk.
            embeddings = HuggingFaceEmbeddings(
                model_name=saved_config["embedding_model_name"],
                model_kwargs={"device": "cpu"},
                encode_kwargs={
                    "normalize_embeddings": (
                        saved_config["normalize_embeddings"]
                    )
                },
            )

            # Load only index files created by your own trusted master script.
            vector_store = FAISS.load_local(
                str(index_path),
                embeddings,
                allow_dangerous_deserialization=True,
            )

            llm = ChatGoogleGenerativeAI(
                model=saved_config["model_name"],
                google_api_key=api_key,
            )

            prompt = PromptTemplate(
                template=saved_config["system_prompt"],
                input_variables=["context", "question"],
            )

            retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": top_k},
            )

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": prompt},
                return_source_documents=True,
            )

            st.session_state.rag_pipeline = {
                "qa_chain": qa_chain,
                "model_name": saved_config["model_name"],
                "embedding_model_name": saved_config["embedding_model_name"],
                "index_path": str(index_path),
                "top_k": top_k,
                "tracing_enabled": tracing_enabled,
            }

            st.session_state.pipeline_signature = pipeline_signature

    except Exception as error:
        st.error(f"The RAG pipeline could not start: {error}")
        st.stop()

pipeline = st.session_state.rag_pipeline


# -----------------------------------------------------------------------------
# SESSION INITIALIZATION
# -----------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_number" not in st.session_state:
    st.session_state.conversation_number = 0

sidebar_content = st.sidebar.container()
chat_tab, comparison_tab = st.tabs([
    "💬 Policy Chatbot",
    "📊 Evaluation Comparison",
])
# -----------------------------------------------------------------------------
# TITLE AND WELCOME MESSAGE
# -----------------------------------------------------------------------------

with chat_tab:
    st.title("🏥 Health Insurance Policy Assistant")

    # Your existing welcome message, examples, input,
    # question processing, chat history and ratings go here.st.title("🏥 Health Insurance Policy Assistant")

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
    </style>
    <div class="welcome-message">
        <span class="welcome-wave">👋</span>
        Hello! How can I help you today?<br><br>
        💬 Ask your questions—we are here to clarify the available policy terms.
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# EXAMPLE QUESTIONS AND USER INPUT
# -----------------------------------------------------------------------------

examples = [
    "Are pre and post hospitalization expenses covered?",
    "What is the waiting period for pre-existing diseases?",
    "How do I file a cashless claim?",
    "Is ambulance charge covered?",
    "Can I port the policy to another insurer?",
    "Which treatments are permanently excluded?",
    "How is the premium calculated?",
    "How can I find a network hospital?",
]

selected_question = None

with st.popover("💡 Try a frequently asked question"):
    st.caption("Select a sample policy question:")

    for number, example in enumerate(examples):
        if st.button(
            example,
            key=f"example_{number}",
            use_container_width=True,
        ):
            selected_question = example

typed_question = st.chat_input("✍️ Type your question here...")
question = selected_question or typed_question

# -----------------------------------------------------------------------------
# QUESTION PROCESSING
# -----------------------------------------------------------------------------

if question:
    clean_question = question.strip()

    if not clean_question:
        st.warning("Please enter a policy question.")
    else:
        st.session_state.messages.append({
            "role": "user",
            "content": clean_question,
        })

        with st.spinner("Retrieving policy clauses and preparing the answer..."):
            total_start = time.perf_counter()

            try:
                response = pipeline["qa_chain"].invoke({
                    "query": clean_question,
                })

                raw_answer = response.get("result", "")
                content = getattr(raw_answer, "content", raw_answer)

                if isinstance(content, str):
                    answer = content
                elif isinstance(content, list):
                    text_parts = []

                    for block in content:
                        if isinstance(block, str):
                            text_parts.append(block)
                        elif (
                            isinstance(block, dict)
                            and block.get("type") == "text"
                        ):
                            text_parts.append(block.get("text", ""))

                    answer = "\n".join(text_parts)
                else:
                    answer = getattr(raw_answer, "text", "")

                if not isinstance(answer, str) or not answer.strip():
                    raise ValueError("The model returned an empty answer.")

                sources = []

                for doc in response.get("source_documents", []):
                    sources.append({
                        "doc_id": doc.metadata.get("doc_id", "UNKNOWN"),
                        "category": doc.metadata.get("category", "general"),
                        "title": doc.metadata.get("title", "Untitled clause"),
                        "source": doc.metadata.get("source", "UNKNOWN"),
                        "chunk_id": doc.metadata.get("chunk_id", "UNKNOWN"),
                        "content": doc.page_content,
                    })

                result = {
                    "question": clean_question,
                    "answer": answer.strip(),
                    "sources": sources,
                    "status": "success",
                    "total_latency": round(
                        time.perf_counter() - total_start, 4
                    ),
                    "model_name": pipeline["model_name"],
                }

            except Exception as error:
                result = {
                    "question": clean_question,
                    "answer": f"Error: {error}",
                    "sources": [],
                    "status": "failed",
                    "total_latency": round(
                        time.perf_counter() - total_start, 4
                    ),
                    "model_name": pipeline["model_name"],
                }

        st.session_state.messages.append({
            "role": "assistant",
            "result": result,
        })

# -----------------------------------------------------------------------------
# CHAT HISTORY, RETRIEVED CLAUSES AND RATINGS
# -----------------------------------------------------------------------------

for index, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
            continue

        result = message["result"]

        if result["status"] == "failed":
            st.error(result["answer"])
        else:
            st.markdown(result["answer"])

        # Deduplicate display only; retain distinct passages from one document.
        unique_sources = []
        seen_sources = set()

        for source in result.get("sources", []):
            source_key = (
                source["doc_id"],
                source["title"],
                source["content"],
            )

            if source_key not in seen_sources:
                seen_sources.add(source_key)
                unique_sources.append(source)

        st.caption(
            f"Response time: {result['total_latency']:.2f}s · "
            f"Retrieved passages: {len(unique_sources)} "
            # f"Model: {result['model_name']}"
        )

        with st.expander(
            f"View retrieved policy clauses ({len(unique_sources)})"
        ):
            if not unique_sources:
                st.info("No source clauses were returned.")
            else:
                for number, source in enumerate(unique_sources, 1):
                    st.markdown(
                        f"**{number}. {source['doc_id']} — {source['title']}**"
                    )
                    st.caption(
                        f"Category: {source['category']} · "
                        f"Source: {source['source']} · "
                        # f"Chunk: {source['chunk_id']}"
                    )
                    st.write(source["content"])

        # Failed requests are excluded from answer ratings.
        if result["status"] != "success":
            continue

        existing_rating = message.get("rating")

        if existing_rating is not None:
            st.caption(f"Your rating: {existing_rating}/5")
        else:
            st.markdown("**🙏 Kindly provide your feedback.**")
            st.caption("Please rate this answer using the stars below.")

            feedback = st.feedback(
                "stars",
                key=(
                    f"response_rating_"
                    f"{st.session_state.conversation_number}_{index}"
                ),
            )

            if feedback is not None:
                st.session_state.messages[index]["rating"] = feedback + 1
                st.rerun()

# -----------------------------------------------------------------------------
# TAB 2: SAVED EMBEDDING COMPARISON
# -----------------------------------------------------------------------------

with comparison_tab:
    st.subheader("Google vs MiniLM Embeddings")

    st.caption(
        "Saved results from two earlier 20-question evaluation runs. "
        "No evaluation API calls are made by this tab."
    )

    comparison_rows = [
        {
            "Metric": "Reported response accuracy",
            "Google": "100.0%",
            "MiniLM": "100.0%",
        },
        {
            "Metric": "Contextual relevance",
            "Google": "94.3%",
            "MiniLM": "93.8%",
        },
        {
            "Metric": "Average application latency",
            "Google": "3.70 s",
            "MiniLM": "1.18 s",
        },
        {
            "Metric": "Simulated satisfaction",
            "Google": "5.00/5",
            "MiniLM": "5.00/5",
        },
        {
            "Metric": "RAGAS faithfulness",
            "Google": "92.7%",
            "MiniLM": "96.3%",
        },
        {
            "Metric": "Strict grounding-flag incidence",
            "Google": "35.0%",
            "MiniLM": "10.0%",
        },
        {
            "Metric": "Faithfulness deficit",
            "Google": "7.3%",
            "MiniLM": "3.7%",
        },
        # {
        #     "Metric": "Estimated generation cost/query",
        #     "Google": "$0.000430",
        #     "MiniLM": "$0.0",
        # },
    ]

    st.dataframe(
        comparison_rows,
        hide_index=True,
        use_container_width=True,
    )

    st.write(
        "MiniLM showed faster responses and higher faithfulness in these "
        "runs. Google showed slightly higher contextual relevance."
    )

    st.caption(
        "These historical results are not the latest revised pipeline "
        "scores and do not prove that embeddings alone caused the "
        "differences. Cost estimates exclude embedding and evaluation "
        "charges. Faithfulness deficit equals 100% minus mean faithfulness."
    )

# -----------------------------------------------------------------------------
# SIDEBAR: SYSTEM STATUS AND SESSION STATISTICS
# -----------------------------------------------------------------------------

with sidebar_content:
    st.subheader("System status")
    st.success("RAG pipeline ready")
    st.caption(f"Model: {pipeline['model_name']}")
    st.caption(f"Embedding model: {pipeline['embedding_model_name']}")

    assistant_messages = [
        message for message in st.session_state.messages
        if message["role"] == "assistant"
    ]

    successful = [
        message for message in assistant_messages
        if message["result"]["status"] == "success"
    ]

    latencies = [
        message["result"]["total_latency"]
        for message in successful
    ]

    ratings = [
        message["rating"]
        for message in successful
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
        data=json.dumps(
            st.session_state.messages,
            indent=2,
            ensure_ascii=False,
        ),
        file_name="insurance_chat_transcript.json",
        mime="application/json",
        use_container_width=True,
        disabled=not st.session_state.messages,
    )

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conversation_number += 1
        st.rerun()