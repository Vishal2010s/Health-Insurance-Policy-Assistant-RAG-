"""
Health Insurance RAG Pipeline
LangChain + Gemini + FAISS
"""
print("***Process Started with Imports***")
import logging
import warnings

# FAISS currently remains in the archived langchain-community package.
# Hide only its known sunset notice; do not hide unrelated warnings.
warnings.filterwarnings(
    "ignore",
    message=r"`langchain-community` is being sunset.*",
    category=DeprecationWarning,
)

# The Google SDK currently prints an AFC migration notice even though this
# RAG pipeline does not define or call any tools. Hide only that SDK logger.
logging.getLogger("google_genai.models").setLevel(logging.ERROR)
logging.getLogger("google.genai.models").setLevel(logging.ERROR)

import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA

print("✓ IMPORTS COMPLETE")
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# EMBEDDING_MODEL_NAME = "models/gemini-embedding-001"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────

def load_config():
    """Load application settings from the local .env file."""
    load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=True)
    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("MODEL_NAME", "gemini-3.5-flash-lite")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env file.")
    return api_key, model_name

# ──────────────────────────────────────────────
# LANGSMITH OBSERVABILITY  
# ──────────────────────────────────────────────

def setup_langsmith():
    """Enable LangSmith only when tracing is explicitly requested."""
    tracing_enabled = (
        os.getenv("LANGSMITH_TRACING", "false").strip().lower() == "true"
    )
    langsmith_key = os.getenv("LANGSMITH_API_KEY")

    if tracing_enabled and langsmith_key:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ.setdefault("LANGSMITH_PROJECT", "insurance-rag")
        print("✓ LangSmith tracing enabled")
    else:
        # Explicitly disable both current and legacy tracing flags. This stops
        # background uploads left enabled by a terminal or older .env file.
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"

        if tracing_enabled and not langsmith_key:
            print("⚠ LangSmith API key not found — tracing disabled")
        else:
            print("ℹ LangSmith tracing disabled")


# ──────────────────────────────────────────────
# DATA CLEANING 
# ──────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Remove HTML tags, boilerplate phrases, and normalize whitespace."""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ──────────────────────────────────────────────
# KNOWLEDGE BASE
# ──────────────────────────────────────────────

def load_knowledge_base(kb_path="data/knowledge_base.json"):
    path = Path(kb_path)
    if not path.is_absolute() and not path.is_file():
        path = Path(__file__).resolve().parent / path
    try:
        with path.open("r", encoding="utf-8") as f:
            kb = json.load(f)
        if not isinstance(kb, list):
            raise ValueError("Knowledge base must be a JSON list.")
        print(f"✓ Loaded {len(kb)} documents from {path}")
        return kb
    except FileNotFoundError:
        raise FileNotFoundError(f"KB file not found: {path}")


def create_documents(kb: list) -> list:
    """Convert KB records into LangChain Documents with metadata + cleaning."""
    documents = []
    for item in kb:
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        content = clean_text(content)
        title = clean_text(item.get("title", "Untitled"))
        documents.append(Document(
            page_content=f"{title}\n\n{content}",
            metadata={
                "doc_id": item.get("doc_id", "UNKNOWN"),
                "category": item.get("category", "general"),
                "title": title,
                "source": f"KB - {item.get('doc_id', 'UNKNOWN')}",
            }
        ))
    print(f"✓ Created {len(documents)} LangChain documents")
    return documents


# ──────────────────────────────────────────────
# VECTOR STORE
# ──────────────────────────────────────────────

def create_vector_store(kb: list, chunk_size=500, chunk_overlap=50):

    documents = create_documents(kb)

    # Manual chunking
    chunks = []

    for doc in documents:
        text = doc.page_content

        start = 0

        while start < len(text):
            end = start + chunk_size

            chunks.append(
                Document(
                    page_content=text[start:end],
                    metadata=doc.metadata.copy()
                )
            )

            start += chunk_size - chunk_overlap

    # Add chunk IDs
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"chunk_{i+1}"

    print(f"✓ Created {len(chunks)} chunks")

    print("Loading embedding model...")
    
    # api_key = os.getenv("GEMINI_API_KEY")

    # embeddings = GoogleGenerativeAIEmbeddings(
    #     model=EMBEDDING_MODEL_NAME,
    #     google_api_key=api_key,
    # )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    print("✓ Embedding model initialized")
    # print("✓ Embedding model initialized: all-MiniLM-L6-v2")

    print("Testing embedding model...")

    test_vector = embeddings.embed_query("test insurance policy")

    print("✓ Embedding generated successfully")
    print(f"Vector length = {len(test_vector)}")

    print("Creating FAISS vector store...")

    try:
        vector_store = FAISS.from_documents(chunks, embeddings)
        print("✓ FAISS vector store created successfully")

    except Exception as e:
        print(f"FAISS creation failed: {e}")
        raise

    return vector_store

# ──────────────────────────────────────────────
# PROMPT
# ──────────────────────────────────────────────

SYSTEM_PROMPT = """You are a professional health insurance policy assistant.

Answer using ONLY the retrieved context below. Follow these rules:

1. Do not invent policy information or use outside knowledge.
2. Preserve all numbers, limits, waiting periods, conditions and deadlines exactly.
3. Clearly mention important exclusions, conditions, or exceptions when relevant.
4. Explain policy terms in simple, customer-friendly language.
5. Use bullet points or numbered steps for procedures.
6. For claim procedures, provide steps in the correct order and mention required documents or deadlines when available.
7.If the answer is not available in the context, say:
   "I'm sorry, I don't have information on that in the available policy information. Would you like me to connect you with a human representative?"
8. Do not guarantee claim approval, coverage, reimbursement, or premium amounts.    
9. Cite the specific clause when possible (e.g., "According to the Hospitalization Coverage clause...").
10. If multiple clauses are relevant, synthesize them into one coherent answer.
11. Keep simple answers concise. Use bullet points or numbered steps for detailed answers.
12. Do not combine information from unrelated policy clauses unless the retrieved context clearly supports the connection.

Retrieved Context:
{context}

User Question:
{question}

Answer:"""


# ──────────────────────────────────────────────
# LLM + QA CHAIN
# ──────────────────────────────────────────────

def initialize_llm(api_key, model_name):
    """Initialize Gemini without unsupported sampling parameters."""
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
    )
    print(f"✓ LLM initialized: {model_name}")
    return llm


def create_qa_chain(llm, vector_store, top_k=3):
    prompt = PromptTemplate(
        template=SYSTEM_PROMPT, input_variables=["context", "question"]
    )
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": top_k}
    )
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
    print(f"✓ QA chain ready (top-k={top_k})")
    return chain


# ──────────────────────────────────────────────
# QUERY FUNCTIONS
# ──────────────────────────────────────────────

def extract_answer_text(result) -> str:
    """Return readable text for both old and new Gemini response formats."""
    if isinstance(result, str):
        return result

    if hasattr(result, "text"):
        return result.text

    if isinstance(result, list):
        text_parts = []
        for block in result:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
        return "\n".join(text_parts)

    return str(result)

def query_simple(qa_chain, question: str) -> str:
    if not question or not question.strip():
        return "Please enter a valid question."
    try:
        response = qa_chain.invoke({"query": question.strip()})
        return extract_answer_text(response["result"])
    except Exception as e:
        return f"Error: {e}"


def query_with_sources(qa_chain, question: str) -> dict:
    if not question or not question.strip():
        return {
            "question": question, "answer": "Please enter a valid question.",
            "sources": [], "retrieval_latency": 0, "generation_latency": 0,
            "total_latency": 0,
        }
    question = question.strip()
    total_start = time.perf_counter()
    try:
        response = qa_chain.invoke({"query": question})
        total_end = time.perf_counter()

        sources = []
        for doc in response.get("source_documents", []):
            sources.append({
                "doc_id": doc.metadata.get("doc_id", "UNKNOWN"),
                "category": doc.metadata.get("category", "UNKNOWN"),
                "title": doc.metadata.get("title", "UNKNOWN"),
                "chunk_id": doc.metadata.get("chunk_id", "UNKNOWN"),
                "content": doc.page_content,
            })

        return {
            "question": question,
            "answer": extract_answer_text(response.get("result", "")),
            "sources": sources,
            "retrieval_latency": 0.0,
            "generation_latency": round(total_end - total_start, 4),
            "total_latency": round(total_end - total_start, 4),
        }
    except Exception as e:
        return {
            "question": question, "answer": f"Error: {e}",
            "sources": [], "retrieval_latency": 0, "generation_latency": 0,
            "total_latency": round(time.perf_counter() - total_start, 4),
        }


# ──────────────────────────────────────────────
# MAIN CLASS
# ──────────────────────────────────────────────

class HealthInsuranceRAG:

    def __init__(self, kb_path="data/knowledge_base.json"):
        self.kb_path = kb_path
        self.kb = None
        self.vector_store = None
        self.llm = None
        self.qa_chain = None
        self.model_name = None
        self.embedding_model_name = EMBEDDING_MODEL_NAME  #Hugging Face embedding model information

    def setup(self):
        print("\n" + "=" * 50)
        print("INITIALIZING RAG PIPELINE")
        print("=" * 50)
        # Load .env before checking the LangSmith settings.
        api_key, self.model_name = load_config()
        setup_langsmith()
        self.kb = load_knowledge_base(self.kb_path)
        self.vector_store = create_vector_store(self.kb)
        self.llm = initialize_llm(api_key, self.model_name)
        self.qa_chain = create_qa_chain(self.llm, self.vector_store)
        print("\n✓ Pipeline ready!\n")

    def query(self, question: str) -> str:
        if self.qa_chain is None:
            raise RuntimeError("Call setup() first.")
        return query_simple(self.qa_chain, question)

    def query_with_sources(self, question: str) -> dict:
        if self.qa_chain is None:
            raise RuntimeError("Call setup() first.")
        return query_with_sources(self.qa_chain, question)

    def batch_query(self, questions: list) -> list:
        return [self.query_with_sources(q) for q in questions]

if __name__ == "__main__":
    print("ENTERED MAIN")

    rag = HealthInsuranceRAG()

    print("OBJECT CREATED")

    rag.setup()

    print("SETUP FINISHED")

    test = rag.query_with_sources("What is the waiting period for pre-existing diseases?")
    print(f"Q: {test['question']}")
    print(f"A: {test['answer']}")
    print(f"Sources: {[s['doc_id'] for s in test['sources']]}")
    print(f"Latency: {test['total_latency']}s")



