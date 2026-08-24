"""Simple eight-metric evaluation for the Health Insurance RAG assistant.

Keep this file beside ``master_HF.py`` and run:

    python evaluation_concise.py

The evaluator uses one clear loop over the test cases. Gemini calls are spaced
out and retried when a quota/HTTP 429 error occurs.
"""

import asyncio
import inspect
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from langchain_google_genai import ChatGoogleGenerativeAI

# If your pipeline file is named master.py, change master_HF to master here.
from master_HF import HealthInsuranceRAG, SYSTEM_PROMPT, extract_answer_text


# -----------------------------------------------------------------------------
# SETTINGS
# -----------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
RESULTS_FILE = BASE_DIR / "evaluation_results.json"
TOP_K = int(os.getenv("EVALUATION_TOP_K", "3"))
TEST_LIMIT = int(os.getenv("EVALUATION_LIMIT", "20"))

# Wait between application, judge and RAGAS calls to reduce quota bursts.
REQUEST_DELAY_SECONDS = float(os.getenv("EVALUATION_DELAY_SECONDS", "5"))
MAX_RETRIES = int(os.getenv("EVALUATION_RETRIES", "4"))
RETRY_WAIT_SECONDS = float(os.getenv("EVALUATION_RETRY_WAIT_SECONDS", "60"))

# # Gemini 3.5 Flash-Lite paid standard-tier prices per one million tokens.
INPUT_PRICE_PER_MILLION = float(os.getenv("LLM_INPUT_PRICE_USD", "0.30"))
OUTPUT_PRICE_PER_MILLION = float(os.getenv("LLM_OUTPUT_PRICE_USD", "2.50"))

# A score below 1 means that RAGAS found at least one unsupported claim.
HALLUCINATION_THRESHOLD = float(
    os.getenv("HALLUCINATION_THRESHOLD", "0.999")
)


# -----------------------------------------------------------------------------
# TEST DATA: ONE QUESTION FOR EACH OF THE 20 POLICY DOCUMENTS
# -----------------------------------------------------------------------------

def case(question, doc_id):
    """Create one readable test case."""
    return {
        "question": question,
        "expected_doc_ids": [doc_id],
    }


TEST_CASES = [
    case("What expenses are covered during hospitalization?", "KB001"),
    case("Are pre and post hospitalization expenses covered?", "KB002"),
    case("Which day care procedures are covered?", "KB003"),
    case("Is ambulance charge covered?", "KB004"),
    case("What is the waiting period for pre-existing diseases?", "KB005"),
    case("Is there an initial waiting period?", "KB006"),
    case("What is the waiting period for cataract or hernia?", "KB007"),
    case("Which treatments are permanently excluded?", "KB008"),
    case("How do I file a cashless claim?", "KB009"),
    case("How do I file a reimbursement claim?", "KB010"),
    case("How long does claim settlement take?", "KB011"),
    case("What are common reasons for claim rejection?", "KB012"),
    case("How is the premium calculated?", "KB013"),
    case("What is the grace period for renewal?", "KB014"),
    case("What is the No Claim Bonus?", "KB015"),
    case("Can I port the policy to another insurer?", "KB016"),
    case("Is maternity covered?", "KB017"),
    case("What does the critical illness rider cover?", "KB018"),
    case("How can I find a network hospital?", "KB019"),
    case("What is the free look period?", "KB020"),
]


# -----------------------------------------------------------------------------
# SIMPLE QUOTA HANDLING
# -----------------------------------------------------------------------------

def is_quota_error(error):
    """Recognize the common Gemini quota messages."""
    message = str(error).casefold()
    return any(
        marker in message
        for marker in ("429", "quota", "rate limit", "resource_exhausted")
    )


async def api_call(action, label):
    """Run one sync or async API action and retry only quota errors."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = action()
            return await result if inspect.isawaitable(result) else result
        except Exception as error:
            if not is_quota_error(error) or attempt == MAX_RETRIES:
                raise
            print(
                f"  Quota reached during {label}. Waiting "
                f"{RETRY_WAIT_SECONDS:.0f}s before retry {attempt + 1}/"
                f"{MAX_RETRIES}..."
            )
            await asyncio.sleep(RETRY_WAIT_SECONDS)


async def pause_for_quota():
    """Space out consecutive Gemini-backed operations."""
    if REQUEST_DELAY_SECONDS > 0:
        await asyncio.sleep(REQUEST_DELAY_SECONDS)


# -----------------------------------------------------------------------------
# METRICS 1, 2 AND 4: ONE QUALITY-JUDGE CALL
# -----------------------------------------------------------------------------

QUALITY_PROMPT = """Evaluate this health-insurance RAG response.

Question:
{question}

Actual policy text:
{policy_text}

Generated answer:
{answer}

Retrieved clauses:
{retrieved_context}

Return only valid JSON:
{{
  "answer_relevance": <0 to 1>,
  "factual_correctness": <0 to 1>,
  "contextual_relevance": <0 to 1>,
  "simulated_user_satisfaction": <integer 1 to 5>
}}

Score factual correctness against the actual policy text. Score contextual
relevance based on whether the retrieved clauses help answer the question.
The satisfaction score is simulated and should reflect correctness, clarity,
completeness and usefulness.
"""


def actual_policy_text(rag, expected_doc_ids):
    """Return the authoritative KB text for the expected policy IDs."""
    matching_records = [
        item for item in rag.kb if item.get("doc_id") in expected_doc_ids
    ]
    if not matching_records:
        raise ValueError(f"Policy IDs not found in KB: {expected_doc_ids}")
    return "\n\n".join(
        f"[{item['doc_id']} - {item['title']}]\n{item['content']}"
        for item in matching_records
    )


def parse_json(message):
    """Parse a JSON object even if Gemini adds a Markdown code fence."""
    text = extract_answer_text(message)
    start = text.find("{")
    end = text.rfind("}") + 1
    if start < 0 or end <= start:
        raise ValueError(f"Quality judge did not return JSON: {text}")
    return json.loads(text[start:end])


def quality_scores(judge_llm, rag, test, response):
    """Return accuracy, context relevance and simulated satisfaction."""
    context = "\n\n".join(
        f"[{source['doc_id']} - {source['title']}]\n{source['content']}"
        for source in response["sources"]
    )
    prompt = QUALITY_PROMPT.format(
        question=test["question"],
        policy_text=actual_policy_text(rag, test["expected_doc_ids"]),
        answer=response["answer"],
        retrieved_context=context,
    )
    scores = parse_json(judge_llm.invoke(prompt))

    relevance = max(0.0, min(1.0, float(scores["answer_relevance"])))
    correctness = max(0.0, min(1.0, float(scores["factual_correctness"])))
    context_relevance = max(
        0.0, min(1.0, float(scores["contextual_relevance"]))
    )
    satisfaction = round(
        max(1.0, min(5.0, float(scores["simulated_user_satisfaction"])))
    )

    return {
        "answer_relevance": relevance,
        "factual_correctness": correctness,
        "response_accuracy": (relevance + correctness) / 2,
        "contextual_relevance": context_relevance,
        "simulated_user_satisfaction": satisfaction,
    }


# # -----------------------------------------------------------------------------
# # METRIC 5: DOCUMENT-LEVEL PRECISION@K AND RECALL@K
# # -----------------------------------------------------------------------------

# def precision_recall(retrieved_ids, expected_ids):
#     """Compare retrieved policy IDs with the annotated expected IDs."""
#     relevant_ids = set(retrieved_ids) & set(expected_ids)
#     precision = len(relevant_ids) / len(retrieved_ids) if retrieved_ids else 0.0
#     recall = len(relevant_ids) / len(expected_ids) if expected_ids else 0.0
#     return precision, recall


# -----------------------------------------------------------------------------
# METRICS 6 AND 7: OFFICIAL RAGAS FAITHFULNESS AND HALLUCINATION
# -----------------------------------------------------------------------------

def create_faithfulness_scorer(model_name, api_key):
    """Create the RAGAS 0.4 Faithfulness scorer for Gemini."""
    try:
        import instructor
        from google import genai
        from ragas.llms import InstructorLLM
        from ragas.metrics.collections import Faithfulness
    except ImportError as error:
        raise ImportError(
            "Install RAGAS support with: python -m pip install "
            "'ragas>=0.4,<1.0' 'instructor[google-genai]>=1.7,<2.0'"
        ) from error

    client = instructor.from_genai(
        genai.Client(api_key=api_key),
        use_async=True,
    )
    evaluator_llm = InstructorLLM(
        client=client,
        model=model_name,
        provider="google",
    )
    return Faithfulness(llm=evaluator_llm)


async def ragas_faithfulness(scorer, test, response):
    """Return the official RAGAS faithfulness score from 0 to 1."""
    result = await scorer.ascore(
        user_input=test["question"],
        response=response["answer"],
        retrieved_contexts=[
            source["content"] for source in response["sources"]
        ],
    )
    return max(0.0, min(1.0, float(result.value)))


# -----------------------------------------------------------------------------
# METRIC 8: ESTIMATED PRODUCTION COST PER QUERY
# -----------------------------------------------------------------------------

def estimate_tokens(text):
    """Estimate tokens at approximately four characters per token."""
    return max(1, round(len(text) / 4))


def production_cost(test, response):
    """Estimate application LLM cost; local Hugging Face embeddings cost $0."""
    context = "\n\n".join(
        source["content"] for source in response["sources"]
    )
    prompt = SYSTEM_PROMPT.format(
        context=context,
        question=test["question"],
    )
    input_tokens = estimate_tokens(prompt)
    output_tokens = estimate_tokens(response["answer"])
    cost = (
        input_tokens * INPUT_PRICE_PER_MILLION
        + output_tokens * OUTPUT_PRICE_PER_MILLION
    ) / 1_000_000
    return {
        "estimated_input_tokens": input_tokens,
        "estimated_output_tokens": output_tokens,
        "embedding_api_cost_usd": 0.0,
        "estimated_cost_usd": cost,
    }


# -----------------------------------------------------------------------------
# MAIN EVALUATION LOOP
# -----------------------------------------------------------------------------

async def run_evaluation():
    """Evaluate every selected case and print all eight metrics."""
    print("=" * 68)
    print("HEALTH INSURANCE RAG EVALUATION — ALL 8 METRICS")
    print("=" * 68)

    rag = HealthInsuranceRAG(
        kb_path=str(BASE_DIR / "data" / "knowledge_base.json")
    )
    rag.setup()
    # Keep actual retrieval and evaluation TOP_K synchronized
    rag.qa_chain.retriever.search_kwargs["k"] = TOP_K
    print(f"✓ Evaluator retrieval configured: top-k={TOP_K}")
    api_key = os.getenv("GEMINI_API_KEY")
    evaluator_model = os.getenv("EVALUATOR_MODEL", rag.model_name)
    judge_llm = ChatGoogleGenerativeAI(
        model=evaluator_model,
        google_api_key=api_key,
    )
    faithfulness_scorer = create_faithfulness_scorer(
        evaluator_model,
        api_key,
    )

    test_count = max(1, min(TEST_LIMIT, len(TEST_CASES)))
    selected_tests = TEST_CASES[:test_count]
    results = []
    started = time.perf_counter()

    for number, test in enumerate(selected_tests, start=1):
        print(f"\n{number:02d}/{test_count:02d} {test['question']}")

        response = await api_call(
            lambda: rag.query_with_sources(test["question"]),
            "RAG answer generation",
        )
        await pause_for_quota()

        quality = await api_call(
            lambda: quality_scores(judge_llm, rag, test, response),
            "quality evaluation",
        )
        await pause_for_quota()

        faithfulness = await api_call(
            lambda: ragas_faithfulness(
                faithfulness_scorer,
                test,
                response,
            ),
            "RAGAS faithfulness",
        )

        retrieved_ids = [
            source["doc_id"] for source in response["sources"]
        ]
        # precision, recall = precision_recall(
        #     retrieved_ids,
        #     test["expected_doc_ids"],
        # )
        cost = production_cost(test, response)

        record = {
            **test,
            "answer": response["answer"],
            "retrieved_doc_ids": retrieved_ids,
            "sources": response["sources"],
            **quality,
            # "precision_at_k": precision,
            # "recall_at_k": recall,
            "latency_seconds": float(response["total_latency"]),
            "ragas_faithfulness": faithfulness,
            "hallucination_detected": (
                faithfulness < HALLUCINATION_THRESHOLD
            ),
            **cost,
        }
        results.append(record)

        print(
            f"  accuracy={record['response_accuracy']:.0%} | "
            f"context={record['contextual_relevance']:.0%} | "
            # f"P@{TOP_K}={precision:.0%} R@{TOP_K}={recall:.0%} | "
            f"RAGAS={faithfulness:.2f} | "
            f"latency={record['latency_seconds']:.2f}s"
        )
        await pause_for_quota()
    metrics = {
        "response_accuracy": mean(r["response_accuracy"] for r in results),
        "answer_relevance": mean(r["answer_relevance"] for r in results),
        "factual_correctness": mean(
            r["factual_correctness"] for r in results
        ),
        "contextual_relevance": mean(
            r["contextual_relevance"] for r in results
        ),
        "average_latency_seconds": mean(
            r["latency_seconds"] for r in results
        ),
        "simulated_user_satisfaction_out_of_5": mean(
            r["simulated_user_satisfaction"] for r in results
        ),
        # "precision_at_k": mean(r["precision_at_k"] for r in results),
        # "recall_at_k": mean(r["recall_at_k"] for r in results),
        "ragas_faithfulness": mean(
            r["ragas_faithfulness"] for r in results
        ),
        "strict_hallucination_rate": mean(
            r["ragas_faithfulness"] < HALLUCINATION_THRESHOLD
            for r in results
        ),
        "material_hallucination_rate": mean(
            r["ragas_faithfulness"] < 0.80
            for r in results
        ),
        "unsupported_claim_rate": 1.0 - mean(
            r["ragas_faithfulness"] for r in results
        ),
        "average_cost_per_query_usd": mean(
            r["estimated_cost_usd"] for r in results
        ),
    }

    output = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "application_model": rag.model_name,
        "evaluator_model": evaluator_model,
        "embedding_model": getattr(
            rag,
            "embedding_model_name",
            "sentence-transformers/all-MiniLM-L6-v2",
        ),
        "embedding_api_cost_usd": 0.0,
        "top_k": TOP_K,
        "test_case_count": len(results),
        "metrics": metrics,
        "evaluation_runtime_seconds": round(
            time.perf_counter() - started,
            2,
        ),
        "cost_note": (
            "Estimated application Gemini generation cost only. Quality-judge "
            "and RAGAS evaluation calls are excluded. Hugging Face embeddings "
            "run locally and have no embedding API charge."
        ),
        "results": results,
    }

    with RESULTS_FILE.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)

    print("\n" + "=" * 68)
    print("FINAL SUMMARY")
    print("=" * 68)
    print(f"1. Response accuracy:       {metrics['response_accuracy']:.1%}")
    print(f"2. Contextual relevance:    {metrics['contextual_relevance']:.1%}")
    print(f"3. Average latency:         {metrics['average_latency_seconds']:.2f}s")
    print(
        "4. Simulated satisfaction: "
        f"{metrics['simulated_user_satisfaction_out_of_5']:.2f}/5"
    )
    print(
        # f"5. Precision@{TOP_K} / Recall@{TOP_K}: "
        # f"{metrics['precision_at_k']:.1%} / {metrics['recall_at_k']:.1%}"
    )
    print(f"6. RAGAS faithfulness:      {metrics['ragas_faithfulness']:.1%}")
    print(
    "7a. Strict hallucination incidence: "
        f"{metrics['strict_hallucination_rate']:.1%}"
    )
    print(
        "7b. Material hallucination rate:    "
        f"{metrics['material_hallucination_rate']:.1%}"
    )
    print(
        "7c. Unsupported-claim share:         "
        f"{metrics['unsupported_claim_rate']:.1%}"
    )
    print(
        "8. Average cost/query:     "
        f"${metrics['average_cost_per_query_usd']:.6f} USD"
    )
    print(f"\nDetailed results: {RESULTS_FILE}")


if __name__ == "__main__":
    asyncio.run(run_evaluation())
