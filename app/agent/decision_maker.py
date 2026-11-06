
import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from app.agent.context_builder import build_retrieval_context

load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "gpt-6-luna")
client = OpenAI()


def is_agentic_rag_question(question: str) -> bool:
    """
    Classify whether a question is directly related to Agentic RAG.

    This check runs before retrieval. Out-of-scope questions are rejected
    without accessing the knowledge base.
    """

    classification_prompt = f"""
You are a strict domain classifier for an Agentic RAG research assistant.

ALLOWED TOPICS:
- Agentic Retrieval-Augmented Generation (Agentic RAG)
- Traditional RAG and its comparison with Agentic RAG
- RAG architectures and retrieval mechanisms
- LLM agents and agentic retrieval workflows
- Multi-agent systems in the context of RAG
- Agentic RAG components, techniques, and workflows
- Technical capabilities, limitations, challenges, and applications
  of Agentic RAG and traditional RAG

CLASSIFICATION RULES:
1. Classify a question as IN_SCOPE only if its actual subject is
   directly related to the allowed topics.
2. A question does not become in-scope merely because it mentions
   Agentic RAG.
3. Reject unrelated programming tasks, general knowledge questions,
   personal advice, financial questions, weather questions, and
   unrelated technical topics.
4. If the question is ambiguous or you are uncertain, classify it
   as OUT_OF_SCOPE.
5. Questions about concepts such as embeddings, vector databases,
   or LLMs are in-scope only when they are being discussed in the
   context of RAG or Agentic RAG.

Return ONLY one of these exact labels:
IN_SCOPE
OUT_OF_SCOPE

USER QUESTION:
{question}
"""

    try:
        response = client.responses.create(
            model=MODEL,
            input=classification_prompt
        )

        result = (response.output_text or "").strip().upper()

        return result == "IN_SCOPE"

    except Exception as e:
        print(f"Domain classification error: {e}")

        # Fail closed: do not allow retrieval if classification fails.
        return False


def generate_answer(question: str, k: int = 5) -> str:
    """
    Traditional single-pass RAG:
    1. Check whether the question belongs to the Agentic RAG domain.
    2. Retrieve relevant knowledge.
    3. Generate a grounded answer using retrieved passages.
    """

    # STEP 1: DOMAIN RESTRICTION
    # This happens BEFORE retrieval.
    if not is_agentic_rag_question(question):
        return (
            "## Out of Scope\n\n"
            "I can only answer questions related to Agentic RAG, "
            "traditional RAG, and their associated architectures "
            "and technologies.\n\n"
            "Your question is outside my domain of knowledge."
        )

    # STEP 2: RETRIEVE KNOWLEDGE
    context = build_retrieval_context(question, k=k)
    retrieved = context["retrieved_knowledge"]

    print("\n--- RETRIEVED KNOWLEDGE ---")
    print(json.dumps(retrieved, indent=2, ensure_ascii=False))
    print("---------------------------\n")

    if not retrieved:
        return (
            "## Insufficient information in the knowledge base\n\n"
            "I could not retrieve relevant passages for this question. "
            "Please check that the research documents and web sources "
            "have been ingested."
        )

    # STEP 3: GENERATE A GROUNDED ANSWER
    prompt = f"""
ROLE:
You are a research assistant specialising exclusively in
Retrieval-Augmented Generation (RAG), Agentic RAG, retrieval systems,
LLM agents, and related architectures.

USER QUESTION:
{question}

RETRIEVED SOURCE PASSAGES:
{json.dumps(retrieved, ensure_ascii=False, indent=2)}

TASK:
Answer the user's question using ONLY the retrieved passages.

STRICT GROUNDING RULES:
- Do not use pretrained general knowledge to fill gaps in the evidence.
- Do not invent facts, explanations, quotations, sources, URLs,
  or page numbers.
- If the retrieved passages do not contain sufficient information
  to answer the question, explicitly say:
  "There is insufficient information in the knowledge base
  to answer this question."
- Do not provide an answer based on assumptions or unsupported claims.
- Only cite sources included in the retrieved passages.
- If sources disagree, explain the disagreement.

CITATIONS:
- Cite every important source-derived claim immediately after the claim.
- For a PDF with a page value, use: [Source: filename.pdf, p. 4].
- For a web page, use: [Source: Page title, URL: https://...].
- For a document without a page value, cite its filename only.
- Only list sources actually used in the answer.

RESPONSE FORMAT:
## Answer
Provide a direct, evidence-based answer.

## Key concepts
Explain relevant concepts, components, or workflow steps where
the retrieved evidence supports them.

## Sources
List the sources actually cited. Do not list unused retrieved sources.

## Limitations
Briefly state any gaps or uncertainty in the retrieved evidence.

Do not produce financial advice or BUY/SELL/HOLD recommendations.
The knowledge base is about Agentic RAG and related research,
not company financial performance.
"""

    response = client.responses.create(
        model=MODEL,
        input=prompt
    )

    return response.output_text or "The model returned an empty response."
