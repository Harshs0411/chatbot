"""
Production-oriented RAG Evaluation Engine

Evaluates a single RAG turn using three core metrics:

1. Faithfulness
   - Is the generated answer supported by the retrieved context?

2. Answer Relevance
   - Does the answer directly answer the user's question?

3. Context Precision
   - Are the retrieved chunks relevant to the user's question?

This module is intentionally independent from the RAG pipeline.
It accepts:
    question + retrieved context + generated answer

and returns structured evaluation results.
"""

from typing import List, Dict, Any

from pydantic import BaseModel, Field, field_validator
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

try:
    from backend.backend import get_chat_model
except ImportError:
    from backend import get_chat_model


# ============================================================
# 1. Evaluation Result Schema
# ============================================================

class RAGTriadEvaluation(BaseModel):

    faithfulness_score: float = Field(
        description=(
            "Score from 1.0 to 5.0 measuring whether "
            "the answer is supported by the retrieved context."
        )
    )

    faithfulness_reasoning: str = Field(
        description="Brief explanation of the faithfulness score."
    )

    answer_relevance_score: float = Field(
        description=(
            "Score from 1.0 to 5.0 measuring whether "
            "the answer directly addresses the question."
        )
    )

    answer_relevance_reasoning: str = Field(
        description="Brief explanation of the answer relevance score."
    )

    context_precision_score: float = Field(
        description=(
            "Score from 1.0 to 5.0 measuring whether "
            "the retrieved chunks are relevant to the question."
        )
    )

    context_precision_reasoning: str = Field(
        description="Brief explanation of the context precision score."
    )

    @field_validator(
        "faithfulness_score",
        "answer_relevance_score",
        "context_precision_score",
    )
    @classmethod
    def validate_score(cls, value: float) -> float:

        if not 1.0 <= value <= 5.0:
            raise ValueError(
                "Score must be between 1.0 and 5.0"
            )

        return value

    
# ============================================================
# 2. LLM + Output Parser
# ============================================================

eval_model = get_chat_model()

eval_parser = PydanticOutputParser(
    pydantic_object=RAGTriadEvaluation
)


# ============================================================
# 3. Evaluation Prompt
# ============================================================

EVALUATION_PROMPT = PromptTemplate(
    template="""
You are an impartial evaluator for a Retrieval-Augmented Generation (RAG)
system.

Evaluate the RAG response using ONLY the information provided below.

IMPORTANT RULES:

- Do not use outside knowledge.
- A fact being true in the real world does NOT make it faithful.
- For faithfulness, every factual claim in the answer must be supported
  by the retrieved context.
- If the context does not contain enough information, the answer should
  not receive a high faithfulness score merely because the answer sounds correct.
- Evaluate the retrieved chunks individually when judging context precision.
- Do not reward verbosity.
- Do not penalize an answer merely for being concise if it answers the question.

============================================================
METRIC 1: FAITHFULNESS
============================================================

Question:
{question}

Retrieved Context:
{context}

Generated Answer:
{answer}

Score:

5.0 = All important claims are clearly supported by the context.
4.0 = Almost completely supported; only very minor unsupported details.
3.0 = Partially supported; some meaningful claims are unsupported.
2.0 = Large portions are unsupported or weakly grounded.
1.0 = Mostly unsupported, contradicts the context, or hallucinates.

============================================================
METRIC 2: ANSWER RELEVANCE
============================================================

Evaluate whether the generated answer directly answers the question.

5.0 = Direct, complete and focused answer.
4.0 = Mostly complete with minor omissions.
3.0 = Partially answers the question.
2.0 = Mostly off-topic or incomplete.
1.0 = Does not answer the question.

============================================================
METRIC 3: CONTEXT PRECISION
============================================================

Evaluate whether the retrieved chunks are useful for answering the question.

5.0 = Almost all retrieved chunks are directly useful.
4.0 = Most chunks are useful with minor irrelevant material.
3.0 = Mixed relevant and irrelevant chunks.
2.0 = Most chunks are weakly related.
1.0 = Retrieved context is essentially irrelevant.

============================================================
OUTPUT
============================================================

Return ONLY structured output matching the following schema:

{format_instructions}

Do not calculate the overall score yourself.
The application will calculate it from the three metric scores.
""",
    input_variables=[
        "question",
        "context",
        "answer",
    ],
    partial_variables={
        "format_instructions": eval_parser.get_format_instructions()
    },
)


# ============================================================
# 4. Helper Functions
# ============================================================

def calculate_overall_score(
    faithfulness: float,
    relevance: float,
    precision: float,
) -> float:
    """
    Calculate the final weighted score.

    Weights:
        Faithfulness       = 40%
        Answer Relevance   = 30%
        Context Precision  = 30%

    Faithfulness receives slightly higher weight because
    grounding is particularly important in RAG systems.
    """

    score = (
        0.40 * faithfulness
        + 0.30 * relevance
        + 0.30 * precision
    )

    return round(score, 2)


def get_verdict(score: float) -> str:
    """
    Convert numerical score into a human-readable category.
    """

    if score >= 4.5:
        return "EXCELLENT"

    if score >= 3.5:
        return "GOOD"

    if score >= 2.5:
        return "ACCEPTABLE"

    return "POOR"


# ============================================================
# 5. Main Evaluation Function
# ============================================================

def evaluate_rag_turn(
    question: str,
    context: List[str],
    answer: str,
) -> Dict[str, Any]:
    """
    Evaluate one RAG question-answer turn.

    Parameters
    ----------
    question:
        User's original question.

    context:
        Retrieved chunks returned by the RAG retriever.

    answer:
        Final answer generated by the RAG system.

    Returns
    -------
    Dictionary containing:
        - evaluation_status
        - three metric scores
        - reasoning
        - overall score
        - verdict
    """

    # --------------------------------------------------------
    # Input validation
    # --------------------------------------------------------

    if not question or not question.strip():
        return {
            "evaluation_status": "invalid_input",
            "error": "Question cannot be empty."
        }

    if not answer or not answer.strip():
        return {
            "evaluation_status": "invalid_input",
            "error": "Answer cannot be empty."
        }

    # --------------------------------------------------------
    # Format retrieved context
    # --------------------------------------------------------

    if not context:
        context_str = "[NO CONTEXT WAS RETRIEVED]"
    else:
        context_str = "\n\n".join(
            [
                f"--- Chunk {i + 1} ---\n{chunk}"
                for i, chunk in enumerate(context)
            ]
        )

    # --------------------------------------------------------
    # Build evaluation chain
    # --------------------------------------------------------

    chain = (
        EVALUATION_PROMPT
        | eval_model
        | eval_parser
    )

    # --------------------------------------------------------
    # Execute evaluation
    # --------------------------------------------------------

    try:

        result = chain.invoke(
            {
                "question": question,
                "context": context_str,
                "answer": answer,
            }
        )

        # ----------------------------------------------------
        # Calculate overall score ourselves
        # ----------------------------------------------------

        overall_score = calculate_overall_score(
            faithfulness=result.faithfulness_score,
            relevance=result.answer_relevance_score,
            precision=result.context_precision_score,
        )

        verdict = get_verdict(overall_score)

        return {
            "evaluation_status": "success",

            "faithfulness_score": result.faithfulness_score,
            "faithfulness_reasoning": result.faithfulness_reasoning,

            "answer_relevance_score": result.answer_relevance_score,
            "answer_relevance_reasoning":
                result.answer_relevance_reasoning,

            "context_precision_score":
                result.context_precision_score,
            "context_precision_reasoning":
                result.context_precision_reasoning,

            "overall_score": overall_score,
            "verdict": verdict,
        }

    # --------------------------------------------------------
    # IMPORTANT:
    # Never fabricate evaluation scores if the evaluator fails.
    # --------------------------------------------------------

    except Exception as e:

        return {
            "evaluation_status": "evaluation_error",

            "faithfulness_score": None,
            "faithfulness_reasoning": None,

            "answer_relevance_score": None,
            "answer_relevance_reasoning": None,

            "context_precision_score": None,
            "context_precision_reasoning": None,

            "overall_score": None,
            "verdict": "ERROR",

            "error": str(e),
        }