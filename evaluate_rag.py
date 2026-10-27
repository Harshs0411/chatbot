"""
Test harness for the RAG evaluation engine.

This does NOT execute the real RAG pipeline.

It provides controlled examples to verify that
evaluator.py is working correctly.
"""

from evaluator import evaluate_rag_turn


TEST_CASES = [
    {
        "id": "TC-01",
        "name": "Grounded Answer",

        "question":
            "What is the primary role of SqliteSaver?",

        "context": [
            (
                "SqliteSaver is a persistent checkpointer for "
                "LangGraph that stores graph state in an SQLite database."
            ),
            (
                "It associates state snapshots with a thread_id, "
                "allowing conversations to persist across sessions."
            ),
        ],

        "answer": (
            "SqliteSaver is a persistent LangGraph checkpointer "
            "that stores graph state in SQLite and associates "
            "state snapshots with a thread_id."
        ),
    },

    {
        "id": "TC-02",
        "name": "Unsupported Answer",

        "question":
            "What is the primary role of SqliteSaver?",

        "context": [
            (
                "SqliteSaver stores LangGraph state in an "
                "SQLite database."
            )
        ],

        "answer": (
            "SqliteSaver stores LangGraph state in SQLite, "
            "uses Redis internally, and automatically encrypts "
            "all user data."
        ),
    },

    {
        "id": "TC-03",
        "name": "Irrelevant Context",

        "question":
            "What is the primary role of SqliteSaver?",

        "context": [
            "Python is a high-level programming language.",
            "CNNs are commonly used for image processing.",
            "Gradient descent is an optimization algorithm.",
        ],

        "answer": (
            "SqliteSaver is used as a persistent checkpointer "
            "for LangGraph."
        ),
    },
]


def run_tests():

    print("=" * 70)
    print("RAG EVALUATION ENGINE TEST")
    print("=" * 70)

    results = []

    for test in TEST_CASES:

        print(f"\n[{test['id']}] {test['name']}")
        print("-" * 70)

        result = evaluate_rag_turn(
            question=test["question"],
            context=test["context"],
            answer=test["answer"],
        )

        results.append(result)

        if result["evaluation_status"] != "success":

            print("❌ Evaluation failed")
            print("Error:", result.get("error"))
            continue

        print(
            f"Faithfulness: "
            f"{result['faithfulness_score']:.2f}/5"
        )

        print(
            f"Answer Relevance: "
            f"{result['answer_relevance_score']:.2f}/5"
        )

        print(
            f"Context Precision: "
            f"{result['context_precision_score']:.2f}/5"
        )

        print(
            f"Overall: "
            f"{result['overall_score']:.2f}/5"
        )

        print(
            f"Verdict: "
            f"{result['verdict']}"
        )

    # --------------------------------------------------------
    # Calculate averages only from successful evaluations
    # --------------------------------------------------------

    successful = [
        r for r in results
        if r.get("evaluation_status") == "success"
    ]

    print("\n")
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"Total tests: {len(TEST_CASES)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(TEST_CASES) - len(successful)}")

    if successful:

        avg_faithfulness = sum(
            r["faithfulness_score"]
            for r in successful
        ) / len(successful)

        avg_relevance = sum(
            r["answer_relevance_score"]
            for r in successful
        ) / len(successful)

        avg_precision = sum(
            r["context_precision_score"]
            for r in successful
        ) / len(successful)

        avg_overall = sum(
            r["overall_score"]
            for r in successful
        ) / len(successful)

        print(
            f"Average Faithfulness: "
            f"{avg_faithfulness:.2f}/5"
        )

        print(
            f"Average Answer Relevance: "
            f"{avg_relevance:.2f}/5"
        )

        print(
            f"Average Context Precision: "
            f"{avg_precision:.2f}/5"
        )

        print(
            f"Average Overall Score: "
            f"{avg_overall:.2f}/5"
        )


if __name__ == "__main__":
    run_tests()