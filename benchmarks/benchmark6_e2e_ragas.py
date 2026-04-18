from __future__ import annotations

# Requirements note:
#   pip install ragas datasets langchain-google-genai

import argparse
import csv
import json
import os
import random
import time
from typing import Any

from datasets import Dataset
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from neo4j import GraphDatabase
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_correctness, context_recall, faithfulness
import requests

from graphrag.config import (
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
    session_kwargs,
)
from graphrag.graph_retriever import graph_search
from graphrag.vector_retriever import vector_search


OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free")


def _pick_one(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return random.choice(rows)


def _openrouter_answer(question: str, context: str) -> str:
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a supply chain risk assistant. "
                    "Answer using only the provided context. "
                    "If context is insufficient, say so clearly."
                ),
            },
            {
                "role": "user",
                "content": f"Question:\n{question}\n\nRetrieved context:\n{context}",
            },
        ],
        "reasoning": {"enabled": True},
    }

    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        data=json.dumps(payload),
        timeout=90,
    )
    resp.raise_for_status()
    body = resp.json()

    choices = body.get("choices") or []
    if not choices:
        raise RuntimeError(f"OpenRouter returned no choices: {body}")

    msg = choices[0].get("message") or {}
    content = msg.get("content")
    if not content:
        raise RuntimeError(f"OpenRouter returned empty content: {body}")
    return str(content)


def generate_test_set(session) -> list[dict[str, str]]:
    test_set: list[dict[str, str]] = []

    # Type 1 - Direct supplier lookup (favors graph)
    rows_1 = [
        dict(r)
        for r in session.run(
            """
            MATCH (s)-[r]->(o)
            WHERE type(r) IN ['SUPPLIES_TO', 'SOURCES_FROM']
              AND coalesce(s.name, '') <> ''
              AND coalesce(o.name, '') <> ''
            RETURN s.name AS subject, o.name AS object, type(r) AS rel_type
            """
        )
    ]
    picked_1 = _pick_one(rows_1)
    if picked_1:
        q = f"Which companies supply {picked_1['object']} to {picked_1['subject']}?"
        gt = str(picked_1["subject"])
        test_set.append({"question": q, "ground_truth": gt, "query_type": "Direct supplier lookup"})
    else:
        print("[warn] No valid relationships found for Type 1 (supplier lookup).")

    # Type 2 - Risk association (favors hybrid)
    rows_2 = [
        dict(r)
        for r in session.run(
            """
            MATCH (a)-[r]->(b)
            WHERE type(r) IN ['POSES_RISK', 'AFFECTED_BY']
              AND coalesce(a.name, '') <> ''
              AND coalesce(b.name, '') <> ''
            RETURN a.name AS left_name,
                   labels(a) AS left_labels,
                   b.name AS right_name,
                   labels(b) AS right_labels
            """
        )
    ]
    candidates_2: list[dict[str, str]] = []
    for row in rows_2:
        left_is_risk = "RiskEvent" in (row.get("left_labels") or [])
        right_is_risk = "RiskEvent" in (row.get("right_labels") or [])
        if left_is_risk and not right_is_risk:
            candidates_2.append({"entity": str(row["right_name"]), "risk": str(row["left_name"])})
        elif right_is_risk and not left_is_risk:
            candidates_2.append({"entity": str(row["left_name"]), "risk": str(row["right_name"])})

    picked_2 = _pick_one(candidates_2)
    if picked_2:
        q = f"What risk events are associated with {picked_2['entity']}?"
        gt = picked_2["risk"]
        test_set.append({"question": q, "ground_truth": gt, "query_type": "Risk association"})
    else:
        print("[warn] No valid relationships found for Type 2 (risk association).")

    # Type 3 - Location/operational context (favors vector)
    rows_3 = [
        dict(r)
        for r in session.run(
            """
            MATCH (a)-[r]->(b)
            WHERE type(r) IN ['LOCATED_IN', 'OPERATES_IN']
              AND coalesce(a.name, '') <> ''
              AND coalesce(b.name, '') <> ''
            RETURN a.name AS entity_name,
                   labels(a) AS entity_labels,
                   b.name AS place_name,
                   labels(b) AS place_labels
            """
        )
    ]
    candidates_3 = [
        {
            "entity": str(r["entity_name"]),
            "place": str(r["place_name"]),
        }
        for r in rows_3
        if "Company" in (r.get("entity_labels") or [])
        and (
            "Location" in (r.get("place_labels") or [])
            or "Country" in (r.get("place_labels") or [])
        )
    ]
    picked_3 = _pick_one(candidates_3)
    if picked_3:
        q = f"Where does {picked_3['entity']} operate or source materials from?"
        gt = picked_3["place"]
        test_set.append({"question": q, "ground_truth": gt, "query_type": "Location/operational"})
    else:
        print("[warn] No valid relationships found for Type 3 (location/operational).")

    # Type 4 - Regulatory (favors hybrid)
    rows_4 = [
        dict(r)
        for r in session.run(
            """
            MATCH (a)-[r]->(b)
            WHERE type(r) IN ['REGULATED_BY', 'REGULATES']
              AND coalesce(a.name, '') <> ''
              AND coalesce(b.name, '') <> ''
            RETURN a.name AS left_name,
                   labels(a) AS left_labels,
                   b.name AS right_name,
                   labels(b) AS right_labels
            """
        )
    ]
    candidates_4: list[dict[str, str]] = []
    for row in rows_4:
        left_reg = "Regulation" in (row.get("left_labels") or [])
        right_reg = "Regulation" in (row.get("right_labels") or [])
        if left_reg and not right_reg:
            candidates_4.append({"entity": str(row["right_name"]), "reg": str(row["left_name"])})
        elif right_reg and not left_reg:
            candidates_4.append({"entity": str(row["left_name"]), "reg": str(row["right_name"])})

    picked_4 = _pick_one(candidates_4)
    if picked_4:
        q = f"What regulations apply to {picked_4['entity']}?"
        gt = picked_4["reg"]
        test_set.append({"question": q, "ground_truth": gt, "query_type": "Regulatory"})
    else:
        print("[warn] No valid relationships found for Type 4 (regulatory).")

    # Type 5 - Material composition (favors graph)
    rows_5 = [
        dict(r)
        for r in session.run(
            """
            MATCH (a)-[r]->(b)
            WHERE type(r) IN ['CONTAINS', 'MADE_FROM', 'USES']
              AND coalesce(a.name, '') <> ''
              AND coalesce(b.name, '') <> ''
            RETURN a.name AS left_name,
                   labels(a) AS left_labels,
                   b.name AS right_name,
                   labels(b) AS right_labels
            """
        )
    ]
    candidates_5: list[dict[str, str]] = []
    for row in rows_5:
        left_prod = "Product" in (row.get("left_labels") or [])
        right_prod = "Product" in (row.get("right_labels") or [])
        left_mat = "Material" in (row.get("left_labels") or [])
        right_mat = "Material" in (row.get("right_labels") or [])

        if left_prod and right_mat:
            candidates_5.append({"product": str(row["left_name"]), "material": str(row["right_name"])})
        elif right_prod and left_mat:
            candidates_5.append({"product": str(row["right_name"]), "material": str(row["left_name"])})

    picked_5 = _pick_one(candidates_5)
    if picked_5:
        q = f"What materials are used in or contained in {picked_5['product']}?"
        gt = picked_5["material"]
        test_set.append({"question": q, "ground_truth": gt, "query_type": "Material composition"})
    else:
        print("[warn] No valid relationships found for Type 5 (material composition).")

    return test_set


def run_pipeline(question: str, top_k: int = 5) -> dict[str, Any]:
    try:
        vector_results = vector_search(question, top_k=top_k)
        graph_result = graph_search(question)

        retrieved_contexts: list[str] = []
        for r in vector_results:
            name = str(r.get("name", "")).strip()
            label = str(r.get("label", "")).strip()
            text = str(r.get("retrieval_text", "")).strip()
            score = r.get("score", "")
            snippet = f"[{label}] {name} (score: {score}) {text}".strip()
            if snippet:
                retrieved_contexts.append(snippet)

        graph_context_str = f"Graph context: {json.dumps(graph_result, default=str)}"
        retrieved_contexts.append(graph_context_str)

        answer = _openrouter_answer(question, "\n\n".join(retrieved_contexts))

        return {
            "question": question,
            "answer": str(answer) if answer is not None else "",
            "retrieved_contexts": retrieved_contexts,
        }
    except Exception as e:  # noqa: BLE001
        return {
            "question": question,
            "answer": "",
            "retrieved_contexts": [],
            "error": str(e),
        }


def build_ragas_config() -> dict[str, Any]:
    llm = ChatOpenAI(
        model=OPENROUTER_MODEL,
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
        max_retries=2,
    )
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    ragas_llm = LangchainLLMWrapper(llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

    metrics = [faithfulness, answer_correctness, context_recall]
    for m in metrics:
        m.llm = ragas_llm
        m.embeddings = ragas_embeddings

    return {"metrics": metrics}


def run_evaluation(test_set: list[dict[str, str]], pipeline_outputs: list[dict[str, Any]], ragas_config: dict[str, Any]):
    rows: list[dict[str, Any]] = []
    for i, q in enumerate(test_set):
        out = pipeline_outputs[i]
        if "error" in out:
            print(f"[skip] Pipeline error for question: {q['question']}")
            print(f"       reason: {out.get('error', '')}")
            continue
        rows.append(
            {
                "question": q["question"],
                "answer": out.get("answer", ""),
                "retrieved_contexts": out.get("retrieved_contexts", []),
                "ground_truth": q["ground_truth"],
            }
        )

    if not rows:
        reasons = [str(o.get("error", "")) for o in pipeline_outputs[:5]]
        raise RuntimeError(
            "All pipeline runs failed. Nothing to evaluate. "
            f"Sample errors: {reasons}"
        )

    dataset = Dataset.from_list(rows)

    last_exc: Exception | None = None
    for attempt in range(1, 4):
        try:
            return evaluate(dataset=dataset, metrics=ragas_config["metrics"])
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < 3:
                print(f"[warn] RAGAS evaluate failed (attempt {attempt}/3): {exc}")
                time.sleep(10)
            else:
                break

    if last_exc:
        raise last_exc
    raise RuntimeError("RAGAS evaluation failed for unknown reason")


def _result_rows(result, test_set: list[dict[str, str]], pipeline_outputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    df = result.to_pandas()

    evaluated_meta = [
        test_set[i]
        for i, out in enumerate(pipeline_outputs)
        if "error" not in out
    ]

    rows: list[dict[str, Any]] = []
    for i, meta in enumerate(evaluated_meta):
        row = {
            "question": meta["question"],
            "query_type": meta["query_type"],
            "faithfulness": float(df.loc[i, "faithfulness"]),
            "answer_correctness": float(df.loc[i, "answer_correctness"]),
            "context_recall": float(df.loc[i, "context_recall"]),
        }
        rows.append(row)
    return rows


def print_report(result, test_set: list[dict[str, str]], pipeline_outputs: list[dict[str, Any]]) -> None:
    rows = _result_rows(result, test_set, pipeline_outputs)

    overall_f = sum(r["faithfulness"] for r in rows) / len(rows)
    overall_c = sum(r["answer_correctness"] for r in rows) / len(rows)
    overall_r = sum(r["context_recall"] for r in rows) / len(rows)

    print("=" * 80)
    print("BENCHMARK 6 - END-TO-END ANSWER QUALITY (RAGAS)")
    print("=" * 80)
    print()
    print("OVERALL SCORES  (range 0-1, higher is better)")
    print(f"  Faithfulness      : {overall_f:.2f}  (is the answer grounded in retrieved context?)")
    print(f"  Answer Correctness: {overall_c:.2f}  (does the answer match ground truth?)")
    print(f"  Context Recall    : {overall_r:.2f}  (does retrieved context cover the ground truth?)")
    print()

    print("PER QUESTION SCORES")
    for i, r in enumerate(rows, start=1):
        print(f"  Q{i} [{r['query_type']}]")
        print(f"     Q: \"{r['question']}\"")
        print(
            "     Faithfulness: "
            f"{r['faithfulness']:.2f} | Correctness: {r['answer_correctness']:.2f} | "
            f"Context Recall: {r['context_recall']:.2f}"
        )
        print()

    failed = sum(1 for out in pipeline_outputs if "error" in out)
    print("FAILED PIPELINE RUNS")
    print(f"  {failed} of {len(test_set)} questions errored during retrieval - see CSV for details")
    print()

    print("=" * 80)
    print("INTERPRETATION")
    print("  Faithfulness < 0.70 : Synthesis is hallucinating beyond retrieved context")
    print("  Correctness  < 0.50 : Retrieval is missing key facts from the graph")
    print("  Context Recall < 0.60 : Retrieved chunks don't cover ground truth facts")
    print()
    print("  Dominant failure pattern:")
    print("    Low faithfulness + high recall  -> synthesis problem")
    print("    High faithfulness + low recall  -> retrieval problem")
    print("    Low faithfulness + low recall   -> both layers failing")
    print("=" * 80)


def write_csv(result, test_set: list[dict[str, str]], pipeline_outputs: list[dict[str, Any]], output_path: str) -> None:
    score_rows = _result_rows(result, test_set, pipeline_outputs)
    score_by_question = {r["question"]: r for r in score_rows}

    merged: list[dict[str, Any]] = []
    for i, q in enumerate(test_set):
        out = pipeline_outputs[i]
        scores = score_by_question.get(q["question"])
        merged.append(
            {
                "question": q["question"],
                "query_type": q["query_type"],
                "ground_truth": q["ground_truth"],
                "answer": out.get("answer", ""),
                "faithfulness": "" if not scores else f"{scores['faithfulness']:.4f}",
                "answer_correctness": "" if not scores else f"{scores['answer_correctness']:.4f}",
                "context_recall": "" if not scores else f"{scores['context_recall']:.4f}",
                "error": out.get("error", ""),
                "_sort_correctness": -1.0 if not scores else float(scores["answer_correctness"]),
            }
        )

    merged.sort(key=lambda x: x["_sort_correctness"])

    fields = [
        "question",
        "query_type",
        "ground_truth",
        "answer",
        "faithfulness",
        "answer_correctness",
        "context_recall",
        "error",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in merged:
            writer.writerow({k: row.get(k, "") for k in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark 6: End-to-End RAGAS Evaluation")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-csv", default="benchmark6_results.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not NEO4J_PASSWORD:
        raise ValueError("NEO4J_PASSWORD is required in .env")
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY is required in .env")

    random.seed(args.seed)

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session(**session_kwargs()) as session:
        test_set = generate_test_set(session)
    driver.close()

    print(f"Generated {len(test_set)} test questions. Running pipeline...")
    pipeline_outputs = [run_pipeline(q["question"], top_k=args.top_k) for q in test_set]

    ragas_config = build_ragas_config()
    result = run_evaluation(test_set, pipeline_outputs, ragas_config)

    print_report(result, test_set, pipeline_outputs)
    write_csv(result, test_set, pipeline_outputs, args.output_csv)
    print(f"\nCSV written to: {args.output_csv}")


if __name__ == "__main__":
    main()
