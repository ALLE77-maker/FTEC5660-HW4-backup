"""Local public-set diagnostics; this script never calls DeepSeek.
Run with: python public_test_check.py
"""
import csv
import json
import math
import time
from pathlib import Path

from langchain_core.documents import Document
import hw4


def evaluate(folder=Path("public_test")):
    questions = hw4.load_json(folder / "questions.json")
    truth = hw4.load_json(folder / "ground_truth.json")
    tables = hw4.load_json(folder / "tables.json")
    docs = hw4.load_json(folder / "docs.json")
    corpus = hw4.load_corpus(folder / "corpus")
    if len(questions) != 60 or len(corpus) != 40:
        print("WARNING: unexpected public corpus size", len(questions), len(corpus))
    started = time.time()
    table_scores, table_errors = [], []
    for i, item in enumerate(tables, 1):
        try:
            score = hw4.table_score(hw4.table_to_text(item["html"]), item)
            table_scores.append(score)
            if score < 1:
                table_errors.append([i, item.get("source", ""), score])
        except Exception as exc:
            table_scores.append(0)
            table_errors.append([i, item.get("source", ""), repr(exc)])
    chunks, documents_ok, doc_errors = [], [], []
    for filename, html in corpus.items():
        try:
            parts = hw4.chunk_document(filename, html)
            reasons = hw4.document_problems(parts, None, docs[filename], filename)
            chunks.extend(parts)
            documents_ok.append(not reasons)
            if reasons:
                doc_errors.append([filename, "; ".join(reasons)])
        except Exception as exc:
            documents_ok.append(False)
            doc_errors.append([filename, repr(exc)])
    idx = hw4.build_index(chunks)
    rows = []
    for qid, question in questions.items():
        g = truth[qid]
        try:
            p = hw4.parse_query(question)
            ps = hw4.parse_score(p, g)
            hits = hw4.retrieve(idx, question)
            valid = (isinstance(hits, list) and len(hits) <= 5 and
                     all(isinstance(c, Document) for c in hits) and
                     all(c.page_content in {v.page_content for v in chunks} for c in hits))
            if not valid:
                hits = []
            recall = hw4.retrieval_hit(hits, g) if g["type"] != "not_found" else None
            keep = len(hits) > 0
            rows.append({
                "id": qid, "category": g["category"], "question": question,
                "expected_companies": json.dumps(g["query"]["companies"]),
                "actual_companies": json.dumps(p.get("companies", [])),
                "parse_accuracy": round(ps, 4),
                "answerable": g["type"] != "not_found",
                "kept": keep, "retrieval_hit": recall if recall is not None else "",
                "sources": " | ".join(c.metadata.get("source", "") for c in hits),
                "error": "",
            })
        except Exception as exc:
            rows.append({
                "id": qid, "category": g["category"], "question": question,
                "expected_companies": json.dumps(g["query"]["companies"]),
                "actual_companies": "", "parse_accuracy": 0, "answerable": g["type"] != "not_found",
                "kept": False, "retrieval_hit": False, "sources": "", "error": repr(exc),
            })
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    kept = sum(r["kept"] for r in answerable) / len(answerable)
    rejected = sum(not r["kept"] for r in unanswerable) / len(unanswerable)
    recall = sum(r["retrieval_hit"] is True for r in answerable) / len(answerable)
    valid_nonempty = sum(r["kept"] for r in answerable) / len(answerable)
    parsing = sum(r["parse_accuracy"] for r in rows) / len(rows)
    doc_pass = sum(documents_ok) / len(documents_ok)
    table_avg = sum(table_scores) / len(table_scores)
    points = {
        "1.1": 8*table_avg, "1.2": 7*doc_pass,
        "2.1": 5*valid_nonempty,
        "2.2": 10*parsing, "2.3": 15*max(0,min(1,(recall-.5)/.4)),
        "3": 10*max(0,kept+rejected-1)
    }
    summary = {
        "tables": len(tables), "table_mean": table_avg,
        "documents": len(corpus), "document_pass": doc_pass,
        "questions": len(rows), "query_parse": parsing,
        "answerable": len(answerable), "unanswerable": len(unanswerable),
        "valid_nonempty": valid_nonempty, "recall_at_5": recall,
        "answerable_kept": kept, "unanswerable_rejected": rejected,
        "points_1_to_3": points, "total_1_to_3": sum(points.values()),
        "duration_seconds": round(time.time()-started, 2),
        "table_errors": table_errors, "document_errors": doc_errors,
        "parse_misses": [r["id"] for r in rows if r["parse_accuracy"] < 1],
        "recall_misses": [r["id"] for r in answerable if not r["retrieval_hit"]],
        "abstention_misses": [r["id"] for r in unanswerable if r["kept"]],
    }
    with open("public_test_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open("retrieval_public_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("PUBLIC TEST RESULTS (no DeepSeek generator):")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    evaluate()
