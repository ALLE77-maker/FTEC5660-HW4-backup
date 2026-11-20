"""Reproducible incremental ablation for FTEC5660-HW4 (public test).
No API key required for retrieval-based scores. Run: python ablation.py
For Task 4 end-to-end scoring, run the instructor's hw4.py with DEEPSEEK_API_KEY.
"""
import argparse
import csv
from pathlib import Path
import hw4

FLAGS = ("table_mode", "structure_chunk", "metadata_filter", "tokeniser",
         "metric_boost", "balanced_comparison", "abstention")
VARIANTS = [
    ("B0", "Naive BM25 over BeautifulSoup text", 0),
    ("V1", "Preserve table captions, rows and year columns", 1),
    ("V2", "Chunk by section and by table", 2),
    ("V3", "Filter by parsed company, year and document type", 3),
    ("V4", "Improved lexical tokenisation", 4),
    ("V5", "Boost metric labels and sections", 5),
    ("V6", "Balance evidence between compared companies", 6),
    ("V7", "Abstain on unavailable evidence", 7),
]

def evaluate(root, count):
    for i, name in enumerate(FLAGS):
        hw4.PIPELINE_FLAGS[name] = i < count
    hw4._KNOWN_COMPANIES.clear()
    corpus = hw4.load_corpus(root / "corpus")
    questions = hw4.load_json(root / "questions.json")
    gold = hw4.load_json(root / "ground_truth.json")
    chunks = []
    for file_name, html in corpus.items():
        chunks.extend(hw4.chunk_document(file_name, html))
    index = hw4.build_index(chunks)
    parsed = []
    hits = []
    kept = []
    rejected = []
    errors = []
    for qid, question in questions.items():
        g = gold[qid]
        prediction = hw4.parse_query(question)
        results = hw4.retrieve(index, question)
        ps = hw4.parse_score(prediction, g)
        parsed.append(ps)
        if g["type"] == "not_found":
            rejected.append(not results)
            if results:
                errors.append((qid, "unanswerable not rejected", question))
        else:
            kept.append(bool(results))
            hit = hw4.retrieval_hit(results, g)
            hits.append(hit)
            if not hit:
                errors.append((qid, "missed evidence", question))
    recall = sum(hits) / max(1, len(hits))
    k = sum(kept) / max(1, len(kept))
    r = sum(rejected) / max(1, len(rejected))
    return {
        "parse_accuracy": round(sum(parsed) / max(1, len(parsed)), 4),
        "recall_at_5": round(recall, 4),
        "answerable_kept": round(k, 4),
        "unanswerable_rejected": round(r, 4),
        "abstention_score": round(max(0, k+r-1), 4),
        "chunks": len(chunks),
        "errors": errors
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-folder", default="public_test")
    args = p.parse_args()
    root = Path(args.data_folder)
    if not (root / "ground_truth.json").exists():
        raise SystemExit("Missing public_test files: use the instructor corpus.")
    rows = []
    for name, desc, count in VARIANTS:
        metrics = evaluate(root, count)
        errors = metrics.pop("errors")
        rows.append({"variant": name, "change": desc, **metrics})
        print(f"{name}: recall@5={metrics['recall_at_5']:.1%}; "
              f"kept={metrics['answerable_kept']:.1%}; "
              f"rejected={metrics['unanswerable_rejected']:.1%}; "
              f"errors={len(errors)}")
        if name == "V7":
            with open("ablation_errors.csv", "w", newline="", encoding="utf8") as f:
                w = csv.writer(f)
                w.writerow(["question_id", "error_type", "question"])
                w.writerows(errors)
    with open("ablation_results.csv", "w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Saved ablation_results.csv and ablation_errors.csv.")
    print("End-to-end answer accuracy needs the unchanged DeepSeek generator.")
    print("Do not claim those metrics until running hw4.py with a valid key.")

if __name__ == "__main__":
    main()
