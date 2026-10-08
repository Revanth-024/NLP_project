"""Generated provided CSV infrastructure; no evaluator or algorithm solution."""
import csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

DEFAULT_QUERIES = ["natural language processing", "machine learning", "database systems",
                   "computer communication networks",
                   "Which subject teaches machines to work with human communication?"]

def load_course_documents():
    with (ROOT / "data/search_corpus/course_descriptions.csv").open(encoding="utf-8", newline="") as file:
        return [{"document_id": row["course_code"], "title": row["course_name"],
                 "text": row["description"], "semester": int(row["semester"])}
                for row in csv.DictReader(file)]
