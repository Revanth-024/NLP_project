"""Classical lexical search: ranking, metadata, ties and safe zero queries."""
from copy import deepcopy
import pytest
from src.retrieval import prepare_search_corpus, search_tfidf, score_documents, rank_documents

def sample():
    return [{"document_id": "D1", "title": "Language", "text": "nlp language nlp"},
            {"document_id": "D2", "title": "Data", "text": "machine learning data"},
            {"document_id": "D3", "title": "Network", "text": "computer network devices"}]

def test_search_relevance_scores_metadata_and_no_mutation():
    documents = sample()
    original = deepcopy(documents)
    corpus = prepare_search_corpus(documents)
    result = search_tfidf("NLP!", corpus, 2)
    assert result["results"][0]["document_id"] == "D1"
    assert result["results"][0]["text"] == documents[0]["text"]
    assert result["results"][0]["title"] == "Language"
    assert len(result["results"]) == 2
    assert result["results"][0]["score"] > result["results"][1]["score"]
    assert result["tokens"] == ["nlp"]
    assert documents == original
    assert corpus["documents"] == original

@pytest.mark.parametrize("query", ["", "robotics unseen", "!!!"])
def test_zero_queries_no_false_match(query):
    corpus = prepare_search_corpus(sample())
    result = search_tfidf(query, corpus, 99)
    assert result["query_vector"] == [0.0] * len(corpus["vocabulary"])
    assert result["scores"] == [0.0] * 3
    assert [row["document_id"] for row in result["results"]] == ["D1", "D2", "D3"]

def test_mixed_unknown_query_ignored_not_new_dimension():
    corpus = prepare_search_corpus(sample())
    mixed = search_tfidf("nlp robotics", corpus)
    known = search_tfidf("nlp", corpus)
    assert mixed["oov_terms"] == ["robotics"]
    assert mixed["query_vector"] == known["query_vector"]

def test_rank_ties_and_top_k():
    documents = sample()
    ranked = rank_documents(documents, [0.5, 0.8, 0.8], 99)
    assert [row["document_id"] for row in ranked] == ["D2", "D3", "D1"]
    assert [row["score"] for row in ranked] == [0.8, 0.8, 0.5]
    assert rank_documents(documents, [0, 0, 0], 0) == []
    assert len(rank_documents(documents, [0, 0, 0], 1)) == 1

@pytest.mark.parametrize("top_k", [-1, 1.5, True])
def test_invalid_top_k(top_k):
    with pytest.raises(ValueError):
        rank_documents(sample(), [0, 0, 0], top_k)

def test_score_coordinates_and_dimensions():
    assert score_documents([1, 0], [[1, 0], [0, 1], [0, 0]]) == pytest.approx([1, 0, 0])
    with pytest.raises(ValueError):
        score_documents([1, 0], [[1]])
    with pytest.raises(ValueError):
        rank_documents(sample(), [1])

def test_empty_corpus_and_empty_document():
    assert search_tfidf("nlp", prepare_search_corpus([]))["results"] == []
    corpus = prepare_search_corpus([{"document_id": "empty", "text": ""}, {"document_id": "word", "text": "nlp"}])
    assert corpus["matrix"][0] == [0.0]
    assert search_tfidf("nlp", corpus)["results"][0]["document_id"] == "word"

def test_corpus_wide_query_has_zero_weight():
    corpus = prepare_search_corpus([{"document_id": "A", "text": "common nlp"}, {"document_id": "B", "text": "common data"}])
    result = search_tfidf("common", corpus)
    assert result["scores"] == [0.0, 0.0]

@pytest.mark.parametrize("documents", [[{"document_id": "A", "text": 4}], [{"document_id": "A", "text": "nlp"}, {"document_id": "A", "text": "text"}], [{"text": "nlp"}]])
def test_invalid_corpus_metadata(documents):
    with pytest.raises(ValueError):
        prepare_search_corpus(documents)


def test_course_description_corpus_search():
    import csv
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "data/search_corpus/course_descriptions.csv"
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    documents = [{"document_id": row["course_code"], "title": row["course_name"], "text": row["description"]} for row in rows]
    corpus = prepare_search_corpus(documents)
    result = search_tfidf("natural language processing", corpus)
    assert result["results"][0]["document_id"] == "HCE-CSE603"
    assert result["results"][0]["score"] > 0
    assert result["results"][0]["title"] == "Natural Language Processing"
