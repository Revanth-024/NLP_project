"""Structural and mathematical Word2Vec tests without fragile semantic rankings."""
import json
import math
from pathlib import Path
import pytest
from src.embeddings import (
    DEFAULT_WORD2VEC, TRAINING_SETTINGS, prepare_tokenized_sentences, train_word2vec,
    get_word_vector, find_similar_words, calculate_word_similarity, project_word_vectors,
)
from src.preprocessing import lowercase_text, remove_punctuation, tokenize_words
from src.classical_nlp import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture(scope="module")
def sentences():
    data = json.loads((ROOT / "data/examples/word2vec_corpus.json").read_text(encoding="utf-8"))
    return prepare_tokenized_sentences([item["text"] for item in data["sentences"]])

@pytest.fixture(scope="module")
def model(sentences):
    return train_word2vec(sentences, vector_size=20, epochs=10)

def test_tokenized_structure_and_shared_preprocessing():
    raw = ["Natural Language Processing!", "", "???", "Students study DATA."]
    actual = prepare_tokenized_sentences(raw)
    assert actual == [["natural", "language", "processing"], ["students", "study", "data"]]
    assert actual[0] == tokenize_words(remove_punctuation(lowercase_text(raw[0])))
    assert prepare_tokenized_sentences([]) == []

@pytest.mark.parametrize("raw", ["raw sentence", [42], None])
def test_bad_sentence_input(raw):
    with pytest.raises(TypeError):
        prepare_tokenized_sentences(raw)

def test_corpus_provenance_and_workshop_vocabulary(sentences, model):
    data = json.loads((ROOT / "data/examples/word2vec_corpus.json").read_text(encoding="utf-8"))
    assert data["is_fictional"] is True
    assert data["disclaimer"] == "Synthetic educational data created for NLP workshop purposes."
    assert len(sentences) > 100
    assert all(sentence for sentence in sentences)
    for item in data["sentences"]:
        assert (ROOT / item["source"]).is_file()
        assert item["source"] in data["source_files"]
        assert not any(term in item["source"] for term in ["metadata", "benchmark", "university_facts"])
    assert {"language", "text", "nlp", "machine", "learning", "deep", "model", "data", "student", "course", "examination", "attendance", "network", "database"} <= set(model.wv.index_to_key)

@pytest.mark.parametrize("sg", [0, 1])
def test_modes_and_configured_dimensions(sentences, sg):
    model = train_word2vec(sentences[:20], vector_size=20, sg=sg, epochs=5)
    assert model.sg == sg
    assert model.vector_size == 20
    assert model.workers == 1
    assert model.seed == DEFAULT_WORD2VEC["seed"]
    assert TRAINING_SETTINGS["workers"] == 1
    assert len(model.wv) > 1

def test_repeat_training_with_fixed_seed():
    sentences = [["language", "models", "learn"], ["data", "models", "learn"]] * 10
    first = train_word2vec(sentences, vector_size=20, epochs=10)
    second = train_word2vec(sentences, vector_size=20, epochs=10)
    assert first.wv.index_to_key == second.wv.index_to_key
    assert get_word_vector(first, "language") == pytest.approx(get_word_vector(second, "language"), abs=1e-7)
    assert [word for word, _ in find_similar_words(first, "language", 3)] == [word for word, _ in find_similar_words(second, "language", 3)]

@pytest.mark.parametrize("sentences", [[], [[]], [["one", "one"]], ["raw sentence"]])
def test_invalid_training_corpus(sentences):
    with pytest.raises((TypeError, ValueError)):
        train_word2vec(sentences)

@pytest.mark.parametrize("parameters", [{"vector_size": 0}, {"window": -1}, {"min_count": 0}, {"epochs": 0}, {"sg": 2}, {"sg": True}, {"seed": -1}, {"seed": True}, {"min_count": 99}])
def test_invalid_parameters(parameters):
    with pytest.raises(ValueError):
        train_word2vec([["language", "text"]], **parameters)

def test_vector_shape_values_and_copy(model):
    vector = get_word_vector(model, "language")
    assert len(vector) == 20
    assert all(isinstance(value, float) and math.isfinite(value) for value in vector)
    original = vector[0]
    vector[0] = 999
    assert get_word_vector(model, "language")[0] == original

@pytest.mark.parametrize("operation", [lambda m: get_word_vector(m, "zzunseen"), lambda m: find_similar_words(m, "zzunseen"), lambda m: calculate_word_similarity(m, "language", "zzunseen")])
def test_oov_help(model, operation):
    with pytest.raises(ValueError, match="outside.*vocabulary"):
        operation(model)

def test_neighbours_structural_properties(model):
    neighbours = find_similar_words(model, "language", 5)
    assert len(neighbours) == 5
    assert len({word for word, _ in neighbours}) == 5
    assert all(word != "language" and word in model.wv for word, _ in neighbours)
    assert all(math.isfinite(score) and -1.000001 <= score <= 1.000001 for _, score in neighbours)
    scores = [score for _, score in neighbours]
    assert scores == sorted(scores, reverse=True)
    assert find_similar_words(model, "language", 0) == []
    assert len(find_similar_words(model, "language", 10000)) == len(model.wv) - 1

@pytest.mark.parametrize("top_n", [-1, 1.5, True])
def test_bad_top_n(model, top_n):
    with pytest.raises(ValueError):
        find_similar_words(model, "language", top_n)

def test_manual_gensim_cosine_symmetry_self(model):
    score = calculate_word_similarity(model, "language", "text")
    assert score == pytest.approx(float(model.wv.similarity("language", "text")), abs=1e-6)
    assert score == pytest.approx(cosine_similarity(get_word_vector(model, "language"), get_word_vector(model, "text")))
    assert score == pytest.approx(calculate_word_similarity(model, "text", "language"))
    assert calculate_word_similarity(model, "language", "language") == pytest.approx(1.0, abs=1e-6)

def test_pca_output_shape_and_labels(model):
    words = ["language", "text", "student", "course"]
    points = project_word_vectors(model, words)
    assert [point["word"] for point in points] == words
    assert all(set(point) == {"word", "x", "y"} for point in points)
    assert all(math.isfinite(point[coordinate]) for point in points for coordinate in ["x", "y"])
    assert len(project_word_vectors(model, words[:2])) == 2

@pytest.mark.parametrize("words", [[], ["language"], ["language", "language"], "language"])
def test_insufficient_pca_words(model, words):
    with pytest.raises((TypeError, ValueError)):
        project_word_vectors(model, words)

def test_pca_small_dimension():
    tiny = train_word2vec([["language", "text"]], vector_size=1, epochs=1)
    with pytest.raises(ValueError, match="dimensions"):
        project_word_vectors(tiny, ["language", "text"])

