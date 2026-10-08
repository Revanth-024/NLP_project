"""Exercise 2 behavior, coordinate alignment and meaningful edge cases."""
from pathlib import Path
import pytest
from sklearn.feature_extraction.text import CountVectorizer
from src.classical_nlp import (
    build_vocabulary, create_vocabulary_index, one_hot_encode, bag_of_words,
    build_document_term_matrix, generate_ngrams, countvectorizer_matrix,
)

@pytest.mark.parametrize("documents,expected", [
    ([["nlp", "language", "nlp"]], ["language", "nlp"]),
    ([["nlp", "text"], ["language", "nlp"]], ["language", "nlp", "text"]),
    ([], []), ([[], []], []), ([[], ["z", "a"]], ["a", "z"]),
])
def test_vocabulary(documents, expected):
    assert build_vocabulary(documents) == expected
    assert build_vocabulary(list(reversed(documents))) == expected

def test_one_hot_coordinate_order():
    vocabulary = ["text", "nlp", "language"]
    assert create_vocabulary_index(vocabulary) == {"text": 0, "nlp": 1, "language": 2}
    for word in vocabulary:
        vector = one_hot_encode(word, vocabulary)
        assert len(vector) == len(vocabulary)
        assert sum(vector) == 1
        assert vector[vocabulary.index(word)] == 1
        assert set(vector) <= {0, 1}

@pytest.mark.parametrize("vocabulary", [[], ["language", "text"]])
def test_unknown_word(vocabulary):
    with pytest.raises(ValueError, match="missing from vocabulary"):
        one_hot_encode("nlp", vocabulary)

def test_bow_counts_and_no_mutation():
    tokens = ["nlp", "language", "nlp"]
    assert bag_of_words(tokens, ["language", "nlp", "text"]) == [1, 2, 0]
    assert tokens == ["nlp", "language", "nlp"]
    assert bag_of_words([], ["nlp"]) == [0]
    assert bag_of_words([], []) == []

@pytest.mark.parametrize("function", [bag_of_words, countvectorizer_matrix])
def test_unknown_tokens_rejected(function):
    tokens = ["unknown"] if function is bag_of_words else [["unknown"]]
    with pytest.raises(ValueError, match="missing from vocabulary"):
        function(tokens, ["nlp"])

@pytest.mark.parametrize("function,args", [
    (create_vocabulary_index, ()), (one_hot_encode, ("nlp",)),
    (bag_of_words, ([],)), (build_document_term_matrix, ([],)),
    (countvectorizer_matrix, ([],)),
])
def test_duplicate_vocabulary_rejected(function, args):
    with pytest.raises(ValueError, match="unique"):
        function(*args, ["nlp", "nlp"])

@pytest.mark.parametrize("n,expected", [
    (1, [("natural",), ("language",), ("processing",)]),
    (2, [("natural", "language"), ("language", "processing")]),
    (3, [("natural", "language", "processing")]), (4, []),
])
def test_ngrams(n, expected):
    assert generate_ngrams(["natural", "language", "processing"], n) == expected
    assert generate_ngrams([], n) == []

@pytest.mark.parametrize("n", [0, -1, 1.5, "2", True, None])
def test_invalid_window(n):
    with pytest.raises(ValueError, match="positive integer"):
        generate_ngrams(["nlp"], n)

def test_repeated_windows():
    assert generate_ngrams(["a", "a", "a"], 2) == [("a", "a"), ("a", "a")]

def test_matrix_rows_and_columns():
    documents = [["nlp", "text", "nlp"], [], ["language"]]
    vocabulary = ["text", "language", "nlp"]
    matrix = build_document_term_matrix(documents, vocabulary)
    assert matrix == [[1, 0, 2], [0, 0, 0], [0, 1, 0]]
    assert [sum(row) for row in matrix] == [len(tokens) for tokens in documents]
    assert build_document_term_matrix([], vocabulary) == []
    assert build_document_term_matrix([[], []], []) == [[], []]

def test_library_comparison_with_case_single_characters_and_punctuation():
    documents = [["NLP", "x", "!", "x"], [], ["nlp", "é"]]
    vocabulary = ["x", "é", "nlp", "!", "NLP", "absent"]
    manual = build_document_term_matrix(documents, vocabulary)
    assert countvectorizer_matrix(documents, vocabulary) == manual
    vectorizer = CountVectorizer(analyzer=list, lowercase=False, token_pattern=None)
    library = vectorizer.fit_transform(documents).toarray()
    learned = vectorizer.get_feature_names_out().tolist()
    aligned = [[int(row[learned.index(term)]) if term in learned else 0 for term in vocabulary] for row in library]
    assert aligned == manual

@pytest.mark.parametrize("documents,vocabulary,expected", [([], ["x"], []), ([[], []], [], [[], []]), ([[]], ["x"], [[0]])])
def test_empty_library(documents, vocabulary, expected):
    assert countvectorizer_matrix(documents, vocabulary) == expected

@pytest.mark.parametrize("function,args", [(build_vocabulary, ("raw text",)), (build_vocabulary, (["raw text"],)), (bag_of_words, ("raw text", ["raw"])), (generate_ngrams, ([""], 2))])
def test_clear_invalid_input(function, args):
    with pytest.raises((TypeError, ValueError)):
        function(*args)

def test_order_limitation():
    documents = [["dog", "bites", "man"], ["man", "bites", "dog"]]
    vocabulary = build_vocabulary(documents)
    rows = build_document_term_matrix(documents, vocabulary)
    assert rows[0] == rows[1]
    assert generate_ngrams(documents[0], 2) != generate_ngrams(documents[1], 2)



# Exercises 3 and 4: hand-calculable weighting and vector mathematics.
import math
from src.classical_nlp import (
    calculate_tf, calculate_df, calculate_idf, calculate_tfidf, build_tfidf_matrix,
    build_query_tfidf_vector, cosine_similarity, sklearn_tfidf_comparison,
)

def test_tf_normalized_and_aligned():
    assert calculate_tf(["nlp", "studies", "language", "nlp"], ["language", "nlp", "studies"]) == [0.25, 0.5, 0.25]
    assert calculate_tf(["nlp"], ["absent", "nlp"]) == [0.0, 1.0]
    assert calculate_tf([], ["nlp"]) == [0.0]
    assert calculate_tf([], []) == []

def test_df_is_document_presence():
    documents = [["nlp", "nlp", "language"], ["nlp", "text"], ["language", "processing"]]
    vocabulary = ["text", "nlp", "processing", "language"]
    assert calculate_df(documents, vocabulary) == [1, 2, 1, 2]
    assert calculate_df([], vocabulary) == [0, 0, 0, 0]

def test_idf_natural_log_and_rarity():
    values = calculate_idf([3, 2, 1], 3)
    assert values == pytest.approx([0, math.log(1.5), math.log(3)])
    assert values[0] < values[1] < values[2]

@pytest.mark.parametrize("df,n", [([0], 3), ([-1], 3), ([4], 3), ([1.5], 3), ([True], 3), ([1], 0), ([1], -1), ([1], True), ([1], 2.5)])
def test_invalid_idf(df, n):
    with pytest.raises(ValueError):
        calculate_idf(df, n)

def test_tfidf_composition_matrix_and_common_zero():
    documents = [["common", "nlp", "nlp"], ["common", "text"], ["common", "text"]]
    vocabulary = ["text", "common", "nlp"]
    idf = calculate_idf(calculate_df(documents, vocabulary), 3)
    matrix = build_tfidf_matrix(documents, vocabulary, idf)
    assert matrix[0] == pytest.approx([0, 0, 2 / 3 * math.log(3)])
    assert matrix[1] == pytest.approx([0.5 * math.log(1.5), 0, 0])
    assert matrix[2] == pytest.approx(matrix[1])
    for document, row in zip(documents, matrix):
        assert row == pytest.approx([tf * rarity for tf, rarity in zip(calculate_tf(document, vocabulary), idf)])
    assert build_tfidf_matrix([], vocabulary, idf) == []
    assert calculate_tfidf([], vocabulary, idf) == [0, 0, 0]

@pytest.mark.parametrize("tokens,expected", [(["nlp"], [0, 2]), (["text", "nlp", "nlp"], [1/3, 4/3]), (["nlp", "robotics"], [0, 2]), (["robotics"], [0, 0]), ([], [0, 0])])
def test_query_shared_dimensions_idf_and_oov(tokens, expected):
    assert build_query_tfidf_vector(tokens, ["text", "nlp"], [1, 2]) == pytest.approx(expected)

@pytest.mark.parametrize("a,b,expected", [([1, 2], [1, 2], 1), ([1, 0], [0, 1], 0), ([1, 2], [3, 6], 1), ([0, 0], [1, 2], 0), ([], [], 0), ([1, 1], [1, 0], 1/math.sqrt(2))])
def test_cosine_math_and_symmetry(a, b, expected):
    assert cosine_similarity(a, b) == pytest.approx(expected)
    assert cosine_similarity(b, a) == pytest.approx(expected)

@pytest.mark.parametrize("a,b", [([1], [1, 2]), ([float('nan')], [1]), ([float('inf')], [1]), (["1"], [1])])
def test_invalid_vectors(a, b):
    with pytest.raises(ValueError):
        cosine_similarity(a, b)


def test_cosine_explanation_reuses_manual_calculation():
    details = cosine_similarity([1, 1], [1, 0], explain=True)
    assert details["dot_product"] == 1
    assert details["magnitude_a"] == pytest.approx(math.sqrt(2))
    assert details["magnitude_b"] == 1
    assert details["denominator"] == pytest.approx(math.sqrt(2))
    assert details["similarity"] == pytest.approx(cosine_similarity([1, 1], [1, 0]))

def test_sklearn_comparison_formula_is_explicitly_different():
    documents = [["common", "nlp", "nlp"], ["common", "text"], ["common", "text"]]
    vocabulary = ["common", "nlp", "text"]
    library = sklearn_tfidf_comparison(documents, vocabulary)
    expected_idf = [1, math.log(4/2) + 1, math.log(4/3) + 1]
    assert library["idf"] == pytest.approx(expected_idf)
    assert library["matrix"][0] == pytest.approx([1, 2 * expected_idf[1], 0])
    manual = build_tfidf_matrix(documents, vocabulary, calculate_idf([3, 1, 2], 3))
    assert manual[0][0] == 0 and library["matrix"][0][0] == 1


def test_library_query_scaling_preserves_cosine_scores():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

    documents = [["nlp", "nlp", "text"], ["data", "text"], ["network"]]
    vocabulary = build_vocabulary(documents)
    vectorizer = TfidfVectorizer(analyzer=list, vocabulary=create_vocabulary_index(vocabulary), lowercase=False, token_pattern=None, smooth_idf=True, norm=None)
    library_matrix = vectorizer.fit_transform(documents)
    query = ["nlp", "nlp", "text", "robotics"]
    raw_library_query = vectorizer.transform([query])
    expected = sklearn_cosine(raw_library_query, library_matrix)[0].tolist()
    normalized_query = build_query_tfidf_vector(query, vocabulary, vectorizer.idf_.tolist())
    actual = [cosine_similarity(normalized_query, row) for row in library_matrix.toarray().tolist()]
    assert actual == pytest.approx(expected)

