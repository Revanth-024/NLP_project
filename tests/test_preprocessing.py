"""Exercise 1 behaviour, resource handling and student-boundary checks."""
import ast
import re
from pathlib import Path

import nltk
import pytest

from src import nltk_resources
from src.nltk_resources import NLTKResourceError
from src.preprocessing import (
    build_vocabulary,
    calculate_word_frequencies,
    lemmatize_words,
    lowercase_text,
    remove_punctuation,
    remove_stopwords,
    stem_words,
    tag_parts_of_speech,
    tokenize_words,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("text,expected", [
    ("", ""), ("NLP is Amazing!", "nlp is amazing!"),
    ("  Language\nLANGUAGE 123 ", "  language\nlanguage 123 "),
])
def test_lowercase_only_changes_case(text, expected):
    assert lowercase_text(text) == expected


@pytest.mark.parametrize("text,expected", [
    ("", ""), ("NLP!", "NLP "), ("NLP!!!??", "NLP     "),
    ("human-language", "human language"),
    ("“Study”—language.", " Study  language "),
    ("  NLP 123\ntext", "  NLP 123\ntext"),
])
def test_punctuation_is_visible_and_does_not_merge_words(text, expected):
    assert remove_punctuation(text) == expected


@pytest.mark.parametrize("text,expected", [
    ("", []), ("  \n ", []),
    ("natural language processing is useful", ["natural", "language", "processing", "is", "useful"]),
    ("NLP NLP language", ["NLP", "NLP", "language"]),
])
def test_tokenization_preserves_order_and_repetitions(text, expected):
    assert tokenize_words(text) == expected


def test_preserved_punctuation_remains_visible_as_tokens():
    tokens = tokenize_words("Language!")
    assert tokens[0] == "Language"
    assert "!" in tokens


def test_stopword_filtering_preserves_input_order_and_case():
    original = ["Natural", "language", "is", "useful", "AND", "language"]
    assert remove_stopwords(original, {"is", "and"}) == ["Natural", "language", "useful", "language"]
    assert original == ["Natural", "language", "is", "useful", "AND", "language"]


def test_default_english_stopwords_and_negation_choice():
    assert remove_stopwords(["natural", "language", "processing", "is", "useful"]) == ["natural", "language", "processing", "useful"]
    assert remove_stopwords(["not", "useful"]) == ["useful"]
    assert remove_stopwords(["not", "useful"], set()) == ["not", "useful"]


def test_porter_stems_can_be_non_dictionary_forms():
    assert stem_words(["studies", "processing", "studies"]) == ["studi", "process", "studi"]


def test_noun_lemmatization_and_unchanged_words():
    assert lemmatize_words(["students", "languages", "learning", "nlp"]) == ["student", "language", "learning", "nlp"]


def test_verb_pos_changes_lemma_without_being_an_automatic_tagger():
    assert lemmatize_words(["learning", "studied"], pos="v") == ["learn", "study"]
    assert lemmatize_words(["learning", "studied"], pos="n") == ["learning", "studied"]


def test_invalid_wordnet_pos_is_actionable():
    with pytest.raises(ValueError, match="WordNet POS"):
        lemmatize_words(["learning"], pos="invalid")


def test_vocabulary_is_unique_sorted_and_accepts_flattened_corpus_tokens():
    tokens = ["nlp", "processes", "language", "nlp", "understands", "text"]
    assert build_vocabulary(tokens) == ["language", "nlp", "processes", "text", "understands"]
    assert build_vocabulary(list(reversed(tokens))) == build_vocabulary(tokens)
    assert build_vocabulary(tokens + tokens) == build_vocabulary(tokens)


def test_frequency_counts_include_repeated_words_and_have_sorted_keys():
    tokens = ["nlp", "is", "useful", "and", "nlp", "is", "interesting"]
    counts = calculate_word_frequencies(tokens)
    assert counts == {"nlp": 2, "is": 2, "useful": 1, "and": 1, "interesting": 1}
    assert list(counts) == ["and", "interesting", "is", "nlp", "useful"]
    assert sum(counts.values()) == len(tokens)


def test_vocabulary_and_counts_do_not_silently_normalize_case():
    assert build_vocabulary(["NLP", "nlp"]) == ["NLP", "nlp"]
    assert calculate_word_frequencies(["NLP", "nlp", "NLP"]) == {"NLP": 2, "nlp": 1}


@pytest.mark.parametrize("function,expected", [
    (remove_stopwords, []), (stem_words, []), (lemmatize_words, []),
    (build_vocabulary, []), (calculate_word_frequencies, {}),
    (tag_parts_of_speech, []),
])
def test_empty_token_inputs_are_valid_without_resources(monkeypatch, function, expected):
    monkeypatch.setattr(nltk_resources, "resource_available", lambda _: False)
    assert function([]) == expected


@pytest.mark.parametrize("function", [
    remove_stopwords, stem_words, lemmatize_words,
    build_vocabulary, calculate_word_frequencies, tag_parts_of_speech,
])
def test_token_functions_reject_accidental_raw_strings(function):
    with pytest.raises(TypeError, match="sequence of tokens"):
        function("nlp language")


@pytest.mark.parametrize("bad_tokens", [["nlp", ""], ["nlp", None], [" "]])
def test_invalid_token_values_have_a_useful_message(bad_tokens):
    with pytest.raises(ValueError, match="non-empty string"):
        build_vocabulary(bad_tokens)


def test_basic_visible_pipeline_and_parallel_morphology():
    raw = "NLP is useful! NLP studies languages."
    normalized = lowercase_text(raw)
    cleaned = remove_punctuation(normalized)
    tokens = tokenize_words(cleaned)
    filtered = remove_stopwords(tokens)
    stems = stem_words(filtered)
    lemmas = lemmatize_words(filtered)
    assert normalized == "nlp is useful! nlp studies languages."
    assert cleaned == "nlp is useful  nlp studies languages "
    assert tokens == ["nlp", "is", "useful", "nlp", "studies", "languages"]
    assert filtered == ["nlp", "useful", "nlp", "studies", "languages"]
    assert stems == ["nlp", "use", "nlp", "studi", "languag"]
    assert lemmas == ["nlp", "useful", "nlp", "study", "language"]
    assert build_vocabulary(lemmas) == ["language", "nlp", "study", "useful"]
    assert calculate_word_frequencies(lemmas) == {"language": 1, "nlp": 2, "study": 1, "useful": 1}
    assert filtered == ["nlp", "useful", "nlp", "studies", "languages"]


def test_missing_resources_provide_setup_guidance(monkeypatch):
    monkeypatch.setattr(nltk_resources, "resource_available", lambda _: False)
    with pytest.raises(NLTKResourceError, match="python -m src.nltk_resources --download"):
        nltk_resources.require_resource("wordnet")
    with pytest.raises(NLTKResourceError, match="include-pos"):
        tag_parts_of_speech(["Students"])
    # Required-resource errors never prevent operations that need no data.
    assert tokenize_words(remove_punctuation(lowercase_text("NLP!"))) == ["nlp"]
    assert remove_stopwords(["is", "useful"], {"is"}) == ["useful"]
    assert stem_words(["studies"]) == ["studi"]


def test_missing_stopword_data_does_not_return_fake_results(monkeypatch):
    nltk_resources.english_stopwords.cache_clear()
    monkeypatch.setattr(nltk_resources, "resource_available", lambda _: False)
    try:
        with pytest.raises(NLTKResourceError, match="stopwords"):
            remove_stopwords(["the", "language"])
    finally:
        nltk_resources.english_stopwords.cache_clear()


def test_resource_setup_does_not_download_installed_resources(monkeypatch):
    monkeypatch.setattr(nltk_resources, "resource_available", lambda _: True)

    def unexpected_download(*args, **kwargs):
        pytest.fail("Already-installed resources must not be downloaded again.")

    monkeypatch.setattr(nltk, "download", unexpected_download)
    nltk_resources.setup_resources(include_pos=True)




def test_nlp_modules_remain_independent_of_streamlit():
    for path in (ROOT / "src").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(not alias.name.startswith("streamlit") for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("streamlit")
