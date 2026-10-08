"""Exercise 6 structure, offline infrastructure and small real-model sanity checks.

Tests never download. Model tests skip with setup guidance on an unprepared
clone; instructor preflight must prepare the model and run without these skips.
"""
import math
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
import pytest
from src.embeddings import encode_sentences, sentence_similarity, pairwise_tfidf_similarity, SIMILARITY_PAIRS
from src.classical_nlp import cosine_similarity
from src.sentence_resources import (SENTENCE_MODEL_ID, EMBEDDING_DIMENSION,
                                    MODEL_CACHE, SENTENCE_MODEL_REVISION,
                                    load_sentence_embedding_model, SentenceModelUnavailable,
                                    _load_cached_model)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def encoder():
    if not list(MODEL_CACHE.glob("models--*/snapshots/*")):
        pytest.skip("Run python -m src.sentence_resources --download before integration tests.")
    # A present-but-broken model/dependency is a failure, not a hidden skip.
    return load_sentence_embedding_model()


def test_model_identity_cpu_eval(encoder):
    assert SENTENCE_MODEL_ID == "sentence-transformers/all-MiniLM-L6-v2"
    assert encoder.device.type == "cpu"
    assert not encoder.training
    assert encoder.get_sentence_embedding_dimension() == EMBEDDING_DIMENSION == 384
    assert encoder.max_seq_length == 256
    assert load_sentence_embedding_model() is encoder


@pytest.mark.parametrize("sentences", [["Students study NLP."], ["Students study NLP.", "Networks connect devices."]])
def test_real_encoding(encoder, sentences):
    vectors = encode_sentences(encoder, sentences)
    assert vectors.shape == (len(sentences), 384)
    assert np.issubdtype(vectors.dtype, np.floating)
    assert np.isfinite(vectors).all()


def test_empty_list_does_not_call_encoder():
    model = Mock()
    assert encode_sentences(model, []).shape == (0, 384)
    model.encode.assert_not_called()


@pytest.mark.parametrize("sentences, error", [("text", TypeError), ([7], TypeError), ([""], ValueError), (["  "], ValueError), (["valid", ""], ValueError)])
def test_invalid_input(sentences, error):
    with pytest.raises(error):
        encode_sentences(Mock(), sentences)




@pytest.mark.parametrize("vectors", [np.ones((1, 12)), np.full((1, 384), np.nan)])
def test_invalid_encoder_output(vectors):
    model = Mock()
    model.encode.return_value = vectors
    with pytest.raises(ValueError, match="finite"):
        encode_sentences(model, ["Sentence"])


def test_similarity_reuses_manual_cosine():
    model = Mock()
    vectors = np.zeros((2, 384), dtype=np.float32)
    vectors[0, 0] = 1
    vectors[1, :2] = [1, 1]
    model.encode.return_value = vectors
    with patch("src.embeddings.cosine_similarity", wraps=cosine_similarity) as manual:
        assert sentence_similarity(model, "A", "B") == pytest.approx(1/math.sqrt(2))
        manual.assert_called_once_with(vectors[0].tolist(), vectors[1].tolist())


def test_similarity_self_symmetry_and_repeat(encoder):
    a, b = "Students learn language processing.", "Computers analyze human language."
    assert sentence_similarity(encoder, a, a) == pytest.approx(1, abs=1e-6)
    forward = sentence_similarity(encoder, a, b)
    assert forward == pytest.approx(sentence_similarity(encoder, b, a), abs=1e-6)
    assert forward == pytest.approx(sentence_similarity(encoder, a, b), abs=1e-6)
    assert math.isfinite(forward)


def test_small_semantic_sanity(encoder):
    a = "Natural language processing helps computers understand human language."
    b = "Machines can analyze the meaning of human communication."
    c = "Database systems organize structured records."
    assert sentence_similarity(encoder, a, b) > sentence_similarity(encoder, a, c)


@pytest.mark.parametrize("pair", SIMILARITY_PAIRS)
def test_pairwise_baseline_and_semantic_values(encoder, pair):
    _, a, b = pair
    assert math.isfinite(pairwise_tfidf_similarity(a, b))
    assert math.isfinite(sentence_similarity(encoder, a, b))
    assert -1.000001 <= pairwise_tfidf_similarity(a, b) <= 1.000001


def test_missing_cache_graceful_no_network(tmp_path):
    # Separate empty cache; no real network call permitted during a normal load.
    with patch("requests.sessions.Session.request", side_effect=AssertionError("Network forbidden")):
        with pytest.raises(SentenceModelUnavailable, match="--download"):
            load_sentence_embedding_model(tmp_path / "empty-cache")


def test_loader_explicit_settings_and_download_policy(tmp_path):
    model = Mock()
    model.get_sentence_embedding_dimension.return_value = 384
    _load_cached_model.cache_clear()
    with patch("huggingface_hub.snapshot_download", return_value=str(tmp_path)) as snapshot, \
         patch("sentence_transformers.SentenceTransformer", return_value=model) as constructor:
        assert load_sentence_embedding_model(tmp_path) is model
        snapshot.assert_called_once_with(SENTENCE_MODEL_ID, cache_dir=str(tmp_path.resolve()), revision=SENTENCE_MODEL_REVISION, local_files_only=True, token=False)
        constructor.assert_called_once_with(str(tmp_path), device="cpu", local_files_only=True, trust_remote_code=False)
        model.eval.assert_called_once()
    _load_cached_model.cache_clear()


def test_explicit_download_only_when_requested(tmp_path):
    with patch("huggingface_hub.snapshot_download") as snapshot, \
         patch("src.sentence_resources._load_cached_model", return_value="prepared"):
        assert load_sentence_embedding_model(tmp_path, allow_download=True) == "prepared"
        assert snapshot.call_args.kwargs["token"] is False
        assert "*.safetensors" in snapshot.call_args.kwargs["allow_patterns"]


