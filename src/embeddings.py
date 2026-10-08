"""Exercises 5–6: word vectors and pairwise sentence embeddings, no semantic search.

Student boundaries mark completed reference cores. Training/PCA infrastructure
is independent of Streamlit. Word2Vec trains locally; pretrained sentence setup is explicit and separate.
"""
import hashlib
from collections.abc import Sequence
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from src.preprocessing import lowercase_text, remove_punctuation, tokenize_words
from src.classical_nlp import cosine_similarity, _validate_documents

DEFAULT_WORD2VEC = {"vector_size": 50, "window": 5, "min_count": 1,
                    "sg": 0, "epochs": 100, "seed": 42}
# Explicit infrastructure settings, identical for CBOW / Skip-Gram experiments.
TRAINING_SETTINGS = {"workers": 1, "sample": 0.001, "negative": 5,
                     "hs": 0, "alpha": 0.025, "min_alpha": 0.0001,
                     "sorted_vocab": 1, "batch_words": 10000, "shrink_windows": True}


def prepare_tokenized_sentences(sentences: Sequence[str]) -> list[list[str]]:
    """Lowercase/punctuation/tokenization only; retain word forms and stopwords.

    Empty/punctuation-only sentences are omitted; empty input returns [].
    Input must be a sequence of raw sentence strings, not one raw string.
    """
    if isinstance(sentences, (str, bytes)) or not isinstance(sentences, Sequence):
        raise TypeError("Provide a list of sentence strings, not a single string.")
    if any(not isinstance(sentence, str) for sentence in sentences):
        raise TypeError("Each sentence must be a text string.")
    # STUDENT TODO 5.1 — Prepare Tokenized Sentences
    # Difficulty: ★ Guided | Student scaffold. Goal: reuse preprocessing.
    # Expected: Non-empty token lists. Hint: apply the three existing operations.
    # BEGIN STUDENT CORE 5.1
    # Implement only this educational core; surrounding setup stays provided.
    raise NotImplementedError('STUDENT TODO 5.1: implement this exercise in the marked core; see docs/STUDENT_GUIDE.md.')
    # END STUDENT CORE 5.1


def _stable_hash(text: str) -> int:
    """Starter initialization hash independent of Python's process hash seed."""
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "little")


def train_word2vec(tokenized_sentences, vector_size=DEFAULT_WORD2VEC["vector_size"],
                   window=DEFAULT_WORD2VEC["window"], min_count=DEFAULT_WORD2VEC["min_count"],
                   sg=DEFAULT_WORD2VEC["sg"], epochs=DEFAULT_WORD2VEC["epochs"], seed=DEFAULT_WORD2VEC["seed"]):
    """Train a CPU-only tiny model; sg=0 CBOW, sg=1 Skip-Gram.

    Fixed seed, one worker and stable initialization hash support reproducibility
    on the same software/platform. Cross-platform exact floats are not promised.
    Empty corpora, one-word vocabularies or min_count removing all terms fail clearly.
    """
    _validate_documents(tokenized_sentences)
    for name, value in {"vector_size": vector_size, "window": window, "min_count": min_count, "epochs": epochs}.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    if isinstance(sg, bool) or not isinstance(sg, int) or sg not in (0, 1):
        raise ValueError("sg must be 0 (CBOW) or 1 (Skip-Gram).")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be an integer from 0 to 2**32 - 1.")
    from collections import Counter
    counts = Counter(token for sentence in tokenized_sentences for token in sentence)
    if sum(count >= min_count for count in counts.values()) < 2:
        raise ValueError("Training needs at least two retained vocabulary words. Add sentences or lower min_count.")
    # STUDENT TODO 5.2 — Train a Tiny Word2Vec Model
    # Difficulty: ★★ Core | Student scaffold. Goal: configure Gensim, not its optimizer.
    # Expected: Trained Word2Vec model. Hint: pass sentence lists and architecture sg.
    # BEGIN STUDENT CORE 5.2
    # Implement only this educational core; surrounding setup stays provided.
    raise NotImplementedError('STUDENT TODO 5.2: implement this exercise in the marked core; see docs/STUDENT_GUIDE.md.')
    # END STUDENT CORE 5.2


def _require_word(model, word: str) -> None:
    if not isinstance(word, str) or not word:
        raise ValueError("Provide a non-empty vocabulary word.")
    if word not in model.wv:
        raise ValueError(f"Word {word!r} is outside this model's vocabulary. Select a known word; case is significant.")


def get_word_vector(model, word: str) -> list[float]:
    """Retrieve a copy as a list, suitable for existing manual cosine functions."""
    _require_word(model, word)
    # STUDENT TODO 5.3 — Retrieve a Word Vector
    # Difficulty: ★ Guided | Student scaffold. Goal: access keyed vectors.
    # Expected: vector_size finite floats. Hint: word indexing uses model.wv.
    # BEGIN STUDENT CORE 5.3
    # Implement only this educational core; surrounding setup stays provided.
    raise NotImplementedError('STUDENT TODO 5.3: implement this exercise in the marked core; see docs/STUDENT_GUIDE.md.')
    # END STUDENT CORE 5.3


def find_similar_words(model, word: str, top_n: int = 5) -> list[tuple[str, float]]:
    """Nearest learned vectors, excluding query; cap N at available neighbours."""
    _require_word(model, word)
    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 0:
        raise ValueError("top_n must be a non-negative integer.")
    count = min(top_n, len(model.wv) - 1)
    if count == 0:
        return []
    # STUDENT TODO 5.4 — Find Similar Words
    # Difficulty: ★★ Core | Student scaffold. Goal: query learned vector geometry.
    # Expected: (word, cosine) pairs. Hint: use the keyed-vector nearest-word method.
    # BEGIN STUDENT CORE 5.4
    # Implement only this educational core; surrounding setup stays provided.
    raise NotImplementedError('STUDENT TODO 5.4: implement this exercise in the marked core; see docs/STUDENT_GUIDE.md.')
    # END STUDENT CORE 5.4


def calculate_word_similarity(model, word_a: str, word_b: str) -> float:
    """Reuse Task 06 manual cosine over the retrieved word vectors."""
    vector_a = get_word_vector(model, word_a)
    vector_b = get_word_vector(model, word_b)
    # STUDENT TODO 5.5 — Calculate Word Similarity
    # Difficulty: ★★ Core | Student scaffold. Goal: connect embeddings to cosine.
    # Expected: Manual cosine, approximately Gensim's similarity.
    # Hint: the representation changed, not the comparison formula.
    # BEGIN STUDENT CORE 5.5
    # Implement only this educational core; surrounding setup stays provided.
    raise NotImplementedError('STUDENT TODO 5.5: implement this exercise in the marked core; see docs/STUDENT_GUIDE.md.')
    # END STUDENT CORE 5.5


def project_word_vectors(model, words: Sequence[str]) -> list[dict]:
    """Starter PCA only: rows word/x/y; not a faithful map of all vector distances.

    Require at least two unique words and two vector dimensions. Word order is
    retained; deterministic full-SVD PCA fits the selected subset only.
    """
    if isinstance(words, (str, bytes)) or not isinstance(words, Sequence):
        raise TypeError("Provide a sequence of selected words.")
    if len(words) < 2 or len(set(words)) != len(words):
        raise ValueError("PCA needs at least two distinct selected words.")
    if model.vector_size < 2:
        raise ValueError("PCA requires at least two vector dimensions.")
    vectors = [get_word_vector(model, word) for word in words]
    if all(vector == vectors[0] for vector in vectors):
        raise ValueError("Identical vectors have no variation for a PCA demonstration.")
    coordinates = PCA(n_components=2, svd_solver="full").fit_transform(vectors).tolist()
    return [{"word": word, "x": point[0], "y": point[1]} for word, point in zip(words, coordinates)]


def encode_sentences(model, sentences: Sequence[str]):
    """Encode a list of complete texts as a float array of shape (N, 384).

    No classical cleaning: retain original case/punctuation for the pretrained
    tokenizer. Empty list gives (0, 384); blank strings are rejected. A single
    sentence must be wrapped in a list. No extra L2 normalization is requested;
    the model's own modules are preserved. Cosine is calculated explicitly.
    """
    import numpy as np
    from src.sentence_resources import EMBEDDING_DIMENSION
    if isinstance(sentences, (str, bytes)) or not isinstance(sentences, Sequence):
        raise TypeError("Provide a list of sentences, including for one sentence.")
    if any(not isinstance(text, str) for text in sentences):
        raise TypeError("Every sentence must be a string.")
    if any(not text.strip() for text in sentences):
        raise ValueError("Blank sentences have no useful meaning; enter non-empty text.")
    if not sentences:
        return np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32)
    # STUDENT TODO 6.1 — Encode Sentences
    # Difficulty: ★★ Core | Student scaffold; model loading is starter code.
    # Goal: turn each complete text into one vector. Expected: N rows × 384 columns.
    # Hint: use the provided model's batch encoder, not Word2Vec or token lists.
    # BEGIN STUDENT CORE 6.1
    # Implement only this educational core; surrounding setup stays provided.
    raise NotImplementedError('STUDENT TODO 6.1: implement this exercise in the marked core; see docs/STUDENT_GUIDE.md.')
    # END STUDENT CORE 6.1
    if vectors.shape != (len(sentences), EMBEDDING_DIMENSION) or not np.isfinite(vectors).all():
        raise ValueError("The shared encoder must return finite N × 384 embeddings.")
    return vectors


def sentence_similarity(model, sentence_a: str, sentence_b: str) -> float:
    """Encode two complete texts, then reuse Exercise 4 manual cosine."""
    # STUDENT TODO 6.2 — Calculate Sentence Similarity
    # Difficulty: ★★ Core | Student scaffold. Goal: representation changes only.
    # Expected: cosine score, not a probability or factual correctness measure.
    # Hint: batch the two texts, take the two rows, reuse existing manual cosine.
    # BEGIN STUDENT CORE 6.2
    # Implement only this educational core; surrounding setup stays provided.
    raise NotImplementedError('STUDENT TODO 6.2: implement this exercise in the marked core; see docs/STUDENT_GUIDE.md.')
    # END STUDENT CORE 6.2


SIMILARITY_PAIRS = [
    ("Similar wording, similar meaning", "Students study natural language processing.",
     "Students learn natural language processing."),
    ("Different wording, similar meaning", "Natural language processing analyzes human language.",
     "Computers learn to work with human communication."),
    ("Shared words, reversed meaning", "The student teaches the instructor.",
     "The instructor teaches the student."),
    ("Unrelated subjects", "Natural language processing analyzes text.",
     "Computer networks route packets between devices."),
]


def pairwise_tfidf_similarity(sentence_a: str, sentence_b: str) -> float:
    """Starter pairwise baseline, without ranking or corpus semantic encoding.

    Fit classroom TF × ln(N/DF) on the eight preset texts plus distinct custom
    inputs. A two-document-only fit makes every shared term's IDF zero, so this
    small reference collection is deliberately used instead. It is NOT Task 06's
    fixed 20-course IDF. The UI/notebook explain this fitted collection.
    """
    from src.retrieval import tokenize_for_search
    from src.classical_nlp import build_vocabulary, calculate_df, calculate_idf, build_tfidf_matrix
    texts = list(dict.fromkeys([text for _, a, b in SIMILARITY_PAIRS for text in (a, b)]
                              + [sentence_a, sentence_b]))
    documents = [tokenize_for_search(text) for text in texts]
    vocabulary = build_vocabulary(documents)
    idf = calculate_idf(calculate_df(documents, vocabulary), len(documents))
    vectors = build_tfidf_matrix(documents, vocabulary, idf)
    return cosine_similarity(vectors[texts.index(sentence_a)], vectors[texts.index(sentence_b)])
