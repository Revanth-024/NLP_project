"""Exercises 2–4 reference representations and vector math; preprocessing is separate.

Inputs are token lists or lists of token lists. Vectors are integer lists.
Supplied vocabulary order defines columns. Unknown terms raise ValueError;
duplicate vocabulary terms are rejected. Search queries deliberately ignore OOV.
"""
from collections.abc import Sequence
from src.preprocessing import _validate_tokens, build_vocabulary as unique_tokens


def _validate_documents(documents: Sequence[Sequence[str]]) -> None:
    if isinstance(documents, (str, bytes)) or not isinstance(documents, Sequence):
        raise TypeError("Expected a sequence of tokenized documents.")
    for tokens in documents:
        _validate_tokens(tokens)


def build_vocabulary(documents: Sequence[Sequence[str]]) -> list[str]:
    """Return sorted corpus vocabulary, reusing Exercise 1's unique-token utility."""
    _validate_documents(documents)
    # STUDENT TODO 2.1 — Multi-document Vocabulary
    # Difficulty: ★★ Core | Student scaffold.
    # Goal: Combine tokens. Expected: Sorted unique terms.
    # Hint: Flatten one level, not characters.
    # BEGIN STUDENT CORE 2.1
    # Implement only this educational core; surrounding setup stays provided.
    raise NotImplementedError('STUDENT TODO 2.1: implement this exercise in the marked core; see docs/STUDENT_GUIDE.md.')
    # END STUDENT CORE 2.1


def create_vocabulary_index(vocabulary: Sequence[str]) -> dict[str, int]:
    """Starter utility: preserve supplied order and reject ambiguous columns."""
    _validate_tokens(vocabulary)
    if len(set(vocabulary)) != len(vocabulary):
        raise ValueError("Vocabulary terms must be unique.")
    return {word: position for position, word in enumerate(vocabulary)}


def _require_known(tokens: Sequence[str], index: dict[str, int]) -> None:
    unknown = sorted(set(tokens) - index.keys())
    if unknown:
        raise ValueError(f"Terms missing from vocabulary: {unknown}. Build a shared vocabulary first.")


def one_hot_encode(word: str, vocabulary: Sequence[str]) -> list[int]:
    """Encode one known term; vector length is vocabulary size and sum is one."""
    _validate_tokens([word])
    index = create_vocabulary_index(vocabulary)
    _require_known([word], index)
    # STUDENT TODO 2.2 — One-Hot Encoding
    # Difficulty: ★★ Core | Student scaffold.
    # Goal: Activate this word's coordinate. Expected: Integer list.
    # Hint: Start with zeros, then use the index.
    # BEGIN STUDENT CORE 2.2
    # Implement only this educational core; surrounding setup stays provided.
    raise NotImplementedError('STUDENT TODO 2.2: implement this exercise in the marked core; see docs/STUDENT_GUIDE.md.')
    # END STUDENT CORE 2.2


def bag_of_words(tokens: Sequence[str], vocabulary: Sequence[str]) -> list[int]:
    """Count known tokens in supplied order; reject unknowns, preserve input."""
    _validate_tokens(tokens)
    index = create_vocabulary_index(vocabulary)
    _require_known(tokens, index)
    # STUDENT TODO 2.3 — Manual Bag of Words
    # Difficulty: ★★ Core | Student scaffold.
    # Goal: Count repetitions. Expected: Counts with zeros for absent terms.
    # Hint: Each token contributes one; do not rebuild the vocabulary.
    # BEGIN STUDENT CORE 2.3
    # Implement only this educational core; surrounding setup stays provided.
    raise NotImplementedError('STUDENT TODO 2.3: implement this exercise in the marked core; see docs/STUDENT_GUIDE.md.')
    # END STUDENT CORE 2.3


def build_document_term_matrix(documents: Sequence[Sequence[str]], vocabulary: Sequence[str]) -> list[list[int]]:
    """Starter assembly: reuse BoW; preserve document rows and vocabulary columns.

    Shape is (len(documents), len(vocabulary)). No documents gives []; an empty
    vocabulary with empty documents gives empty rows.
    """
    _validate_documents(documents)
    create_vocabulary_index(vocabulary)
    return [bag_of_words(tokens, vocabulary) for tokens in documents]


def generate_ngrams(tokens: Sequence[str], n: int) -> list[tuple[str, ...]]:
    """Contiguous tuple windows, with repetitions; oversized n returns []."""
    _validate_tokens(tokens)
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer (1, 2, 3, ...).")
    # STUDENT TODO 2.4 — Sliding-Window N-Grams
    # Difficulty: ★★ Core | Student scaffold.
    # Goal: Visit complete windows. Expected: Ordered tuples.
    # Hint: Last start is len(tokens) - n; keep documents separate.
    # BEGIN STUDENT CORE 2.4
    # Implement only this educational core; surrounding setup stays provided.
    raise NotImplementedError('STUDENT TODO 2.4: implement this exercise in the marked core; see docs/STUDENT_GUIDE.md.')
    # END STUDENT CORE 2.4


def countvectorizer_matrix(documents: Sequence[Sequence[str]], vocabulary: Sequence[str]) -> list[list[int]]:
    """Starter comparison after manual work, with identical tokens and columns.

    Callable analyzer=list consumes token lists, bypassing raw-text defaults,
    including sklearn's ngram_range. No additional case conversion is applied.
    Handle empty rows/columns explicitly; sklearn requires features to fit.
    """
    from sklearn.feature_extraction.text import CountVectorizer

    _validate_documents(documents)
    index = create_vocabulary_index(vocabulary)
    for tokens in documents:
        _require_known(tokens, index)
    if not documents or not vocabulary:
        return [[] for _ in documents]
    vectorizer = CountVectorizer(analyzer=list, vocabulary=index, lowercase=False, token_pattern=None)
    return vectorizer.fit_transform(documents).toarray().tolist()


def calculate_tf(tokens: Sequence[str], vocabulary: Sequence[str]) -> list[float]:
    """Normalized TF: count / document token count; empty document gives zeros."""
    counts = bag_of_words(tokens, vocabulary)
    # STUDENT TODO 3.1 — Term Frequency
    # Difficulty: ★★ Core | Reference complete. Hint: normalize existing counts.
    # BEGIN STUDENT CORE 3.1
    # Implement only this educational core; surrounding setup stays provided.
    raise NotImplementedError('STUDENT TODO 3.1: implement this exercise in the marked core; see docs/STUDENT_GUIDE.md.')
    # END STUDENT CORE 3.1


def calculate_df(documents: Sequence[Sequence[str]], vocabulary: Sequence[str]) -> list[int]:
    """Count documents containing each term, not total term occurrences."""
    counts = build_document_term_matrix(documents, vocabulary)
    # STUDENT TODO 3.2 — Document Frequency
    # Difficulty: ★★ Core | Reference complete. Hint: count positive rows per column.
    # BEGIN STUDENT CORE 3.2
    # Implement only this educational core; surrounding setup stays provided.
    raise NotImplementedError('STUDENT TODO 3.2: implement this exercise in the marked core; see docs/STUDENT_GUIDE.md.')
    # END STUDENT CORE 3.2


def calculate_idf(document_frequencies: Sequence[int], document_count: int) -> list[float]:
    """Natural-log IDF = ln(N / DF). Require N>0 and every DF in [1,N]."""
    import math
    if isinstance(document_count, bool) or not isinstance(document_count, int) or document_count <= 0:
        raise ValueError("Document count N must be a positive integer.")
    if isinstance(document_frequencies, (str, bytes)) or not isinstance(document_frequencies, Sequence):
        raise TypeError("Expected a sequence of document frequencies.")
    if any(isinstance(df, bool) or not isinstance(df, int) or not 1 <= df <= document_count for df in document_frequencies):
        raise ValueError("Each DF must be an integer between 1 and N; omit absent vocabulary terms.")
    # STUDENT TODO 3.3 — Inverse Document Frequency
    # Difficulty: ★★ Core | Reference complete. Hint: natural logarithm of N / DF.
    # BEGIN STUDENT CORE 3.3
    # Implement only this educational core; surrounding setup stays provided.
    raise NotImplementedError('STUDENT TODO 3.3: implement this exercise in the marked core; see docs/STUDENT_GUIDE.md.')
    # END STUDENT CORE 3.3


def _validate_vector(vector: Sequence[float], expected_length: int | None = None) -> None:
    """Starter validation for finite numeric vectors, outside student cores."""
    import math
    from numbers import Real
    if isinstance(vector, (str, bytes)) or not isinstance(vector, Sequence):
        raise TypeError("Expected a numeric vector sequence.")
    if expected_length is not None and len(vector) != expected_length:
        raise ValueError("Vector lengths must match the shared vocabulary dimensions.")
    if any(isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) for value in vector):
        raise ValueError("Vector values must be finite numbers.")


def calculate_tfidf(tokens: Sequence[str], vocabulary: Sequence[str], idf: Sequence[float]) -> list[float]:
    """Compose normalized TF × corpus IDF, keeping the supplied column order."""
    _validate_vector(idf, len(vocabulary))
    tf = calculate_tf(tokens, vocabulary)
    # STUDENT TODO 3.4 — TF-IDF
    # Difficulty: ★★ Core | Reference complete. Hint: multiply aligned coordinates.
    # BEGIN STUDENT CORE 3.4
    # Implement only this educational core; surrounding setup stays provided.
    raise NotImplementedError('STUDENT TODO 3.4: implement this exercise in the marked core; see docs/STUDENT_GUIDE.md.')
    # END STUDENT CORE 3.4


def build_tfidf_matrix(documents: Sequence[Sequence[str]], vocabulary: Sequence[str], idf: Sequence[float]) -> list[list[float]]:
    """Starter assembly reuses the individual TF-IDF operation."""
    _validate_documents(documents)
    create_vocabulary_index(vocabulary)
    _validate_vector(idf, len(vocabulary))
    return [calculate_tfidf(tokens, vocabulary, idf) for tokens in documents]


def build_query_tfidf_vector(tokens: Sequence[str], vocabulary: Sequence[str], idf: Sequence[float]) -> list[float]:
    """Ignore query OOV terms, reuse corpus columns/IDF, normalize by known tokens.

    Search queries intentionally tolerate unknowns; Exercise 2 BoW remains strict.
    A fully OOV/empty query yields a vocabulary-sized zero vector.
    """
    _validate_tokens(tokens)
    index = create_vocabulary_index(vocabulary)
    _validate_vector(idf, len(vocabulary))
    # STUDENT TODO 3.5 — Query TF-IDF Vector
    # Difficulty: ★★★ Challenge | Reference complete. Hint: filter, then reuse TF-IDF.
    # BEGIN STUDENT CORE 3.5
    # Implement only this educational core; surrounding setup stays provided.
    raise NotImplementedError('STUDENT TODO 3.5: implement this exercise in the marked core; see docs/STUDENT_GUIDE.md.')
    # END STUDENT CORE 3.5


def cosine_similarity(vector_a: Sequence[float], vector_b: Sequence[float], *, explain: bool = False) -> float | dict:
    """Manual cosine; zero vectors score 0.0. explain=True exposes the same math.

    The optional starter display mode avoids duplicating calculations in the UI.
    Normal calls return a float; display mode returns intermediate values too.
    """
    import math
    _validate_vector(vector_a)
    _validate_vector(vector_b, len(vector_a))
    # STUDENT TODO 4.1 — Cosine Similarity
    # Difficulty: ★★ Core | Reference complete. Hint: guard a zero denominator.
    # BEGIN STUDENT CORE 4.1
    # Implement only this educational core; surrounding setup stays provided.
    raise NotImplementedError('STUDENT TODO 4.1: implement this exercise in the marked core; see docs/STUDENT_GUIDE.md.')
    # END STUDENT CORE 4.1
    if explain:
        return {"dot_product": dot_product, "magnitude_a": magnitude_a,
                "magnitude_b": magnitude_b, "denominator": denominator, "similarity": similarity}
    return similarity


def sklearn_tfidf_comparison(documents: Sequence[Sequence[str]], vocabulary: Sequence[str]) -> dict:
    """Starter comparison: sklearn smoothed IDF and raw counts, no L2 normalization.

    sklearn IDF = ln((1+N)/(1+DF)) + 1; weighting uses counts, not normalized TF.
    Identical tokens and columns permit inspection, not raw numerical equality.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    _validate_documents(documents)
    index = create_vocabulary_index(vocabulary)
    for tokens in documents:
        _require_known(tokens, index)
    if not documents or not vocabulary:
        return {"idf": [], "matrix": [[] for _ in documents]}
    vectorizer = TfidfVectorizer(analyzer=list, vocabulary=index, lowercase=False, token_pattern=None, norm=None, smooth_idf=True, sublinear_tf=False)
    matrix = vectorizer.fit_transform(documents).toarray().tolist()
    return {"idf": vectorizer.idf_.tolist(), "matrix": matrix}
