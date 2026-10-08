"""Exercise 1 reference implementation, reusable without Streamlit.

STUDENT TODO markers identify completed educational cores that a
later starter checkpoint will remove. Validation and resource handling stay.
Operations remain separate. Stem and lemma are parallel alternatives applied
to the same tokens, never a stem-then-lemmatize chain.
"""
import string
import unicodedata
from collections.abc import Collection, Sequence

from nltk import pos_tag
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import wordpunct_tokenize

from src.nltk_resources import NLTKResourceError, POS_RESOURCE, english_stopwords, require_resource


def _validate_text(text: str) -> None:
    """Starter validation is outside student boundaries."""
    if not isinstance(text, str):
        raise TypeError("Expected a text string.")


def _validate_tokens(tokens: Sequence[str]) -> None:
    """Reject accidental strings instead of treating them as characters."""
    if isinstance(tokens, (str, bytes)) or not isinstance(tokens, Sequence):
        raise TypeError("Expected a sequence of tokens, such as a list of strings.")
    if any(not isinstance(token, str) or not token.strip() for token in tokens):
        raise ValueError("Every token must be a non-empty string.")


def _is_punctuation(character: str) -> bool:
    """Recognize ASCII punctuation plus Unicode punctuation such as an em dash."""
    return character in string.punctuation or unicodedata.category(character).startswith("P")


def lowercase_text(text: str) -> str:
    """Lowercase only; preserve punctuation, digits and whitespace.

    Example: 'NLP is Useful!' becomes 'nlp is useful!'.
    """
    _validate_text(text)
    # ============================================================
    # STUDENT TODO 1.1 — Lowercase Normalization
    # Difficulty: ★ Guided | Student scaffold.
    # Goal: Make capitalization consistent without other cleaning.
    # Expected: A string of the same text with lowercase letters.
    # Hint: Strings provide a built-in case conversion method.
    # ============================================================
    # BEGIN STUDENT CORE 1.1
    # Implement only this educational core; surrounding setup stays provided.
    raise NotImplementedError('STUDENT TODO 1.1: implement this exercise in the marked core; see docs/STUDENT_GUIDE.md.')
    # END STUDENT CORE 1.1


def remove_punctuation(text: str) -> str:
    """Replace punctuation with spaces, avoiding accidental word merging.

    Example: 'human-language!' becomes 'human language '.
    Case, digits and existing whitespace remain unchanged. Apostrophes and
    decimal points also split: this simple policy is deliberately visible.
    """
    _validate_text(text)
    # ============================================================
    # STUDENT TODO 1.2 — Punctuation Handling
    # Difficulty: ★ Guided | Student scaffold.
    # Goal: Replace punctuation, keeping neighbouring words separate.
    # Expected: A string; punctuation becomes spaces, not merged text.
    # Hint: Use the provided _is_punctuation helper on each character.
    # ============================================================
    # BEGIN STUDENT CORE 1.2
    # Implement only this educational core; surrounding setup stays provided.
    raise NotImplementedError('STUDENT TODO 1.2: implement this exercise in the marked core; see docs/STUDENT_GUIDE.md.')
    # END STUDENT CORE 1.2


def tokenize_words(text: str) -> list[str]:
    """Use NLTK WordPunct tokenization without requiring Punkt resources.

    Example: 'natural language processing' becomes three tokens.
    Preserved punctuation also becomes tokens; decimals/hyphens may split.
    This function does not lowercase or filter the input.
    """
    _validate_text(text)
    # ============================================================
    # STUDENT TODO 1.3 — Word Tokenization
    # Difficulty: ★ Guided | Student scaffold.
    # Goal: Apply the provided library tokenizer, not its internals.
    # Expected: An ordered list of tokens, retaining repetitions.
    # Hint: wordpunct_tokenize accepts a text string.
    # ============================================================
    # BEGIN STUDENT CORE 1.3
    # Implement only this educational core; surrounding setup stays provided.
    raise NotImplementedError('STUDENT TODO 1.3: implement this exercise in the marked core; see docs/STUDENT_GUIDE.md.')
    # END STUDENT CORE 1.3


def remove_stopwords(
    tokens: Sequence[str], stopword_set: Collection[str] | None = None
) -> list[str]:
    """Filter against a provided set or NLTK English stopwords.

    Membership is case-insensitive; retained spelling and order are unchanged.
    Stopword removal is task-dependent, and can remove meaningful negation.
    A custom collection works without downloaded stopword resources.
    """
    _validate_tokens(tokens)
    if not tokens:
        return []
    if stopword_set is None:
        stopword_set = english_stopwords()
    if isinstance(stopword_set, str) or any(not isinstance(w, str) for w in stopword_set):
        raise TypeError("Stopwords must be a collection of strings, not one string.")
    stopword_set = {word.lower() for word in stopword_set}
    # ============================================================
    # STUDENT TODO 1.4 — Stopword Removal
    # Difficulty: ★ Guided | Student scaffold.
    # Goal: Retain tokens absent from the supplied stopword set.
    # Expected: A new ordered list; the input list is not modified.
    # Hint: Filter by membership, comparing a lowercase token.
    # ============================================================
    # BEGIN STUDENT CORE 1.4
    # Implement only this educational core; surrounding setup stays provided.
    raise NotImplementedError('STUDENT TODO 1.4: implement this exercise in the marked core; see docs/STUDENT_GUIDE.md.')
    # END STUDENT CORE 1.4


def stem_words(tokens: Sequence[str]) -> list[str]:
    """Apply Porter stemming; stems need not be dictionary words.

    Example: 'studies' becomes 'studi'; 'processing' becomes 'process'.
    PorterStemmer's default behaviour lowercases each token.
    """
    _validate_tokens(tokens)
    stemmer = PorterStemmer()
    # ============================================================
    # STUDENT TODO 1.5 — Stemming
    # Difficulty: ★ Guided | Student scaffold.
    # Goal: Apply a standard stemmer to each token.
    # Expected: A list of stems; keep order and repeated tokens.
    # Hint: The provided stemmer has a stem method.
    # ============================================================
    # BEGIN STUDENT CORE 1.5
    # Implement only this educational core; surrounding setup stays provided.
    raise NotImplementedError('STUDENT TODO 1.5: implement this exercise in the marked core; see docs/STUDENT_GUIDE.md.')
    # END STUDENT CORE 1.5


def lemmatize_words(tokens: Sequence[str], pos: str = "n") -> list[str]:
    """Apply WordNet lemmatization with one assumed POS for all tokens.

    Noun ('n') is the default: 'students' becomes 'student', but 'learning'
    may remain unchanged. Verb ('v') changes 'learning' to 'learn'. This
    parameter demonstrates context; it is not automatic POS-aware tagging.
    Valid POS codes are n, v, a, r and s. Normalize case before using WordNet.
    """
    _validate_tokens(tokens)
    if pos not in {"n", "v", "a", "r", "s"}:
        raise ValueError("WordNet POS must be n, v, a, r or s.")
    if not tokens:
        return []
    require_resource("wordnet")
    lemmatizer = WordNetLemmatizer()
    # ============================================================
    # STUDENT TODO 1.6 — Lemmatization
    # Difficulty: ★ Guided | Student scaffold.
    # Goal: Apply the provided lemmatizer using the supplied POS.
    # Expected: A list of lemmas; some tokens may remain unchanged.
    # Hint: lemmatize accepts a token and a pos argument.
    # ============================================================
    try:
        # BEGIN STUDENT CORE 1.6
        # Implement only this educational core; surrounding setup stays provided.
        raise NotImplementedError('STUDENT TODO 1.6: implement this exercise in the marked core; see docs/STUDENT_GUIDE.md.')
        # END STUDENT CORE 1.6
    except LookupError as error:
        raise NLTKResourceError(
            "WordNet cannot be read. Repair or replace its .nltk_data corpus "
            "and run python -m src.nltk_resources --download."
        ) from error


def build_vocabulary(tokens: Sequence[str]) -> list[str]:
    """Return sorted unique tokens; callers flatten tokenized documents first.

    This explicit token-only input avoids guessing whether a string is a
    document or a token. Empty input returns an empty vocabulary.
    """
    _validate_tokens(tokens)
    # ============================================================
    # STUDENT TODO 1.7 — Vocabulary and Word Frequency
    # Difficulty: ★★ Core | Student scaffold.
    # Goal: Vocabulary is the unique token set of the chosen representation.
    # Expected: A sorted list, with each token appearing exactly once.
    # Hint: Deduplicate first, then choose deterministic ordering.
    # ============================================================
    # BEGIN STUDENT CORE 1.7 — vocabulary
    # Implement only this educational core; surrounding setup stays provided.
    raise NotImplementedError('STUDENT TODO 1.7 — vocabulary: implement this exercise in the marked core; see docs/STUDENT_GUIDE.md.')
    # END STUDENT CORE 1.7 — vocabulary


def calculate_word_frequencies(tokens: Sequence[str]) -> dict[str, int]:
    """Count occurrences with a small loop, returning alphabetically sorted keys.

    Repetitions contribute to counts, unlike vocabulary construction.
    The counts sum to the number of tokens. Empty input returns {}.
    """
    _validate_tokens(tokens)
    # ============================================================
    # STUDENT TODO PART 1.7 — Word Frequency (paired with vocabulary above)
    # Difficulty: ★★ Core | Student scaffold.
    # Goal: Count how often each token occurs, without removing repeats.
    # Formula: frequency(t) = number of tokens equal to t.
    # Expected: Token-to-count mapping; counts sum to len(tokens).
    # Hint: Initialize unseen tokens, then increment their count.
    # ============================================================
    # BEGIN STUDENT CORE 1.7 — frequency
    # Implement only this educational core; surrounding setup stays provided.
    raise NotImplementedError('STUDENT TODO 1.7 — frequency: implement this exercise in the marked core; see docs/STUDENT_GUIDE.md.')
    # END STUDENT CORE 1.7 — frequency


def tag_parts_of_speech(tokens: Sequence[str]) -> list[tuple[str, str]]:
    """Starter-provided optional POS demonstration; not a student TODO."""
    _validate_tokens(tokens)
    if not tokens:
        return []
    require_resource(POS_RESOURCE)
    try:
        return pos_tag(list(tokens), lang="eng")
    except LookupError as error:
        raise NLTKResourceError(
            "The optional English POS tagger cannot be read. Run "
            "python -m src.nltk_resources --download --include-pos after "
            "repairing or replacing the incomplete resource folder."
        ) from error
