"""Exercises 8–9: transparent attention math and starter task inference.

The three bounded student cores remain Student scaffolds.
Model setup, validation and pretrained inference are provided infrastructure.
This module never imports Streamlit and never downloads a model.
"""
import numpy as np


def _vector(values, name):
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite one-dimensional numeric vector.")
    return array


def _matrix(values, name):
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite matrix with non-empty vector rows.")
    return array


def calculate_attention_scores(query, keys):
    """Return unscaled dot products Q·K_i. Empty keys give an empty vector.

    This is a simplified educational attention example, not learned BERT/GPT
    attention. Query and each key must have the same non-zero dimension.
    """
    query = _vector(query, "Query")
    if query.size == 0:
        raise ValueError("Query must have at least one dimension.")
    if len(keys) == 0:
        return np.empty(0, dtype=float)
    keys = _matrix(keys, "Keys")
    if keys.shape[1] != query.size:
        raise ValueError("Query and key dimensions must match.")
    # STUDENT TODO 8.1 — Calculate Attention Scores
    # Difficulty: ★★ Core | Student scaffold.
    # Goal: compare one query with each key. Formula: score_i = Q · K_i.
    # Expected: one score per key. Hint: calculate a dot product for each row.
    # BEGIN STUDENT CORE 8.1
    # Implement only this educational core; surrounding setup stays provided.
    raise NotImplementedError('STUDENT TODO 8.1: implement this exercise in the marked core; see docs/STUDENT_GUIDE.md.')
    # END STUDENT CORE 8.1
    if not np.isfinite(scores).all():
        raise ValueError("Dot products overflowed; use smaller finite vectors.")
    return scores


def softmax_attention_weights(scores):
    """Stable softmax over the whole score vector; empty input gives empty output.

    Subtracting max(scores) preserves ratios while preventing exp overflow.
    Extreme score gaps can underflow to zero in floating-point arithmetic;
    mathematical softmax weights are positive, computed ones are non-negative.
    """
    scores = _vector(scores, "Scores")
    if scores.size == 0:
        return np.empty(0, dtype=float)
    # STUDENT TODO 8.2 — Softmax Attention Weights
    # Difficulty: ★★ Core | Student scaffold.
    # Goal: normalize scores. Formula: exp(s_i) / sum(exp(s_j)).
    # Expected: weights summing to one. Hint: shift by the maximum first.
    # BEGIN STUDENT CORE 8.2
    # Implement only this educational core; surrounding setup stays provided.
    raise NotImplementedError('STUDENT TODO 8.2: implement this exercise in the marked core; see docs/STUDENT_GUIDE.md.')
    # END STUDENT CORE 8.2
    return weights


def calculate_weighted_context(weights, values):
    """Combine value rows using non-negative, unit-sum attention weights.

    Value dimensions may differ from query/key dimensions. An empty collection
    has no defined context dimension and is rejected. Input is never mutated.
    """
    weights = _vector(weights, "Weights")
    if weights.size == 0:
        raise ValueError("Context needs at least one weight and value row.")
    values = _matrix(values, "Values")
    if values.shape[0] != weights.size:
        raise ValueError("Each weight must correspond to one value row.")
    if np.any(weights < 0) or not np.isclose(weights.sum(), 1.0, atol=1e-8, rtol=1e-7):
        raise ValueError("Attention weights must be non-negative and sum to one.")
    # STUDENT TODO 8.3 — Weighted Context Representation
    # Difficulty: ★★ Core | Student scaffold.
    # Goal: combine values. Formula: context = sum(weight_i × V_i).
    # Expected: one vector with the value dimension. Hint: add weighted rows.
    # BEGIN STUDENT CORE 8.3
    # Implement only this educational core; surrounding setup stays provided.
    raise NotImplementedError('STUDENT TODO 8.3: implement this exercise in the marked core; see docs/STUDENT_GUIDE.md.')
    # END STUDENT CORE 8.3
    if not np.isfinite(context).all():
        raise ValueError("Weighted sum overflowed; use smaller values.")
    return context


def attention_example(query, keys, values, *, scaled=False):
    """Starter composition exposing every stage; optional scaling is not a TODO."""
    scores = calculate_attention_scores(query, keys)
    effective_scores = scores / np.sqrt(len(query)) if scaled else scores
    weights = softmax_attention_weights(effective_scores)
    context = calculate_weighted_context(weights, values)
    return {"scores": scores, "effective_scores": effective_scores, "weights": weights,
            "weighted_values": weights[:, None] * np.asarray(values, dtype=float), "context": context}


def _text(text, name):
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{name} must be non-empty text.")


def _check_length(pipe, text, name):
    # Reject excessive input rather than silently discard text or create chunks.
    count = len(pipe.tokenizer(text, add_special_tokens=False, truncation=False, verbose=False)["input_ids"])
    if count > 400:
        raise ValueError(f"{name} has {count} tokens. Use at most 400 for this short-input lab.")


def analyze_sentiment(pipe, text):
    """Starter inference: task label/score, not a factual confidence measure."""
    _text(text, "Sentiment input")
    _check_length(pipe, text, "Sentiment input")
    result = pipe(text, truncation=False)[0]
    return {"label": result["label"], "score": float(result["score"])}


def recognize_entities(pipe, text):
    """Return aggregated entities with exact original text and character spans."""
    _text(text, "NER input")
    _check_length(pipe, text, "NER input")
    return [{"text": text[int(row["start"]):int(row["end"])],
             "label": row["entity_group"], "score": float(row["score"]),
             "start": int(row["start"]), "end": int(row["end"])} for row in pipe(text)]


def answer_from_context(pipe, question, context):
    """Extract one span from supplied short context. No retrieval or generation.

    The SQuAD-v1 model is not trained to reliably abstain on unsupported questions;
    a returned span may be wrong. Scores do not prove factual correctness.
    """
    _text(question, "Question")
    _text(context, "Context")
    _check_length(pipe, question + " " + context, "Combined question/context")
    if len(pipe.tokenizer(question, add_special_tokens=False)["input_ids"]) > 64:
        raise ValueError("Use a question of at most 64 tokens in this lab.")
    result = pipe(question=question, context=context, max_seq_len=512, max_question_len=64,
                  doc_stride=0, top_k=1)
    return {"answer": result["answer"], "score": float(result["score"]),
            "start": int(result["start"]), "end": int(result["end"])}
