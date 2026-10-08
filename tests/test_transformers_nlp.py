"""Hand-calculated attention, starter contracts and limited offline CPU integration."""
from pathlib import Path
from unittest.mock import Mock, patch
import math
import numpy as np
import pytest
from src.transformers_nlp import (calculate_attention_scores, softmax_attention_weights,
                                  calculate_weighted_context, attention_example,
                                  analyze_sentiment, recognize_entities, answer_from_context)
from src.transformer_resources import (MODELS, model_directory, load_transformer_pipeline,
                                       TransformerModelUnavailable)


def test_attention_scores_8_1():
    assert calculate_attention_scores([1, 0], [[1, 0], [0, 1], [.5, .5]]).tolist() == [1, 0, .5]


@pytest.mark.parametrize("query,keys", [([], [[1]]), ([1, 0], [[1]]), ([1], [[1], [2, 3]]),
                                       ([float('nan')], [[1]]), ([1], [[float('inf')]])])
def test_invalid_scores(query, keys):
    with pytest.raises(ValueError):
        calculate_attention_scores(query, keys)


def test_empty_keys_and_scores():
    assert calculate_attention_scores([1, 0], []).shape == (0,)
    assert softmax_attention_weights([]).shape == (0,)


@pytest.mark.parametrize("scores", [[1, 0, .5], [10000, 9999, 9998], [-10000, -10001, -10002]])
def test_stable_softmax_8_2(scores):
    weights = softmax_attention_weights(scores)
    assert np.isfinite(weights).all() and (weights > 0).all()
    assert weights.sum() == pytest.approx(1)
    assert weights[np.argmax(scores)] == max(weights)


def test_softmax_equal_scores_and_shift_invariance():
    assert softmax_attention_weights([0, 0, 0]) == pytest.approx([1/3]*3)
    assert softmax_attention_weights([1, 2, 3]) == pytest.approx(softmax_attention_weights([1001, 1002, 1003]))


@pytest.mark.parametrize("scores", [[math.nan], [math.inf], [[1, 2]]])
def test_softmax_invalid(scores):
    with pytest.raises(ValueError):
        softmax_attention_weights(scores)


def test_weighted_context_8_3_and_no_mutation():
    values = [[2, 0, 1], [0, 2, 3]]
    assert calculate_weighted_context([.25, .75], values) == pytest.approx([.5, 1.5, 2.5])
    assert values == [[2, 0, 1], [0, 2, 3]]


@pytest.mark.parametrize("weights,values", [([], []), ([1], [[1], [2]]), ([-1, 2], [[1], [2]]),
                                            ([.2, .2], [[1], [2]]), ([1], [[math.nan]]),
                                            ([math.inf], [[1]]), ([1], [[]])])
def test_context_validation(weights, values):
    with pytest.raises(ValueError):
        calculate_weighted_context(weights, values)


def test_end_to_end_hand_calculation():
    result = attention_example([1, 0], [[1, 0], [0, 1], [.5, .5]], [[2, 0], [0, 2], [1, 1]])
    assert result['scores'] == pytest.approx([1, 0, .5])
    assert result['weights'] == pytest.approx([.5064803911, .1863237232, .3071958857])
    assert result['context'] == pytest.approx([1.3201566679, .6798433321])
    assert result['weighted_values'].sum(axis=0) == pytest.approx(result['context'])


def test_scaled_starter_example():
    result = attention_example([1, 0], [[1, 0], [0, 1]], [[2, 0], [0, 2]], scaled=True)
    assert result['effective_scores'] == pytest.approx([1/math.sqrt(2), 0])


def fake_pipe(output):
    pipe = Mock(return_value=output)
    pipe.tokenizer.return_value = {'input_ids': [1, 2, 3]}
    return pipe


def test_sentiment_starter_structure():
    assert analyze_sentiment(fake_pipe([{'label': 'POSITIVE', 'score': np.float32(.8)}]), 'Good.') == {
        'label': 'POSITIVE', 'score': pytest.approx(.8)}


def test_ner_preserves_original_spans():
    pipe = fake_pipe([{'word': 'different reconstructed text', 'entity_group': 'PER', 'score': .8, 'start': 0, 'end': 4}])
    assert recognize_entities(pipe, 'Riya studies.') == [{'text': 'Riya', 'label': 'PER', 'score': .8, 'start': 0, 'end': 4}]
    assert recognize_entities(fake_pipe([]), 'No entity.') == []


def test_qa_starter_no_retrieval():
    pipe = fake_pipe({'answer': 'Semester 6', 'score': .7, 'start': 3, 'end': 13})
    assert answer_from_context(pipe, 'When?', 'In Semester 6.')['answer'] == 'Semester 6'
    assert pipe.call_args.kwargs['doc_stride'] == 0


@pytest.mark.parametrize('text', ['', '   ', None])
def test_invalid_task_text(text):
    with pytest.raises(ValueError):
        analyze_sentiment(fake_pipe([]), text)
    with pytest.raises(ValueError):
        recognize_entities(fake_pipe([]), text)
    with pytest.raises(ValueError):
        answer_from_context(fake_pipe({}), 'Question?', text)


def test_long_task_input_rejected_without_inference():
    pipe = fake_pipe([])
    pipe.tokenizer.return_value = {'input_ids': [1]*401}
    with pytest.raises(ValueError, match='400'):
        analyze_sentiment(pipe, 'Long text')
    pipe.assert_not_called()


def test_missing_models_do_not_attempt_download(tmp_path):
    with patch('huggingface_hub.snapshot_download', side_effect=AssertionError('No download')):
        with pytest.raises(TransformerModelUnavailable, match='--download --task ner'):
            load_transformer_pipeline('ner', tmp_path)
    assert calculate_attention_scores([1], [[2]]).tolist() == [2]


def test_resource_identity_and_invalid_task():
    assert len(MODELS) == 3
    assert all(len(spec['revision']) == 40 and spec['license'] == 'Apache-2.0' for spec in MODELS.values())
    with pytest.raises(ValueError):
        load_transformer_pipeline('generation')


@pytest.fixture(scope='module', params=['sentiment', 'ner', 'qa'])
def offline_pipeline(request):
    if not (model_directory(request.param) / 'model.safetensors').exists():
        pytest.skip('Explicitly preload: python -m src.transformer_resources --download')
    with patch('requests.sessions.Session.request', side_effect=AssertionError('Network forbidden')):
        pipe = load_transformer_pipeline(request.param)
        assert pipe.device.type == 'cpu'
        assert not pipe.model.training
        assert load_transformer_pipeline(request.param) is pipe
        yield request.param, pipe


def test_limited_cpu_model_integration(offline_pipeline):
    task, pipe = offline_pipeline
    if task == 'sentiment':
        result = analyze_sentiment(pipe, 'The NLP workshop was engaging and easy to follow.')
        assert result['label'] == 'POSITIVE' and 0 <= result['score'] <= 1
    elif task == 'ner':
        text = 'Riya studies Natural Language Processing at Hindu College of Engineering.'
        entities = recognize_entities(pipe, text)
        assert entities
        for entity in entities:
            assert entity['text'] == text[entity['start']:entity['end']]
            assert entity['label'] and math.isfinite(entity['score'])
            assert 0 <= entity['start'] < entity['end'] <= len(text)
    else:
        context = 'Natural Language Processing is offered in Semester 6.'
        result = answer_from_context(pipe, 'In which semester is Natural Language Processing offered?', context)
        assert '6' in result['answer']
        assert result['answer'] == context[result['start']:result['end']]
        assert math.isfinite(result['score'])


