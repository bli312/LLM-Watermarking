import math

import torch

from watermark import GreenListLogitsProcessor, GreenListWatermark, WatermarkConfig


def test_determinism():
    watermark = GreenListWatermark(
        vocab_size=100, config=WatermarkConfig(secret_key=123)
    )

    first = watermark.greenlist_ids(prev_token_id=42)
    second = watermark.greenlist_ids(prev_token_id=42)

    assert torch.equal(first, second)
    assert len(first) == 25


def test_keychange():
    first = GreenListWatermark(vocab_size=100, config=WatermarkConfig(secret_key=1))
    second = GreenListWatermark(vocab_size=100, config=WatermarkConfig(secret_key=2))

    assert not torch.equal(
        first.greenlist_ids(prev_token_id=42),
        second.greenlist_ids(prev_token_id=42),
    )


def test_greenboost_only():
    watermark = GreenListWatermark(
        vocab_size=20,
        config=WatermarkConfig(gamma=0.25, delta=1.5, secret_key=123),
    )
    processor = GreenListLogitsProcessor(watermark)
    input_ids = torch.tensor([[7]])
    scores = torch.zeros((1, 20))

    updated_scores = processor(input_ids, scores.clone())
    green_ids = set(watermark.greenlist_ids(prev_token_id=7).tolist())

    for token_id in range(20):
        expected = 1.5 if token_id in green_ids else 0.0
        assert updated_scores[0, token_id].item() == expected


def test_detector_green():
    watermark = GreenListWatermark(
        vocab_size=100,
        config=WatermarkConfig(gamma=0.25, secret_key=123, z_threshold=4.0),
    )

    token_ids = [0]
    for _ in range(40):
        previous = token_ids[-1]
        token_ids.append(int(watermark.greenlist_ids(previous)[0]))

    result = watermark.score_token_ids(token_ids)

    assert result.total_tokens == 40
    assert result.green_tokens == 40
    assert result.green_fraction == 1.0
    assert result.z_score > 4.0
    assert result.is_watermarked
    assert math.isclose(result.expected_fraction, 0.25)


def test_detector_onetoken():
    watermark = GreenListWatermark(vocab_size=100)

    result = watermark.score_token_ids([1])

    assert result.total_tokens == 0
    assert result.green_tokens == 0
    assert result.z_score == 0.0
    assert not result.is_watermarked
