import torch

from experiments.glunet.glunet_model_candidate import (
    GluNetCandidate, GatedCausalConv1d
)


def test_candidate_output_shapes():
    model = GluNetCandidate(input_features=3, lookback=16)
    x = torch.randn(4, 16, 3)
    logits = model(x)
    probs = model.predict_proba(x)

    assert logits.shape == (4, 256)
    assert probs.shape == (4, 256)
    assert torch.isfinite(logits).all()
    assert torch.allclose(
        probs.sum(dim=1), torch.ones(4), atol=1e-6
    )


def test_candidate_channel_layout():
    model = GluNetCandidate()
    assert model.input_projection.out_channels == 32
    assert [block.gated.filter_conv.out_channels
            for block in model.blocks] == [32, 32, 64, 64]
    assert [block.gated.filter_conv.dilation[0]
            for block in model.blocks] == [1, 2, 4, 8]


def test_gated_convolution_is_causal():
    torch.manual_seed(7)
    layer = GatedCausalConv1d(2, 4, dilation=2)
    x1 = torch.randn(1, 2, 10)
    x2 = x1.clone()
    x2[:, :, 7:] += 100.0

    y1 = layer(x1)
    y2 = layer(x2)

    # Changing inputs from time 7 onward cannot affect outputs before 7.
    assert torch.allclose(y1[:, :, :7], y2[:, :, :7], atol=1e-6)


def test_candidate_rejects_wrong_input_shape():
    model = GluNetCandidate()
    try:
        model(torch.randn(2, 15, 3))
    except ValueError:
        pass
    else:
        raise AssertionError("Expected wrong lookback to be rejected")
