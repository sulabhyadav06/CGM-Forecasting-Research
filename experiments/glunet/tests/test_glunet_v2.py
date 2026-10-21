import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from glunet_model_v2 import CausalConv1d, GluNetV2


def test_output_shape_and_probabilities():
    model = GluNetV2(input_features=3, lookback=16)
    x = torch.randn(4, 16, 3)
    assert model(x).shape == (4, 256)
    p = model.predict_proba(x)
    assert p.shape == (4, 256)
    assert torch.allclose(p.sum(-1), torch.ones(4), atol=1e-5)


def test_causal_convolution_no_future_leakage():
    torch.manual_seed(7)
    layer = CausalConv1d(2, 4, kernel_size=3, dilation=2)
    x1 = torch.randn(1, 2, 12)
    x2 = x1.clone()
    x2[:, :, 7:] += 1000
    assert torch.allclose(layer(x1)[:, :, :7], layer(x2)[:, :, :7], atol=1e-6)


def test_gradient_flow():
    model = GluNetV2(input_features=3, lookback=16)
    x = torch.randn(2, 16, 3)
    target = torch.tensor([12, 220], dtype=torch.long)
    torch.nn.CrossEntropyLoss()(model(x), target).backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert grads and all(g is not None for g in grads)
    assert any(torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)
