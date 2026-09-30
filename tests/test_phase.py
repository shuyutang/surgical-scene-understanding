import numpy as np
import torch

from surgscene.phase import CausalMSTCN, causal_mode, macro_f1


def test_mstcn_is_causal():
    torch.manual_seed(0)
    net = CausalMSTCN(8, ch=16, layers=6, stages=2).eval()
    x = torch.randn(1, 8, 200)
    y = x.clone()
    y[:, :, 120:] = torch.randn(1, 8, 80)  # change only the future of t = 119
    with torch.no_grad():
        a, b = net(x)[-1], net(y)[-1]
    for u, v in zip(a, b):
        assert torch.allclose(u[..., :120], v[..., :120], atol=1e-6)
        assert not torch.allclose(u[..., 120:], v[..., 120:])


def test_causal_mode_uses_only_the_past():
    lab = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2, 0])
    out = causal_mode(lab, 3, 3)
    assert out[1] == 0 and out[4] == 1 and out[8] == 2
    assert (causal_mode(lab, 1, 3) == lab).all()


def test_macro_f1_ignores_absent_classes():
    y = np.array([0, 0, 1, 1])
    assert macro_f1(y, y, 5) == 1.0
    assert abs(macro_f1(y, np.array([0, 1, 1, 1]), 5) - np.mean([2 / 3, 0.8])) < 1e-9
