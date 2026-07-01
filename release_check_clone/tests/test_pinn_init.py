import sys
import pytest

sys.path.insert(0, 'aipv')


def torch_available():
    try:
        import torch
        return True
    except Exception:
        return False


@pytest.mark.skipif(not torch_available(), reason='torch not installed')
def test_pinn_forward_init():
    import torch
    from heart_valve_project.models.forwardmodel import ForwardMLP

    model = ForwardMLP(input_dim=2, output_dim=20, hidden_dim=32)
    x = torch.randn(4, 2)
    y = model(x)
    assert y.shape[0] == x.shape[0]
    assert y.shape[1] == 20
