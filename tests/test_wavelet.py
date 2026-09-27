import pytest
import torch

from torch.nn import functional as F
from torch.testing import assert_close

from spd_learn.functional.wavelet import compute_gabor_wavelet
from spd_learn.modules import WaveletConv


@pytest.mark.parametrize("shape", [(2, 3, 61), (2, 2, 3, 61)])
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("complex_input", [False, True])
@pytest.mark.parametrize("padding,stride", [(0, 1), (2, 2), ("same", 1)])
@pytest.mark.parametrize("autocast_enabled", [False, True])
def test_wavelet_matches_complex_convolution(
    shape, dtype, complex_input, padding, stride, autocast_enabled
):
    """The real-input shortcut preserves outputs and all learnable gradients."""
    torch.manual_seed(19)
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    layer = WaveletConv(
        kernel_width_s=0.11,
        foi_init=torch.tensor([2.0, 3.0], dtype=real_dtype),
        sfreq=100,
        padding=padding,
        stride=stride,
        dtype=dtype,
    )
    # A strided time view also checks non-contiguous real and complex inputs.
    x = torch.randn(*shape, dtype=dtype if complex_input else real_dtype)[..., ::2]
    x.requires_grad_()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast_enabled):
        actual = layer(x)
    kernels = compute_gabor_wavelet(
        layer.tt, layer.foi, layer.fwhm, sfreq=100, dtype=dtype
    )
    expected = F.conv1d(
        x.to(dtype).reshape(-1, 1, x.shape[-1]),
        kernels.unsqueeze(1),
        padding=padding,
        stride=stride,
    )
    if len(shape) == 3:
        expected = expected.reshape(shape[0], shape[1], 2, -1).swapaxes(1, 2)
    else:
        expected = expected.reshape(*shape[:-1], 2, -1).permute(0, 3, 2, 1, 4)
        expected = expected.flatten(-2)
    tolerance = 1e-10 if dtype == torch.complex128 else 2e-5
    assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    weights = torch.randn_like(actual.real)
    actual_grads = torch.autograd.grad(
        (actual.abs().square() * weights).mean(), (x, layer.foi, layer.fwhm)
    )
    expected_grads = torch.autograd.grad(
        (expected.abs().square() * weights).mean(), (x, layer.foi, layer.fwhm)
    )
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        assert_close(actual_grad, expected_grad, atol=tolerance, rtol=tolerance)
