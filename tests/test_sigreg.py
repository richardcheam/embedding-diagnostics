import pytest
import torch

from jepa_lens.training.sigreg import sigreg_loss


def test_returns_finite_scalar():
    torch.manual_seed(0)
    loss = sigreg_loss(torch.randn(256, 32))
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_gaussian_embeddings_score_lower_than_collapsed():
    """The whole point: isotropic Gaussian is the target distribution."""
    torch.manual_seed(0)
    gaussian = torch.randn(512, 16)
    collapsed = torch.zeros(512, 16) + 1e-4 * torch.randn(512, 16)
    assert sigreg_loss(gaussian).item() < sigreg_loss(collapsed).item()


def test_gaussian_scores_lower_than_heavily_anisotropic():
    torch.manual_seed(0)
    gaussian = torch.randn(512, 16)
    anisotropic = torch.randn(512, 16)
    anisotropic[:, 1:] *= 0.01
    assert sigreg_loss(gaussian).item() < sigreg_loss(anisotropic).item()


def test_loss_is_non_negative():
    torch.manual_seed(0)
    for _ in range(5):
        assert sigreg_loss(torch.randn(128, 8)).item() >= 0.0


def test_gradients_flow_to_embeddings():
    embeddings = torch.randn(128, 16, requires_grad=True)
    sigreg_loss(embeddings).backward()
    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all()


def test_deterministic_under_same_generator_seed():
    embeddings = torch.randn(128, 16)
    first = sigreg_loss(embeddings, generator=torch.Generator().manual_seed(3))
    second = sigreg_loss(embeddings, generator=torch.Generator().manual_seed(3))
    assert torch.allclose(first, second)


def test_wrong_scale_is_penalized():
    """The target is the STANDARD isotropic Gaussian, so scale must matter.

    An earlier version standardized each projection before comparison, which
    made the loss blind to scale and to anisotropy — the property it exists to
    enforce. This pins that the blindness is gone.
    """
    torch.manual_seed(0)
    unit = torch.randn(512, 16)
    inflated = unit * 3.0
    shrunk = unit * 0.2
    assert sigreg_loss(unit).item() < sigreg_loss(inflated).item()
    assert sigreg_loss(unit).item() < sigreg_loss(shrunk).item()


@pytest.mark.skipif(
    not torch.backends.mps.is_available() and not torch.cuda.is_available(),
    reason="needs a non-CPU device to exercise the cross-device path",
)
def test_cpu_generator_works_with_accelerator_embeddings():
    """A CPU generator must not break training on GPU.

    `torch.randn` rejects a generator whose device differs from the target
    device. The strategy deliberately holds a CPU generator so slice directions
    stay reproducible across machines, so the draw has to happen on the
    generator's device and then move. Without that, every SIGReg condition
    would crash on the first step of a GPU run while passing every CPU test.
    """
    device = "mps" if torch.backends.mps.is_available() else "cuda"
    embeddings = torch.randn(64, 16, device=device)
    generator = torch.Generator().manual_seed(0)

    loss = sigreg_loss(embeddings, num_slices=8, num_freqs=4, generator=generator)

    assert loss.device.type == torch.device(device).type
    assert torch.isfinite(loss)


def test_gradient_vanishes_as_collapse_deepens():
    """SIGReg's restoring force weakens in proportion to the collapse.

    Near zero, cos(f*p) ~ 1 - (f*p)^2/2, so the characteristic-function
    deviation is O(p^2) and its derivative is O(p). The loss therefore
    saturates at a constant while the gradient decays linearly: the objective
    keeps reporting "this is very non-Gaussian" while supplying almost no force
    to fix it. A fully collapsed representation is close to a stationary point.

    Pinned because it explains the pilot, where sigreg_nostopgrad slid into
    collapse over the last quarter of training with the regularizer active and
    nominally dominating the loss. NOTE this may be a property of this
    from-summary implementation rather than of SIGReg as published -- see
    docs/open-questions.md.
    """
    torch.manual_seed(0)
    base = torch.randn(256, 16)

    def loss_and_grad(scale):
        embeddings = (base * scale).clone().requires_grad_(True)
        loss = sigreg_loss(
            embeddings, num_slices=64, num_freqs=16, generator=torch.Generator().manual_seed(0)
        )
        loss.backward()
        return loss.item(), embeddings.grad.norm().item()

    loss_big, grad_big = loss_and_grad(1e-2)
    loss_small, grad_small = loss_and_grad(1e-4)

    # The loss barely moves: it has saturated.
    assert abs(loss_small - loss_big) < 1e-3
    # The gradient falls by roughly the same factor as the scale, 100x here.
    assert grad_small < grad_big / 50
