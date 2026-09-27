import pytest
import torch

from embedding_diagnostics.training.sigreg import sigreg_loss


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

    loss = sigreg_loss(embeddings, num_slices=8, num_freqs=5, generator=generator)

    assert loss.device.type == torch.device(device).type
    assert torch.isfinite(loss)


def test_gradient_does_not_vanish_as_collapse_deepens():
    """The restoring force must survive deep collapse.

    An earlier version of this module centered the embeddings before
    projecting. A collapsed encoder emits some constant vector c, and centering
    maps that exactly onto the origin, where the characteristic function is flat
    and its derivative is zero. The loss saturated while the gradient decayed to
    nothing, so collapse became a stationary point of the objective meant to
    prevent it. Measured then: gradient norm fell from 2.5e-4 to 2.5e-6 as the
    residual noise shrank by 100x.

    Without centering, a collapse to a non-zero constant still produces wildly
    non-Gaussian projections and a strong gradient. This pins that.
    """
    torch.manual_seed(0)
    constant = torch.randn(16) * 2.0
    noise = torch.randn(256, 16)

    def grad_norm(scale):
        embeddings = (constant + noise * scale).clone().requires_grad_(True)
        sigreg_loss(
            embeddings, num_slices=64, num_freqs=17, freq_max=3.0,
            generator=torch.Generator().manual_seed(0),
        ).backward()
        return embeddings.grad.norm().item()

    shallow = grad_norm(1e-2)
    deep = grad_norm(1e-4)

    # The force is essentially unchanged 100x deeper into the collapse.
    assert deep > shallow * 0.5, f"restoring force decayed: {shallow:.3e} -> {deep:.3e}"
    assert deep > 1e-2, f"restoring force implausibly weak: {deep:.3e}"


def test_matches_the_reference_implementation():
    """Bit-for-bit against a port of rbalestr-lab/lejepa.

    Guards the four things that differed when the from-summary version was
    checked against the reference: no centering, trapezoid weights including
    t=0 with half-weight endpoints, weights multiplied by phi and NOT
    normalized, and the batch-size scaling.
    """

    def reference(x, num_slices, t_max, n_points, gen):
        dim, count = x.size(-1), x.size(-2)
        directions = torch.randn(dim, num_slices, generator=gen)
        directions = directions / directions.norm(p=2, dim=0)
        projections = x @ directions
        t = torch.linspace(0, t_max, n_points)
        step = t_max / (n_points - 1)
        weights = torch.full((n_points,), 2 * step)
        weights[0] = step
        weights[-1] = step
        phi = torch.exp(-0.5 * t**2)
        weights = weights * phi
        angles = projections.unsqueeze(-1) * t
        err = (torch.cos(angles).mean(-3) - phi).square() + torch.sin(angles).mean(-3).square()
        return (err @ weights).mean() * count

    torch.manual_seed(0)
    cases = {
        "isotropic": torch.randn(256, 16),
        "anisotropic": torch.randn(256, 16) * torch.tensor([5.0] + [0.05] * 15),
        "collapsed": torch.randn(16).repeat(256, 1) + 1e-4 * torch.randn(256, 16),
        "shifted": torch.randn(256, 16) + 3.0,
    }
    for name, x in cases.items():
        expected = reference(x, 1024, 3.0, 17, torch.Generator().manual_seed(7))
        actual = sigreg_loss(
            x, num_slices=1024, num_freqs=17, freq_max=3.0,
            generator=torch.Generator().manual_seed(7),
        )
        assert torch.allclose(expected, actual, rtol=1e-6), name


def test_even_num_freqs_is_rejected():
    """The trapezoid rule needs an odd point count, as the reference asserts."""
    import pytest

    with pytest.raises(ValueError, match="must be odd"):
        sigreg_loss(torch.randn(32, 8), num_freqs=16)
