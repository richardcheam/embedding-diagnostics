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
