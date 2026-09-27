import torch

from embedding_diagnostics.models.predictor import MLPPredictor


def test_output_shape_matches_target_count():
    predictor = MLPPredictor(embed_dim=32, hidden_dim=64, num_patches=64)
    pooled = torch.randn(4, 32)
    target_indices = torch.randint(0, 64, (4, 12))
    output = predictor(pooled, target_indices)
    assert output.shape == (4, 12, 32)


def test_different_positions_give_different_predictions():
    """Position conditioning must actually change the prediction."""
    torch.manual_seed(0)
    predictor = MLPPredictor(embed_dim=32, hidden_dim=64, num_patches=64)
    pooled = torch.randn(1, 32)
    first = predictor(pooled, torch.tensor([[0]]))
    second = predictor(pooled, torch.tensor([[63]]))
    assert not torch.allclose(first, second)


def test_gradients_flow():
    predictor = MLPPredictor(embed_dim=16, hidden_dim=32, num_patches=64)
    pooled = torch.randn(2, 16, requires_grad=True)
    output = predictor(pooled, torch.randint(0, 64, (2, 5)))
    output.sum().backward()
    assert pooled.grad is not None
