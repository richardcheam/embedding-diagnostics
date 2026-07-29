import torch

from jepa_lens.models.vit import ViTEncoder


def make_encoder() -> ViTEncoder:
    return ViTEncoder(image_size=32, patch_size=4, embed_dim=32, depth=2, num_heads=4)


def test_num_patches_matches_grid():
    encoder = make_encoder()
    assert encoder.num_patches == 64


def test_forward_without_mask_returns_all_tokens():
    encoder = make_encoder()
    images = torch.randn(2, 3, 32, 32)
    tokens, pooled = encoder(images)
    assert tokens.shape == (2, 64, 32)
    assert pooled.shape == (2, 32)


def test_forward_with_mask_returns_only_kept_tokens():
    encoder = make_encoder()
    images = torch.randn(2, 3, 32, 32)
    keep_mask = torch.zeros(2, 64, dtype=torch.bool)
    keep_mask[:, :10] = True
    tokens, pooled = encoder(images)
    masked_tokens, masked_pooled = encoder(images, keep_mask)
    assert masked_tokens.shape == (2, 10, 32)
    assert masked_pooled.shape == (2, 32)
    assert not torch.allclose(pooled, masked_pooled)


def test_mask_must_keep_same_count_per_sample():
    encoder = make_encoder()
    images = torch.randn(2, 3, 32, 32)
    keep_mask = torch.zeros(2, 64, dtype=torch.bool)
    keep_mask[0, :10] = True
    keep_mask[1, :12] = True
    try:
        encoder(images, keep_mask)
    except ValueError as error:
        assert "same number" in str(error)
    else:
        raise AssertionError("expected ValueError for ragged mask")


def test_gradients_flow_to_parameters():
    encoder = make_encoder()
    images = torch.randn(2, 3, 32, 32)
    _, pooled = encoder(images)
    pooled.sum().backward()
    assert encoder.patch_embed.weight.grad is not None
