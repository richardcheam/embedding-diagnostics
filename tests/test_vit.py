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


def test_mask_gathers_each_sample_its_own_patches():
    """Per-sample masks must select per-sample patches.

    The other mask tests use the SAME kept-index pattern for every row, so an
    implementation that gathered sample 0's indices for the whole batch would
    pass them. This uses disjoint, non-contiguous patterns per sample so that
    bug would surface.
    """
    encoder = make_encoder()
    torch.manual_seed(0)
    images = torch.randn(2, 3, 32, 32)

    keep_mask = torch.zeros(2, 64, dtype=torch.bool)
    first_kept = [1, 3, 5]
    second_kept = [0, 4, 7]
    keep_mask[0, first_kept] = True
    keep_mask[1, second_kept] = True

    tokens, _ = encoder(images, keep_mask)
    assert tokens.shape == (2, 3, 32)

    # Encode each sample alone with its own mask; results must match the row
    # that sample occupied in the batched call.
    for row, kept in enumerate((first_kept, second_kept)):
        single_mask = torch.zeros(1, 64, dtype=torch.bool)
        single_mask[0, kept] = True
        single_tokens, _ = encoder(images[row : row + 1], single_mask)
        assert torch.allclose(tokens[row], single_tokens[0], atol=1e-5)
