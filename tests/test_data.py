import torch

from jepa_lens.data import sample_block_masks


def test_masks_have_expected_shapes():
    generator = torch.Generator().manual_seed(0)
    context, targets = sample_block_masks(
        batch_size=4,
        grid_size=8,
        num_target_blocks=4,
        block_scale=(0.15, 0.2),
        block_aspect=(0.75, 1.5),
        generator=generator,
    )
    assert context.shape == (4, 64)
    assert context.dtype == torch.bool
    assert targets.ndim == 2
    assert targets.shape[0] == 4


def test_context_and_targets_are_disjoint():
    """A patch used as a target must not also be visible as context."""
    generator = torch.Generator().manual_seed(1)
    context, targets = sample_block_masks(
        batch_size=8,
        grid_size=8,
        num_target_blocks=4,
        block_scale=(0.15, 0.2),
        block_aspect=(0.75, 1.5),
        generator=generator,
    )
    for row in range(8):
        assert not context[row][targets[row]].any()


def test_every_sample_keeps_same_context_count():
    """The ViT requires a rectangular kept-token tensor."""
    generator = torch.Generator().manual_seed(2)
    context, _ = sample_block_masks(
        batch_size=16,
        grid_size=8,
        num_target_blocks=4,
        block_scale=(0.15, 0.2),
        block_aspect=(0.75, 1.5),
        generator=generator,
    )
    counts = context.sum(dim=1)
    assert bool((counts == counts[0]).all())


def test_context_is_not_empty():
    generator = torch.Generator().manual_seed(3)
    context, _ = sample_block_masks(
        batch_size=4,
        grid_size=8,
        num_target_blocks=4,
        block_scale=(0.15, 0.2),
        block_aspect=(0.75, 1.5),
        generator=generator,
    )
    assert int(context.sum(dim=1).min()) > 0


def test_deterministic_under_same_seed():
    def draw():
        generator = torch.Generator().manual_seed(7)
        return sample_block_masks(4, 8, 4, (0.15, 0.2), (0.75, 1.5), generator)

    first_context, first_targets = draw()
    second_context, second_targets = draw()
    assert torch.equal(first_context, second_context)
    assert torch.equal(first_targets, second_targets)
