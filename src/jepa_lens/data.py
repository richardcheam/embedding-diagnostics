"""CIFAR-10 loading and I-JEPA-style block masking.

Masking follows the multi-block strategy of I-JEPA (arXiv 2301.08243):
sample several rectangular target blocks on the patch grid, and take the
context to be the patches none of those blocks cover.

Two implementation constraints shape the code below. The encoder needs a dense
kept-token tensor, so every sample in a batch must retain the same number of
context patches and the same number of target patches. Both are enforced by
truncating to the batch minimum after sampling.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def _sample_one_block(
    grid_size: int,
    block_scale: tuple[float, float],
    block_aspect: tuple[float, float],
    generator: torch.Generator,
) -> torch.Tensor:
    """Return a bool mask (grid_size**2,) covering one rectangular block."""
    num_patches = grid_size**2
    scale = block_scale[0] + torch.rand(1, generator=generator).item() * (
        block_scale[1] - block_scale[0]
    )
    aspect = block_aspect[0] + torch.rand(1, generator=generator).item() * (
        block_aspect[1] - block_aspect[0]
    )
    area = max(1.0, scale * num_patches)
    height = max(1, min(grid_size, int(round((area * aspect) ** 0.5))))
    width = max(1, min(grid_size, int(round(area / height))))

    top = int(torch.randint(0, grid_size - height + 1, (1,), generator=generator).item())
    left = int(torch.randint(0, grid_size - width + 1, (1,), generator=generator).item())

    mask = torch.zeros(grid_size, grid_size, dtype=torch.bool)
    mask[top : top + height, left : left + width] = True
    return mask.flatten()


def sample_block_masks(
    batch_size: int,
    grid_size: int,
    num_target_blocks: int,
    block_scale: tuple[float, float],
    block_aspect: tuple[float, float],
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample context masks and target patch indices for a batch.

    Returns:
        Tuple of (context_keep_mask, target_indices). `context_keep_mask` is a
        bool tensor (batch, grid_size**2) that is True for visible context
        patches. `target_indices` is a long tensor (batch, num_targets) of patch
        indices to predict. The two are disjoint by construction.
    """
    num_patches = grid_size**2
    context_masks: list[torch.Tensor] = []
    target_lists: list[torch.Tensor] = []

    for _ in range(batch_size):
        target_mask = torch.zeros(num_patches, dtype=torch.bool)
        for _ in range(num_target_blocks):
            target_mask |= _sample_one_block(grid_size, block_scale, block_aspect, generator)
        if bool(target_mask.all()):
            target_mask[0] = False
        context_masks.append(~target_mask)
        target_lists.append(target_mask.nonzero(as_tuple=True)[0])

    context_stack = torch.stack(context_masks)
    min_context = int(context_stack.sum(dim=1).min())
    min_targets = int(min(len(indices) for indices in target_lists))

    # Truncation must drop a RANDOM subset, not a prefix. `nonzero` returns
    # ascending flat indices, so slicing `[:min_context]` would keep the
    # lowest-indexed patches every single call — measured at 88% retention for
    # the top-left corner versus 2.8% for the bottom-right. That is a fixed
    # spatial bias in what counts as context, which would confound every
    # comparison this study makes.
    trimmed_context = torch.zeros_like(context_stack)
    for row in range(batch_size):
        candidates = context_stack[row].nonzero(as_tuple=True)[0]
        order = torch.randperm(len(candidates), generator=generator)
        trimmed_context[row, candidates[order][:min_context]] = True

    trimmed_targets = torch.stack(
        [
            indices[torch.randperm(len(indices), generator=generator)][:min_targets]
            for indices in target_lists
        ]
    )
    return trimmed_context, trimmed_targets


def build_dataloaders(config: dict[str, Any]) -> tuple[DataLoader, tuple, tuple]:
    """Build the SSL loader plus frozen probe splits for the configured dataset.

    Dispatches on `data.dataset`: "cifar10" (this module) or "bdd100k"
    (`jepa_lens.bdd100k`). Both return the same (ssl_loader, probe_train,
    probe_test) contract; BDD's probe labels are a dict of scenario attributes
    rather than a single class array.

    Returns:
        Tuple of (ssl_loader, probe_train, probe_test). Each probe split is a
        pair of (images_tensor, labels_array) held in memory; they are small by
        construction and are reused unchanged at every checkpoint so probe
        accuracy is comparable across steps.
    """
    dataset_name = config["data"].get("dataset", "cifar10")
    if dataset_name == "bdd100k":
        from .bdd100k import build_bdd_dataloaders

        return build_bdd_dataloaders(config)
    if dataset_name != "cifar10":
        raise ValueError(f"unknown dataset {dataset_name!r}")

    data_config = config["data"]
    normalize = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)

    ssl_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(config["model"]["image_size"], scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_transform = transforms.Compose([transforms.ToTensor(), normalize])

    ssl_dataset = datasets.CIFAR10(
        root=data_config["root"], train=True, download=True, transform=ssl_transform
    )
    # Explicit generator, not the global RNG. Without one, RandomSampler draws a
    # fresh shuffle seed — and the worker augmentation base_seed — from the
    # global RNG at every epoch. Any condition whose loss also consumes the
    # global RNG would then get a different data order and different
    # augmentations, making data order a confound between conditions.
    loader_generator = torch.Generator().manual_seed(config["seed"])
    ssl_loader = DataLoader(
        ssl_dataset,
        batch_size=data_config["batch_size"],
        shuffle=True,
        num_workers=data_config["num_workers"],
        drop_last=True,
        persistent_workers=data_config["num_workers"] > 0,
        generator=loader_generator,
    )

    probe_train_set = datasets.CIFAR10(
        root=data_config["root"], train=True, download=True, transform=eval_transform
    )
    probe_test_set = datasets.CIFAR10(
        root=data_config["root"], train=False, download=True, transform=eval_transform
    )

    def take(dataset, count: int) -> tuple[torch.Tensor, np.ndarray]:
        """Deterministic probe subset, keyed on eval_split_seed not the run seed."""
        rng = np.random.default_rng(config["data"].get("eval_split_seed", 0))
        indices = rng.choice(len(dataset), size=min(count, len(dataset)), replace=False)
        images = torch.stack([dataset[int(index)][0] for index in indices])
        labels = np.array([dataset[int(index)][1] for index in indices])
        return images, labels

    probe_train = take(probe_train_set, data_config["probe_train_samples"])
    probe_test = take(probe_test_set, data_config["probe_test_samples"])
    return ssl_loader, probe_train, probe_test
