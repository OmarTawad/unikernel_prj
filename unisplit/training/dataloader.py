"""DataLoader factory with configuration support."""

from __future__ import annotations

from torch.utils.data import DataLoader, Dataset

from unisplit.shared.config import TrainingConfig


def create_dataloader(
    dataset: Dataset,
    config: TrainingConfig | None = None,
    batch_size: int | None = None,
    shuffle: bool = True,
    num_workers: int | None = None,
    drop_last: bool = False,
) -> DataLoader:
    """Create a DataLoader with sensible defaults.

    Args:
        dataset: PyTorch Dataset instance.
        config: Training config for batch_size and num_workers.
        batch_size: Override batch size (takes precedence over config).
        shuffle: Whether to shuffle data.
        num_workers: Override num_workers.
        drop_last: Whether to drop the last incomplete batch.

    Returns:
        Configured DataLoader.
    """
    if config is not None:
        bs = batch_size or config.batch_size
        nw = num_workers if num_workers is not None else config.num_workers
    else:
        bs = batch_size or 256
        nw = num_workers if num_workers is not None else 0

    return DataLoader(
        dataset,
        batch_size=bs,
        shuffle=shuffle,
        num_workers=nw,
        pin_memory=False,  # CPU-first, no pin_memory needed
        drop_last=drop_last,
    )
