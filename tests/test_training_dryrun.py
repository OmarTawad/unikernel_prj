"""Training dry-run test — validates architecture works with synthetic data."""

import torch

from unisplit.model.cnn import IoTCNN
from unisplit.training.trainer import dry_run


class TestTrainingDryRun:
    def test_dry_run_passes(self):
        model = IoTCNN()
        device = torch.device("cpu")
        assert dry_run(model, device, num_batches=2)

    def test_model_trains(self):
        """2 batches of synthetic data, loss should be finite."""
        from unisplit.training.dataset import SyntheticDataset
        from unisplit.training.dataloader import create_dataloader

        model = IoTCNN()
        dataset = SyntheticDataset(num_samples=64)
        loader = create_dataloader(dataset, batch_size=32, num_workers=0)

        model.train()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())

        for features, labels in loader:
            optimizer.zero_grad()
            out = model(features)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            assert loss.item() > 0
            assert torch.isfinite(loss)
            break  # Just 1 batch
