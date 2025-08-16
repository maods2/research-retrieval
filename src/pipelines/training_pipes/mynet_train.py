from core.base_trainer import BaseTrainer

import torch


class MyNetTrain(BaseTrainer):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def train_one_epoch(
        self,
        model,
        loss_fn,
        optimizer,
        train_loader,
        device,
        log_interval,
        epoch,
    ):
        model.train()
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if batch_idx % log_interval == 0:
                print(
                    f'Epoch {epoch+1} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}'
                )
        avg_loss = running_loss / len(train_loader)
        return avg_loss

    def __call__(
        self,
        model,
        loss_fn,
        optimizer,
        train_loader,
        test_loader,
        config,
        logger,
        metric_logger,
    ):
        device = config.get(
            'device', 'cuda' if torch.cuda.is_available() else 'cpu'
        )
        model.to(device)
        epochs = config['training'].get('epochs', 10)
        log_interval = config['training'].get('log_interval', 10)
        for epoch in range(epochs):
            avg_loss = self.train_one_epoch(
                model,
                loss_fn,
                optimizer,
                train_loader,
                device,
                log_interval,
                epoch,
            )
            logger.info(f'Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}')
        return model
