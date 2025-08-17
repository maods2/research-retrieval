from core.base_metric_logger import BaseMetricLogger
from core.base_trainer import BaseTrainer
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Any
from typing import Callable
from typing import Dict
from typing import Tuple
from schemas.training_context import TrainingContext
from utils.metrics_utils import compute_metrics

import numpy as np
import torch


class DefaultTrainer(BaseTrainer):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config

    def evaluate(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        device: str,
        logger: Callable = None,
    ) -> Dict[str, Any]:
        """Evaluate the model on the given dataloader."""
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for inputs, labels in tqdm(dataloader, desc='Evaluating'):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                preds = outputs.argmax(dim=1)

                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        y_true = np.concatenate(all_labels)
        y_pred = np.concatenate(all_preds)
        return compute_metrics(y_true, y_pred, logger)

    # --------------------------
    # One epoch
    # --------------------------
    def train_one_epoch(
        self,
        model,
        loss_fn,
        optimizer,
        train_loader,
        device,
        epoch,
    ):
        model.train()
        running_loss, running_acc = 0.0, 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}')

        for inputs, labels in progress_bar:
            optimizer.zero_grad()
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            _, preds = torch.max(outputs, 1)
            acc = (preds == labels).float().mean().item()

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_acc += acc
            progress_bar.set_postfix(loss=loss.item(), acc=acc)

        avg_loss = running_loss / len(train_loader)
        avg_acc = running_acc / len(train_loader)
        return avg_loss, avg_acc

    # --------------------------
    # Main training loop
    # --------------------------
    def __call__(self, ctx: TrainingContext):      
        device = ctx.config.get(
            'device', 'cuda' if torch.cuda.is_available() else 'cpu'
        )
        epochs = ctx.config['training']['epochs']
        min_loss, epochs_no_improve, checkpoint_path = float('inf'), 0, None
        history = {'loss': [], 'acc': [], 'acc_val': [], 'f1_score_val': []}

        ctx.model.to(device)
        for epoch in range(epochs):
            avg_loss, avg_acc = self.train_one_epoch(
                ctx.model,
                ctx.loss_fn,
                ctx.optimizer,
                ctx.train_loader,
                device,
                epoch,
            )

            metrics = self.evaluate(ctx.model, ctx.eval_loader, device, ctx.logger)

            ctx.logger.info(
                f'[Epoch {epoch+1}/{epochs}] Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}'
            )
            print(
                f'[Epoch {epoch+1}/{epochs}] Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}'
            )

            history['loss'].append(avg_loss)
            history['acc'].append(avg_acc)
            history['acc_val'].append(metrics['accuracy'])
            history['f1_score_val'].append(metrics['f1_score'])

            (
                should_stop,
                min_loss,
                epochs_no_improve,
                checkpoint_path,
            ) = self.save_model_if_best(
                model=ctx.model,
                metric=avg_loss,
                best_metric=min_loss,
                epochs_without_improvement=epochs_no_improve,
                checkpoint_path=checkpoint_path,
                config=ctx.config,
                metric_logger=ctx.metric_logger,
                mode='loss',
            )

            if should_stop:
                ctx.logger.info(
                    f'Early stopping after {epochs_no_improve} epochs with no improvement.'
                )
                print(
                    f'Early stopping after {epochs_no_improve} epochs with no improvement.'
                )
                break

        history['last_epoch_metrics'] = metrics
        ctx.metric_logger.log_json(history, 'train_metrics')
        return ctx.model
