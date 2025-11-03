from typing import Any, Callable, Dict, Tuple, Optional

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm

from core.base_metric_logger import BaseMetricLogger
from core.base_trainer import BaseTrainer
from factories.metric_factory import get_similarity_function, get_similarity_function_from_config
from models.n_branch_mlp import N_BranchMLP
from metrics.map_at_k import MapAtK
from schemas.training_context import TrainingContext
from utils.debug_utils import print_grad_stats

class TerumoContrastiveTrainer(BaseTrainer):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config

    def evaluate(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        device: str,
        logger: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Evaluate the model on the given dataloader."""
        # # dummy test: MAP@k against itself
        # model.eval().to(device)
        # embeddings = []
        # labels = []
        # with torch.no_grad():
        #     for inputs, targets in tqdm(dataloader, desc='Evaluating'):
        #         inputs = inputs.to(device)
        #         outputs = model(*[inputs]*len(model.mlps))
        #         # Use the first branch for evaluation
        #         emb = outputs[0].cpu().numpy()
        #         targets = targets.cpu().numpy()
        #         embeddings.append(emb)
        #         labels.append(targets)
        # embeddings = np.vstack(embeddings)
        # labels = np.hstack(labels)
        # metric = MapAtK(k_values=[10], similarity_fn=(get_similarity_function('cosine'), 'cosine'))
        # metric.map_at_k({
        #     'query_embeddings': embeddings,
        #     'query_labels': labels,
        #     'db_embeddings': embeddings,
        #     'db_labels': labels,
        # })
        # model.train()  # to enable dropout if any
        return {}

    # --------------------------
    # One epoch
    # --------------------------
    def train_one_epoch(
        self,
        model: N_BranchMLP,
        loss_fn,
        optimizer,
        train_loader,
        device,
        epoch,
    ):
        model.train().to(device)
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}')

        for inputs, labels in progress_bar:
            optimizer.zero_grad()
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(*[inputs]*len(model.mlps))
            loss = loss_fn(outputs, [labels]*len(model.mlps))

            loss.backward()
            print_grad_stats(model, progress_bar, loss=loss.item())
            optimizer.step()

            running_loss += loss.item()
            # progress_bar.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(train_loader)
        return avg_loss

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
            avg_loss = self.train_one_epoch(
                ctx.model,
                ctx.loss_fn,
                ctx.optimizer,
                ctx.train_loader,
                device,
                epoch,
            )

            metrics = self.evaluate(ctx.model, ctx.eval_loader, device, ctx.logger)

            ctx.logger.info(
                f'[Epoch {epoch+1}/{epochs}] Loss: {avg_loss:.4f}' 
            )
            print(
                f'[Epoch {epoch+1}/{epochs}] Loss: {avg_loss:.4f}' 
            )

            history['loss'].append(avg_loss)

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
