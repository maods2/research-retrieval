from typing import Any, Callable, Dict, Tuple, Optional

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm

from core.base_metric_logger import BaseMetricLogger
from core.base_trainer import BaseTrainer
from factories.metric_factory import get_similarity_function, get_similarity_function_from_config, get_metrics
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
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        device: str,
        logger: Callable,
    ) -> Dict[str, Any]:
        # TODO FIXME reuse something from embedding_utils?
        """Evaluate N-Branch MLP using MAP@K with asymmetric branches/splits.

        - Queries: embeddings from TEST dataloader using branch 0 (first branch).
        - Database/Keys: embeddings from TRAIN dataloader using branch 1 (second branch).
        """
        assert hasattr(model, 'mlps') and len(model.mlps) >= 2, \
            'N_BranchMLP with at least 2 branches is required for this evaluation.'

        model.eval().to(device)

        # Resolve class mapping from train dataset
        train_ds = train_dataloader.dataset
        # test_ds = test_dataloader.dataset
        class_mapping_inv = {cls_idx: cls_name for cls_name, cls_idx in train_ds.class_mapping.items()}

        # -------------------------------
        # Build database (keys) from training data using branch 1
        # -------------------------------
        db_embeddings_chunks, db_labels_chunks = [], []
        with torch.no_grad():
            for inputs, targets in tqdm(train_dataloader, desc='Eval: building DB (train, branch=1)'):
                inputs = inputs.to(device)
                outputs = model(*[inputs] * len(model.mlps))
                emb = outputs[1].detach().cpu().numpy()  # second branch
                db_embeddings_chunks.append(emb)
                db_labels_chunks.append(targets.detach().cpu().numpy())

        db_embeddings = np.vstack(db_embeddings_chunks) if db_embeddings_chunks else np.empty((0,))
        db_labels = np.hstack(db_labels_chunks) if db_labels_chunks else np.empty((0,), dtype=int)

        # -------------------------------
        # Build queries from test data using branch 0
        # -------------------------------
        query_embeddings_chunks, query_labels_chunks = [], []
        with torch.no_grad():
            for inputs, targets in tqdm(test_dataloader, desc='Eval: building Queries (test, branch=0)'):
                inputs = inputs.to(device)
                outputs = model(*[inputs] * len(model.mlps))
                emb = outputs[0].detach().cpu().numpy()  # first branch
                query_embeddings_chunks.append(emb)
                query_labels_chunks.append(targets.detach().cpu().numpy())

        query_embeddings = np.vstack(query_embeddings_chunks) if query_embeddings_chunks else np.empty((0,))
        query_labels = np.hstack(query_labels_chunks) if query_labels_chunks else np.empty((0,), dtype=int)
        query_classes = [class_mapping_inv.get(int(l), str(int(l))) for l in query_labels]

        # Paths are optional in MapAtK; supply None to keep API satisfied
        query_paths = [None] * len(query_labels)
        db_paths = [None] * len(db_labels)

        # Metric setup
        map_at_k_metric = get_metrics(self.config)[0]
        assert isinstance(map_at_k_metric, MapAtK), "Expected only one metric: MapAtK"

        # Compute and log MAP@K
        embeddings_payload = {
            'query_embeddings': query_embeddings,
            'query_labels': query_labels,
            'query_classes': query_classes,
            'query_paths': query_paths,
            'db_embeddings': db_embeddings,
            'db_labels': db_labels,
            'db_path': db_paths,
            'class_mapping': class_mapping_inv or {},
        }
        results = map_at_k_metric(
            model=None,
            train_loader=None,
            test_loader=None,
            embeddings=embeddings_payload,
            config=None,
            logger=logger,
        )
        for k, mapk in results['map_at_k_results'].items():
            logger.info(f'MAP@{k}: {mapk:.4f}')
            print(f'MAP@{k}: {mapk:.4f}')

        model.train().to(device)  # re-enable training mode (e.g., dropout)
        return results

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
            # print(loss.item())
            per_img_loss = loss.item() / len(inputs)
            print_grad_stats(model, progress_bar, per_img_loss=per_img_loss)
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

            # Evaluate using asymmetric branches: queries from eval (test), db from train
            metrics = self.evaluate(
                ctx.model,
                ctx.train_loader,
                ctx.eval_loader,
                device,
                ctx.logger,
            )

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
