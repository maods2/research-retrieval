import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
)

from core.base_metric_logger import BaseMetricLogger
from core.base_trainer import BaseTrainer
from losses.supervised_hashing_loss import SupervisedHashingLoss
from schemas.training_context import TrainingContext
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Tuple, Dict, Any, Callable
from tqdm import tqdm


class SupervisedHashingTrainer(BaseTrainer):
    """
    Supervised Hashing Trainer for Deep Supervised Hashing (DSH).
    
    This trainer implements the supervised hashing approach for learning
    binary hash codes for image retrieval tasks. It uses a triplet-like loss
    function to learn representations that preserve semantic similarity.
    """

    def __init__(self, config: dict):
        self.config = config
        super().__init__()

    def train_one_epoch(
        self, model, loss_fn, optimizer, dataloader, device, epoch
    ):
        """Train the model for one epoch."""
        model.train()
        running_loss = 0.0
        running_positive_loss = 0.0
        running_negative_loss = 0.0
        running_regularization_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch + 1}')

        for batch in progress_bar:
            x_imgs, x_targets, y_imgs, y_targets, target_equals = batch
            x_imgs = x_imgs.to(device)
            y_imgs = y_imgs.to(device)
            target_equals = target_equals.to(device).float()

            optimizer.zero_grad()
            
            # Forward pass
            x_out = model(x_imgs)
            y_out = model(y_imgs)
            
            # Compute loss
            loss_dict = loss_fn(x_out, y_out, target_equals)
            total_loss = loss_dict['total_loss']
            
            # Backward pass
            total_loss.backward()
            optimizer.step()

            # Update running losses
            running_loss += total_loss.item()
            running_positive_loss += loss_dict['positive_loss'].item()
            running_negative_loss += loss_dict['negative_loss'].item()
            running_regularization_loss += loss_dict['regularization_loss'].item()

            progress_bar.set_postfix(
                loss=total_loss.item(),
                pos_loss=loss_dict['positive_loss'].item(),
                neg_loss=loss_dict['negative_loss'].item(),
                reg_loss=loss_dict['regularization_loss'].item()
            )

        avg_loss = running_loss / len(dataloader)
        avg_positive_loss = running_positive_loss / len(dataloader)
        avg_negative_loss = running_negative_loss / len(dataloader)
        avg_regularization_loss = running_regularization_loss / len(dataloader)
        
        return {
            'total_loss': avg_loss,
            'positive_loss': avg_positive_loss,
            'negative_loss': avg_negative_loss,
            'regularization_loss': avg_regularization_loss
        }

    def evaluate(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        ctx: TrainingContext,
        device: str,
        logger: Callable = None,
    ) -> Dict[str, Any]:
        """Evaluate the model on the given dataloader."""
        model.eval()
        all_losses = {
            'total_loss': [],
            'positive_loss': [],
            'negative_loss': [],
            'regularization_loss': []
        }
        loss_fn = ctx.loss_fn
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Evaluating'):
                x_imgs, x_targets, y_imgs, y_targets, target_equals = batch
                x_imgs = x_imgs.to(device)
                y_imgs = y_imgs.to(device)
                target_equals = target_equals.to(device).float()

                # Forward pass
                x_out = model(x_imgs)
                y_out = model(y_imgs)
                
                # Compute loss using the loss function from config
                
                loss_dict = loss_fn(x_out, y_out, target_equals)
                
                for key in all_losses:
                    all_losses[key].append(loss_dict[key].item())

        # Compute average losses
        avg_losses = {key: np.mean(values) for key, values in all_losses.items()}
        return avg_losses

    def __call__(self, ctx: TrainingContext):
        """Main training loop."""
        device = ctx.config.get(
            'device', 'cuda' if torch.cuda.is_available() else 'cpu'
        )
        ctx.model.to(device)
        epochs = ctx.config['training']['epochs']
        patience = ctx.config['training'].get('early_stopping_patience', 10)

        min_loss = float('inf')
        epochs_without_improvement = 0
        checkpoint_path = None
        train_history = {
            'total_loss': [],
            'positive_loss': [],
            'negative_loss': [],
            'regularization_loss': [],
            'val_total_loss': [],
            'val_positive_loss': [],
            'val_negative_loss': [],
            'val_regularization_loss': []
        }

        for epoch in range(epochs):
            # Training
            train_metrics = self.train_one_epoch(
                ctx.model, ctx.loss_fn, ctx.optimizer, ctx.train_loader, device, epoch
            )

            # Validation
            val_metrics = self.evaluate(
                ctx.model, ctx.eval_loader, ctx, device, ctx.logger
            )

            # Logging
            ctx.logger.info(
                f'[Epoch {epoch + 1}/{epochs}] '
                f'Train Loss: {train_metrics["total_loss"]:.4f} | '
                f'Val Loss: {val_metrics["total_loss"]:.4f}'
            )
            print(
                f'[Epoch {epoch + 1}/{epochs}] '
                f'Train Loss: {train_metrics["total_loss"]:.4f} | '
                f'Val Loss: {val_metrics["total_loss"]:.4f}'
            )

            # Update history
            for key in train_metrics:
                train_history[key].append(train_metrics[key])
            for key in val_metrics:
                train_history[f'val_{key}'].append(val_metrics[key])

            # Save model if best
            (
                should_stop,
                min_loss,
                epochs_without_improvement,
                checkpoint_path,
            ) = self.save_model_if_best(
                model=ctx.model,
                metric=train_metrics['total_loss'],
                best_metric=min_loss,
                epochs_without_improvement=epochs_without_improvement,
                checkpoint_path=checkpoint_path,
                config=ctx.config,
                metric_logger=ctx.metric_logger,
                mode='loss',
            )

            if should_stop:
                ctx.logger.info(
                    f'Early stopping triggered after {epochs_without_improvement} epochs.'
                )
                print(
                    f'Early stopping triggered after {epochs_without_improvement} epochs.'
                )
                break

        train_history['last_epoch_metrics'] = {
            'train': train_metrics,
            'val': val_metrics
        }
        ctx.metric_logger.log_json(train_history, 'train_metrics')

        return ctx.model


# -------------------------
# Quick Test (Dummy)
# -------------------------
if __name__ == '__main__':
    import torchvision.transforms as T
    from torchvision.datasets import MNIST
    from torch.utils.data import DataLoader
    from models.liu_dsh import LiuDSH
    from losses.supervised_hashing_loss import SupervisedHashingLoss
    from dataloaders.dataset_supervised_hashing import SupervisedHashingDataset

    # Transform
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.1307,), std=(0.3081,))
    ])

    # Dataset - Note: This example uses MNIST as base dataset
    # In real usage, you would use the framework's dataset loading
    base_dataset = MNIST(root='./data', train=True, transform=transform, download=True)
    
    # For demonstration, we'll create a simple wrapper to use MNIST with our dataset
    class MNISTWrapper:
        def __init__(self, mnist_dataset):
            self.mnist_dataset = mnist_dataset
        
        def __len__(self):
            return len(self.mnist_dataset)
        
        def __getitem__(self, idx):
            return self.mnist_dataset[idx]
    
    # Create dataset using the framework's approach
    # In practice, you would use the dataset factory or direct instantiation
    # with proper root_dir and class_mapping
    dataset = SupervisedHashingDataset(
        root_dir='./data/example',  # This would be your actual dataset path
        transform=None,  # Transforms are handled by the parent class
        train=True
    )
    
    # For this example, let's create a simple dataloader
    # dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    # Model
    model = LiuDSH(code_size=8, num_classes=10)

    # Loss function
    loss_fn = SupervisedHashingLoss(config={'margin': 5.0, 'alpha': 0.01})

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    # Trainer
    trainer = SupervisedHashingTrainer(
        config={'training': {'epochs': 10, 'early_stopping_patience': 3}}
    )

    print("SupervisedHashingTrainer created successfully!")
    print("To use with the framework, integrate with the dataset factory and training pipeline.")
