from core.base_evaluator import BaseEvaluator
from core.base_metric_logger import BaseMetricLogger
from dataloaders.dataset_fewshot import SupportSetDataset
from pipelines.training_pipes.few_shot_trainer import FewShotTrainer
from torch.utils.data import DataLoader

import torch


class FSLEvaluator(BaseEvaluator):
    @staticmethod
    def load_support_set_from_loader(config, device='cpu'):
        """
        Load all images and labels from the support_loader and stack them into two single tensors.
        Returns:
            support_set: Tensor [N_total, C, H, W]
            support_labels: Tensor [N_total]
        """
        support_loader = DataLoader(
            dataset=SupportSetDataset(
                config['data']['train_dir'],
                transform=None,
                class_mapping=config['data']['class_mapping'],
                config=config,
                n_per_class=config['model']['k_shot'],
            ),
            batch_size=10,
            shuffle=False,
        )
        all_images = []
        all_labels = []
        for images, labels in support_loader:
            all_images.append(images)
            all_labels.append(labels)
        support_set = torch.cat(all_images, dim=0).to(device)
        support_labels = torch.cat(all_labels, dim=0).to(device)
        return support_set, support_labels

    def test(
        self,
        model,
        train_loader,
        test_loader,
        config,
        logger,
        metric_logger: BaseMetricLogger,
    ):
        """
        Run metrics on the test data and log the results.
        """
        device = (
            config['device']
            if config.get('device')
            else ('cuda' if torch.cuda.is_available() else 'cpu')
        )
        fslt = FewShotTrainer(config={})
        model.to(device)
        support_set = self.load_support_set_from_loader(config, device=device)
        train_val = fslt.eval_few_shot_classification(
            model,
            train_loader,
            support_set,
            device=device,
            config=config,
            logger=logger,
        )
        test_val = fslt.eval_few_shot_classification(
            model,
            test_loader,
            support_set,
            device=device,
            config=config,
            logger=logger,
        )
        results = {'train': train_val, 'test': test_val}
        logger.info(f'Results for fsl classification: {results}')
        metric_logger.log_metrics(results)
