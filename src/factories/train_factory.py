def get_train_function(config):

    if config['training']['pipeline'] == 'default_trainer':
        from pipelines.training_pipes.default_trainer import DefaultTrainer

        return DefaultTrainer(config)

    elif config['training']['pipeline'] == 'fsl_trainer':
        from pipelines.training_pipes.few_shot_trainer import FewShotTrainer

        return FewShotTrainer(config)

    elif config['training']['pipeline'] == 'supcon_trainer':
        from pipelines.training_pipes.supcon_trainer import SupConTrainer

        return SupConTrainer(config)

    elif config['training']['pipeline'] == 'supervised_hashing_trainer':
        from pipelines.training_pipes.supervised_hashing_trainer import SupervisedHashingTrainer

        return SupervisedHashingTrainer(config)

    else:
        raise ValueError(
            f'Training pipeline {config["training"]["pipeline"]} is not supported'
        )
