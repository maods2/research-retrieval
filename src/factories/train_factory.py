def get_train_function(config):

    if config['training']['pipeline'] == 'default_trainer':
        from pipelines.training_pipes.default_trainer import DefaultTrainer

        return DefaultTrainer(config)

    elif config['training']['pipeline'] == 'fsl_trainer':
        from pipelines.training_pipes.few_shot_trainer import FewShotTrainer

        return FewShotTrainer(config)

    else:
        raise ValueError(
            f'Training pipeline {config["training"]["pipeline"]} is not supported'
        )
