def get_train_function(config):

    if config['training']['pipeline'] == 'train_few_shot_leaning':
        from pipelines.training_pipes.few_shot_train import FewShotTrain

        return FewShotTrain(config)

    elif config['training']['pipeline'] == 'train_mynet':
        from pipelines.training_pipes.mynet_train import MyNetTrain

        return MyNetTrain(config)

    else:
        raise ValueError(
            f'Training pipeline {config["training"]["pipeline"]} is not supported'
        )
