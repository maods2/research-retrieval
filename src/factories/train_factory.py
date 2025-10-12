import pipelines.training_pipes as pipelines

def get_train_function(config):

    match config['training']['pipeline']:
        case 'default_trainer':
            return pipelines.DefaultTrainer(config)

        case 'fsl_trainer':
            return pipelines.FewShotTrainer(config)

        case 'terumo_prototypical_trainer':
            return pipelines.terumo.PrototypicalRetrievalTrainer(config)

        case _:
            raise ValueError(
                f'Training pipeline {config["training"]["pipeline"]} is not supported'
            )
