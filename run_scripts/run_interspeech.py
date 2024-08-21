from run_experiments import run_experiments
import torch

if __name__ == '__main__':
    group = 'interspeech24'
    model_config = {'dense_layers': 2, 'layer_size': 256, 'training_precision': torch.float16, 'kde_training_temperature': 8, 'kde_training_grid_size': 64, 'predict_dist': False}
    one_hot_config = {**model_config, 'one_hot_annotators': True}
    jobs = {
        # 'baseline': (group, 'baseline', model_config),
        # 'mt_task_1': (group, 'task1', model_config),
        'mt_task_3': (group, 'task3', model_config),
        # 'mt_task_13': (group, 'task1,task3', model_config),
        'one_hot_task_1': (group, 'task1', one_hot_config),
        'one_hot_task_3': (group, 'task3', one_hot_config),
        'one_hot_task_13': (group, 'task1,task3', one_hot_config),
    }
    run_experiments(jobs, output_path='/z/tavernor/code/ModelingIndividualEvaluators/.output_files', dataset_cache_path='/z/tavernor/code/ModelingIndividualEvaluators/prepared_datasets/improv_dataset.pk')