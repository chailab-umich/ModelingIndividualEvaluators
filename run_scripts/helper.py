from MIA.run_helper import run_task
from MIA.utils import ConfigClass
import os

def run_helper(script_name, script_group, tasks, config, model_config, dataset_cache):
    config = ConfigClass(config)

    # Get current directory
    log_path = os.path.join(config['tb_log_dir'], f'{script_group}/{script_name}')
    model_save_path = f'{script_group}/{script_name}'

    run_task(log_path, model_save_path, model_config, regenerate_kde_labels=False, prob_grid_size=4, task_str=tasks, improv_path=dataset_cache)