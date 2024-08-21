import re
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
from datetime import datetime
from MIA.utils import Singleton

class Logger(metaclass=Singleton):
    def __init__(self, log_dir='runs'):
        print('creating logger with dir', log_dir)
        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y_%H:%M:%S")
        self.base_log_dir = f'{log_dir}_{dt_string}'
        self.loggers = {} # defaultdict(lambda model_name: SummaryWriter(log_dir=f'{self.base_log_dir}/{model_name}'))
        self.layouts = defaultdict(dict)
        self.cached_names = {} # Use a dictionary for caching for faster lookup 

    def update_custom_scalars(self, log_type, group, model_name):
        cache_name = f'{log_type}/{group}/{log_type}/{group}/{model_name}' # Add model name onto the end even though it's not used during logging scalar as cache will be logger specific
        if cache_name in self.cached_names:
            return
        # If not in the cache we need to update layout to include this value 
        scalar_name = fr'{log_type}/{group}([^ ]|$)'
        if log_type not in self.layouts[model_name]:
            self.layouts[model_name][log_type] = {}
        if group not in self.layouts[model_name][log_type]:
            self.layouts[model_name][log_type][group] = ['Multiline', []]
        if scalar_name not in self.layouts[model_name][log_type][group][1]:
            self.layouts[model_name][log_type][group][1].append(scalar_name)
        
        logger = self.get_logger(model_name)
        logger.add_custom_scalars(self.layouts[model_name])
        self.cached_names[cache_name] = True

    def log_scalar(self, log_type, group, model_name, scalar_value, step):
        # If we have been asked to plot a scalar that addresses individual annotators, we want to group this by model rather than by metric
        # Brackets mess up the custom scalar logging. Remove all here.
        if 'train' in log_type.lower() and step % 100 and 'batch_loss' not in group.lower():
            return # Only log every 100 steps
        log_type = log_type.replace('(', '').replace(')', '')
        group = group.replace('(', '').replace(')', '')
        # scalar_name = scalar_name.replace('(', '').replace(')', '')
        # m = re.search(r'Annotator \d+', group)
        # if m is not None:
        #     # We want to replace the matching text in group with the model name, and replace model name with the match + "by Annotator"
        #     # This means that instead of a graph titled e.g. Loss Annotator 0 Activation CCC loss we will have a graph titled Loss model_name by Annotator Activation CCC loss
        #     # With each line being the annotators of the model in the graph
        #     m = m.group(0)
        #     group = group.replace(m, f'{scalar_name} by Annotator')
        #     scalar_name = m

        self.update_custom_scalars(log_type, group, model_name)
        self.get_logger(model_name).add_scalar(f'{log_type}/{group}', scalar_value, step)

    def get_logger(self, name):
        if name not in self.loggers:
            self.loggers[name] = SummaryWriter(log_dir=f'{self.base_log_dir}/{name}')
        return self.loggers[name]