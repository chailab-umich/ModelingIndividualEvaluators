import torch
import torch.optim as optim
from MIA.tb_logging import Logger
from collections import defaultdict
from MIA.loss_metrics import BaselineLoss, Task1Loss, Task2Loss, Task3Loss
from .models import GenericModel
from .early_stopping import EarlyStopping
from .utils import load_torch_model
from MIA.utils import ConfigClass

class ModelTrainer:
    def __init__(self, prob_grid_size, model_name, model_arguments, model_save_path, training_tasks):
        self.training_tasks = training_tasks
        self.prob_grid_size = prob_grid_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if training_tasks == 'baseline':
            model_arguments['predict_dist'] = True
        self.model = GenericModel(**model_arguments).to(self.device)
        config = ConfigClass()

        self.name = model_name
        self.optimizer = optim.SGD(self.model.parameters(), lr=1e-3, momentum=0.9)
        self.early_stopping = EarlyStopping(f'{config["model_save_dir"]}/{model_save_path}/{model_name}')
        self.continue_training = True
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.1, patience=5)
        self.current_lr = None
        self.logger = Logger()
        self.global_batch_step = -1
        self.epoch_losses = defaultdict(list)

        # Define loss functions
        training_task_mapping = {
            'baseline': BaselineLoss,
            'task1': lambda **kwargs: Task1Loss(**kwargs, loss_fn='CCC'),
            'task3': Task3Loss,
        }
        training_tasks = training_tasks.split(',')
        self.loss_fns = []
        for task in training_tasks:
            sparse = '_sparse' in task
            later = '_later' in task
            task = task.replace('_sparse', '').replace('_later', '')
            # Sparse -> once every 10 epochs, later -> use loss after 20 epochs of training
            task_args = {'name': task, 'calculate_kde': task=='task3', 'sparsity': 10 if sparse else None, 'after_warmup': 20 if later else False}
            self.loss_fns.append(training_task_mapping[task](**task_args))

    def load_best_model(self):
        self.model = load_torch_model(GenericModel, self.early_stopping.get_best_model_path(), True)

    def change_prob_grid_size(self, new_size):
        print('Warning may not work depending on model parameters')
        self.model.prob_grid_size = new_size
        self.prob_grid_size = new_size

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def reset_epoch_losses(self):
        self.epoch_losses = defaultdict(list)

    def training_step(self, audio, text, targets, epoch):
        # Track batch number for logging and sparse losses 
        self.global_batch_step += 1

        # Generate masks
        # Only need to use act annotators as we are just checking for non-nan values and this will match with val annotators 
        masks = targets['annotator_masks'] if 'annotator_masks' in targets else None # Set to none if not in targets

        curr_loss_str = ''
        for i, loss_fn in enumerate(self.loss_fns):
            if loss_fn.sparsity is not None and self.global_batch_step % loss_fn.sparsity:
                continue # Skip this loss function as we are only calculating this every sparsity batches
            if loss_fn.after_warmup and epoch < loss_fn.after_warmup:
                continue # Skip this loss function as we are only calculating this after the model has trained that many epochs
            if loss_fn.validation_only:
                continue # Skip this loss as it is only calculated for validation set
            # if loss_fn.name == 'task2':
            #     for p in self.model.parameters():
            #         p.requires_grad = False
            #     for p in self.model.act_head_observer.parameters():
            #         p.requires_grad = True
            #     for p in self.model.val_head_observer.parameters():
            #         p.requires_grad = True
            model_output = self.model(audio, text, skip_kde=not loss_fn.calculate_kde, annotator_masks=masks, curr_task=loss_fn.name) # curr_task only used for logging the last seen task during model output explosion to diagnose culprit
            losses = loss_fn(model_output, targets, masks)
            for loss_name in list(losses.keys()):
                if losses[loss_name] is None:
                    del losses[loss_name]
                    continue # Brent optimisation failed so skip this batch for this loss function
                l = losses[loss_name].item()
                self.logger.log_scalar('Train', f'{loss_fn.name}-batch_loss-{loss_name}', self.name, l, step=self.global_batch_step)
                self.epoch_losses[f'{loss_fn.name}-{loss_name}'].append(l)
            # if loss_fn.name == 'task2':
                # for p in self.model.parameters():
                    # p.requires_grad = True
            if len(losses):
                loss = sum(losses.values())
                loss.backward()
            else:
                return f'{loss_fn.name}: failed'

            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            curr_loss_str += f'{loss_fn.name}: {loss.item():.3f} '

        return curr_loss_str

    def log_average_loss(self, epoch):
        for loss_type in self.epoch_losses:
            average = sum(self.epoch_losses[loss_type])/len(self.epoch_losses[loss_type])
            self.logger.log_scalar(f'train ({self.prob_grid_size}x{self.prob_grid_size})', f'batch_loss_{loss_type}', self.name, average, step=epoch)
        self.reset_epoch_losses()
