from MIA.training import training_epoch, validation_epoch, test_epoch, cc_test_epoch
from MIA.tb_logging import Logger
from MIA.models import ModelTrainer
from torch.utils.data import DataLoader
from SERDatasets import IEMOCAPDatasetConstructor, PodcastDatasetConstructor, ImprovDatasetConstructor, MuSEDatasetConstructor
from MIA.training import BatchCollator
import torch
import random
import numpy as np

def run_task(log_path, model_save_path, model_arguments, regenerate_kde_labels, prob_grid_size, improv_path, task_str='task1,task2'):
    logger = Logger(log_path) # Create singleton logger now 
    logger.__init__(log_path) # Ensure that the singleton is re-initialised in case log path changes
    # Need to make sure that identical training methods with different log_path are not overwriting each other 
    train_dataset = ImprovDatasetConstructor(dataset_save_location=improv_path)
    train_dataset.close_h5()
    train_dataset.open_h5_read_only()
    torch.use_deterministic_algorithms(True)
    torch.autograd.set_detect_anomaly(True) 
    torch.backends.cudnn.deterministic = True
    # Constant model arguments
    model_arguments['prob_grid_size'] = prob_grid_size
    annotators = set(train_dataset.individual_annotators.keys())
    model_arguments['annotators'] = annotators
    for i in range(5):
        torch.manual_seed(i)
        torch.cuda.manual_seed(i)
        torch.cuda.manual_seed_all(i)
        random.seed(i)
        np.random.seed(i)

        batch_size = 32
        max_epochs = 500
        train_sets = train_dataset.build('speaker-independent', kde_size=prob_grid_size)
        # Model name to appear in tensorboard
        model_name = f'model_prob{prob_grid_size}_seed_{i}'
        model_trainer = ModelTrainer(prob_grid_size, model_name, model_arguments, model_save_path, training_tasks=task_str)
        if hasattr(model_trainer.model, 'act_heads'):
            batch_collator = BatchCollator(annotator_mapper=model_trainer.model.act_heads.annotator_mapper)
        elif hasattr(model_trainer.model, 'one_hot_encode'):
            batch_collator = BatchCollator(annotator_mapper=model_trainer.model.one_hot_encode.annotator_mapper)
        else:
            batch_collator = BatchCollator()

        train_dataloader = DataLoader(train_sets['train'], batch_size=batch_size, shuffle=True, drop_last=False, collate_fn=batch_collator, num_workers=0, pin_memory=False)
        val_dataloader = DataLoader(train_sets['val'], batch_size=256, shuffle=False, drop_last=False, collate_fn=batch_collator, num_workers=0, pin_memory=False)

        validation_epoch(val_dataloader, model_trainer, prob_grid_size, -1)
        # if model_trainer.moe_models:
            # validation_epoch(val_dataloader, model_trainer, None, prob_grid_size, -1, soft_hist=True)
        for epoch in range(max_epochs):
            print(f'Epoch {epoch} begin')
            training_epoch(train_dataloader, model_trainer, epoch)
            validation_epoch(val_dataloader, model_trainer, prob_grid_size, epoch)
            # if model_trainer.moe_models:
                # validation_epoch(val_dataloader, model_trainer, None, prob_grid_size, epoch, soft_hist=True)
            all_finished = not model_trainer.continue_training
            # for name in model_trainer.models:
            #     all_finished = all_finished and not model_trainer.models[name].continue_training
            if all_finished:
                break
            if 'moe_baseline' in model_name or not regenerate_kde_labels:
                continue # No point in recreating the KDE values as these models do not use the KDE values. The validation set metrics will be less trustworthy, but early stopping will be used on CCC 
            train_sets = train_dataset.build('speaker-independent', kde_size=prob_grid_size)
            train_dataloader = DataLoader(train_sets['train'], batch_size=batch_size, shuffle=True, drop_last=False, collate_fn=batch_collator, num_workers=0, pin_memory=False)
            val_dataloader = DataLoader(train_sets['val'], batch_size=256, shuffle=False, drop_last=False, collate_fn=batch_collator, num_workers=0, pin_memory=False)
        # Test model 
        print('Loading best model based on early stopping criteria')
        model_trainer.load_best_model()
        test_dataloader = DataLoader(train_sets['test'], batch_size=256, shuffle=False, drop_last=False, collate_fn=batch_collator, num_workers=0, pin_memory=False)
        print('Testing improv (1)')
        test_epoch(test_dataloader, model_trainer, prob_grid_size, soft_hist=False, use_all_annotators=False)
        if not model_trainer.model.args.predict_dist:
            print('Testing improv (2)')
            test_epoch(test_dataloader, model_trainer, prob_grid_size, soft_hist=False, use_all_annotators=True)
            print('Testing improv (3)')
            test_epoch(test_dataloader, model_trainer, prob_grid_size, soft_hist=True, use_all_annotators=False)
            print('Testing improv (4)')
            test_epoch(test_dataloader, model_trainer, prob_grid_size, soft_hist=True, use_all_annotators=True)

        # Now load cross-corpus data
        train_dataset.close_h5() 
        test_datasets = {}
        test_datasets['podcast'] = PodcastDatasetConstructor(dataset_save_location=improv_path.replace('improv', 'podcast'))
        test_datasets['podcast'].close_h5()
        test_datasets['iemocap'] = IEMOCAPDatasetConstructor(dataset_save_location=improv_path.replace('improv', 'iemocap'))
        test_datasets['iemocap'].close_h5()
        test_datasets['muse'] = MuSEDatasetConstructor(dataset_save_location=improv_path.replace('improv', 'muse'))
        test_datasets['muse'].close_h5()
        train_dataset.open_h5_read_only()
        test_datasets['podcast'].open_h5_read_only()
        test_datasets['iemocap'].open_h5_read_only()
        test_datasets['muse'].open_h5_read_only()

        build_split_mapping = {
            'podcast': 'speaker-independent',
            'iemocap': 'all',
            'muse': 'stress_type_split',
        }
        built_datasets = {
            key: test_datasets[key].build(build_split_mapping[key], kde_size=prob_grid_size, test_only=True) for key in test_datasets
        }
        built_datasets['iemocap'] = {'all': built_datasets['iemocap']}
        test_splits = {
            f'{dataset} {split}': built_datasets[dataset][split] for dataset in built_datasets for split in built_datasets[dataset]
        }
        test_dataloaders = {
            f'{split}_generation': DataLoader(test_splits[split], batch_size=256, shuffle=False, drop_last=False, collate_fn=batch_collator, num_workers=0, pin_memory=False) for split in test_splits
        }
        # Stop the batch collator from attempting to batch annotators as other datasets won't contain improv annotators
        batch_collator.batch_annotators = False
        for key in test_dataloaders:
            print(f'Testing {key} (1)')
            cc_test_epoch(test_dataloaders[key], model_trainer, prob_grid_size, test_name=f'Test ({key})', soft_hist=False)
            if not model_trainer.model.args.predict_dist:
                print(f'Testing {key} (2)')
                cc_test_epoch(test_dataloaders[key], model_trainer, prob_grid_size, test_name=f'Test ({key})', soft_hist=True)
        batch_collator.batch_annotators = True
