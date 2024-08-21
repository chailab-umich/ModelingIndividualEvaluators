from tqdm import tqdm
from MIA.loss_metrics import log_metrics, task1_helper
from MIA.tb_logging import Logger
import torch
import numpy as np

# Receive
# dataloader of training samples
# dictionary of name: model_trainer class 
# output size for probability grid (i.e. 4: 4x4 probability grid size)
# Will then computer 1 training epoch for each model 
def training_epoch(train_dataloader, model_trainer, epoch):
    train_pbar = tqdm(train_dataloader)

    model_trainer.train()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for batch in train_pbar:
        audio, text, y_act, y_val, kde_2d_probs, y_act_var, y_val_var = batch['audio'], batch['transcript'], batch[f'act'], batch[f'val'], batch['kde_2d_probability'], batch['act_variance'], batch['val_variance']

        # Train VAE model
        target_probs = torch.as_tensor(np.array(kde_2d_probs), dtype=torch.float32, device=device)
        targets = {'act': batch['act'], 'val': batch['val'], 'kde_2d_probability': target_probs, 'act_variance': y_act_var, 'val_variance': y_val_var}
        if 'padded_individual_annotators_act' in batch:
            targets = {**targets, 'padded_individual_annotators_act': batch['padded_individual_annotators_act'], 'padded_individual_annotators_val': batch['padded_individual_annotators_val'], 'annotator_masks': batch['annotator_masks']}

        output_str = model_trainer.training_step(audio, text, targets, epoch)

        train_pbar.set_description(f'Training loss {output_str}')

    model_trainer.log_average_loss(epoch)

class EvalReturnValues:
    def __init__(self):
        self.return_values = {}
        self.finished = False

    def concat(self, key):
        if key not in self.return_values:
            print(f'Warning attempted to concatenate {key} but eval epoch did not return any values for {key}.')
            return
        self.return_values[key] = torch.cat(self.return_values[key]).squeeze()

    def pad(self, key, two_d_pad=False):
        print('attempting pad', key, two_d_pad)
        if key not in self.return_values:
            print(f'Warning attempted to pad {key} but eval epoch did not return any values for {key}.')
            return
        results = self.return_values[key]
        # Unpad the list of padded batches into list of list
        if not two_d_pad:
            results = [torch.nn.utils.rnn.unpad_sequence(item, (~item.isnan()).sum(dim=-1), batch_first=True) for item in results]
        else:
            results = [torch.nn.utils.rnn.unpad_sequence(item, (~item.sum(dim=-1).isnan()).sum(dim=-1), batch_first=True) for item in results]
        # Flatten into one list
        results = [i for y in results for i in y]
        # Repad all values into one eval epoch of values
        results = torch.nn.utils.rnn.pad_sequence(results, batch_first=True, padding_value=torch.nan)

        self.return_values[key] = results

    def __getitem__(self, key):
        if key not in self.return_values:
            if self.finished:
                raise ValueError(f'Key not found in EvalReturnValues: {key}')
            self.return_values[key] = []
        return self.return_values[key]

    def __setitem__(self, key, value):
        self.return_values[key] = value

    def close(self, concat_ks, pad_1d_ks, pad_2d_ks):
        for key in concat_ks:
            self.concat(key)
        for key in pad_1d_ks:
            self.pad(key, two_d_pad=False)
        for key in pad_2d_ks:
            self.pad(key, two_d_pad=True)
        self.finished = True

def eval_epoch(eval_dataloader, model_trainer, prob_grid_size, soft_hist=True, use_all_annotators=False, fixed_annotators=None, annotator_mappings=None):
    pbar = tqdm(eval_dataloader)
    with torch.no_grad():
        model_results = EvalReturnValues()

        model_trainer.eval()
        device = model_trainer.model.device

        for batch in pbar:
            audio, text, y_act, y_act_bin, y_val, y_val_bin = batch['audio'], batch['transcript'], batch[f'act'], batch[f'act_bin'], batch[f'val'], batch[f'val_bin']
            soft_act, soft_val = batch['soft_act_labels'], batch['soft_val_labels']

            if model_trainer.model.args.predict_dist:
                predictions = model_trainer.model(audio, text)
                model_results['probability_logits'].append(predictions.cpu())
                best = torch.softmax(predictions.view(predictions.shape[0],-1), dim=-1)
            else:
                # Generate the masks for annotator methods -- valence will be the same so only need to read act 
                if annotator_mappings is not None:
                    raise ValueError('Not currently supported')
                    masks = batch['annotator_masks']
                elif fixed_annotators is not None: # Override use_all_annotators in this case 
                    mapper = model_trainer.model.args.annotator_mapper
                    mapper.set_get_idx()
                    masks = [mapper[fixed_annotators] for _ in range(audio.shape[0])]
                elif use_all_annotators:
                    # Use all annotators for prediction -- this will be the case in cross-corpus tests
                    # Can set to None and the models will handle this 
                    masks = None
                else:
                    masks = batch['annotator_masks']

                outs = model_trainer.model(audio, text, skip_kde=False, soft_hist=soft_hist, annotator_masks=masks)
                kde, mean_act_preds, mean_val_preds, eval_act_preds, eval_val_preds = outs[0], outs[1], outs[2], outs[3], outs[4]

                # First check means
                # Values will already be of shape batch_size x model_output_size
                if mean_act_preds.shape[-1] > 1 and len(mean_act_preds.shape) > 1:
                    # If outputting multiple observations per annotator then mean the observations
                    mean_act_preds = mean_act_preds.nanmean(dim=-1)
                    mean_val_preds = mean_val_preds.nanmean(dim=-1)
                mean_act_preds = mean_act_preds.squeeze()
                mean_val_preds = mean_val_preds.squeeze()
                model_results['avg_act_preds'].append(mean_act_preds.cpu())
                model_results['avg_val_preds'].append(mean_val_preds.cpu())
                model_results['eval_act_preds'].append(eval_act_preds.cpu())
                model_results['eval_val_preds'].append(eval_val_preds.cpu())
                if 'padded_individual_annotators_act' in batch:
                    model_results['padded_target_acts'].append(batch['padded_individual_annotators_act'].cpu())
                    model_results['padded_target_vals'].append(batch['padded_individual_annotators_val'].cpu())

                # For the individual annotator predictions this is best stored flattened, fortunately the task1_helper function for task1 will flatten these keeping the relative order the same
                if not use_all_annotators:
                    model_output = (kde, mean_act_preds, mean_val_preds, eval_act_preds, eval_val_preds)
                    act_preds, val_preds, act_targets, val_targets = task1_helper(model_output, batch, masks)
                    model_results['full_act_preds'].append(act_preds.cpu())
                    model_results['full_val_preds'].append(val_preds.cpu())
                    model_results['full_act_targets'].append(act_targets.cpu())
                    model_results['full_val_targets'].append(val_targets.cpu())
                if kde is None:
                    # Just make the distribution uniform 
                    kde = torch.ones((audio.shape[0],prob_grid_size,prob_grid_size), device=device)
                model_results['probability_logits'].append(kde.cpu())
                best = kde.view(kde.shape[0],-1)# - kde.view(kde.shape[0],-1).min(dim=-1).values.unsqueeze(dim=-1)
                best = best / best.sum(dim=-1).unsqueeze(dim=-1)
                best = best.view(kde.shape[0], -1)

            model_results['probability_preds'].append(best.view(audio.shape[0], prob_grid_size, prob_grid_size).cpu())

            model_results['acts'].append(y_act.cpu())
            model_results['vals'].append(y_val.cpu())
            model_results['act_bin'].append(y_act_bin.cpu())
            model_results['val_bin'].append(y_val_bin.cpu())
            model_results['probs'].append(torch.as_tensor(np.array(batch['kde_2d_probability'])).cpu())
            model_results['soft_act_labels'].append(soft_act)
            model_results['soft_val_labels'].append(soft_val)
            model_results['act_variances'].append(batch['act_variance'].cpu())
            model_results['val_variances'].append(batch['val_variance'].cpu())
            if 'act_variance_knn' in batch:
                model_results['act_knn_variances'].append(batch['act_variance_knn'].cpu())
                model_results['val_knn_variances'].append(batch['val_variance_knn'].cpu())
            if 'act_variance_grid' in batch:
                model_results['act_grid_variances'].append(batch['act_variance_grid'].cpu())
                model_results['val_grid_variances'].append(batch['val_variance_grid'].cpu())

        concat_keys = ['probability_preds', 'probability_logits', 'acts', 'vals', 'act_bin', 'val_bin', 'probs']
        pad_keys = []
        pad_2d_keys = []
        if len(model_results['avg_act_preds']):
            concat_keys.extend(['avg_act_preds', 'avg_val_preds', 'act_variances', 'val_variances'])
            pad_keys.extend(['eval_act_preds', 'eval_val_preds', 'padded_target_acts', 'padded_target_vals'])
            if 'val_log_vars' in model_results.return_values and None not in model_results['val_log_vars']:
                pad_keys.extend(['val_log_vars', 'act_log_vars', 'act_knn_variances', 'val_knn_variances', 'act_grid_variances', 'val_grid_variances'])
                pad_2d_keys.extend(['soft_act_zs', 'soft_val_zs'])
            if not use_all_annotators:
                concat_keys.extend(['full_act_preds', 'full_val_preds', 'full_act_targets', 'full_val_targets'])

        model_results.close(concat_keys, pad_keys, pad_2d_keys)

        return model_results

def validation_epoch(validation_dataloader, model_trainer, prob_grid_size, epoch, soft_hist=True, use_all_annotators=False):
    eval_results = eval_epoch(validation_dataloader, model_trainer, prob_grid_size, soft_hist, use_all_annotators)
    y_metric_inputs = {'y_act': eval_results['acts'], 'y_val': eval_results['vals'], 'y_act_bin': eval_results['act_bin'], 'y_val_bin': eval_results['val_bin'], 'y_probabilities': eval_results['probs'], 'y_soft_act_labels': eval_results['soft_act_labels'], 'y_soft_val_labels': eval_results['soft_val_labels']}
    prediction_metric_inputs = {'probability_preds': eval_results['probability_preds'].to(model_trainer.model.device)}
    if not use_all_annotators and not model_trainer.model.args.predict_dist:
        y_metric_inputs['full_annotator_act_preds'] = eval_results['full_act_preds']
        y_metric_inputs['full_annotator_val_preds'] = eval_results['full_val_preds']
        y_metric_inputs['full_act_targets'] = eval_results['full_act_targets']
        y_metric_inputs['full_val_targets'] = eval_results['full_val_targets']
        prediction_metric_inputs['full_act_preds'] = eval_results['avg_act_preds']
        prediction_metric_inputs['full_val_preds'] = eval_results['avg_val_preds']

    soft_hist_log_str = ' soft hist' if soft_hist else ''
    all_evals_str = ' all annotators enabled' if use_all_annotators else ''
    logger = Logger()
    metric_inputs = {**y_metric_inputs, **prediction_metric_inputs}
    total_variation = log_metrics(f'validation ({prob_grid_size}x{prob_grid_size}){soft_hist_log_str}{all_evals_str}', epoch, model_trainer.name, True, **metric_inputs)

    # Now log loss functions
    with torch.no_grad():
        # First need to rebuild model inputs 
        targets = {'act': eval_results['acts'].to(model_trainer.model.device), 'val': eval_results['vals'].to(model_trainer.model.device), 'kde_2d_probability': eval_results['probs'].to(model_trainer.model.device)}
        if model_trainer.model.args.predict_dist:
            model_output = eval_results['probability_logits'].to(model_trainer.model.device)
        else:
            # Last two values are none as they can't be easily tracked and so we manually calculate losses involving these values
            model_output = (eval_results['probability_logits'].to(model_trainer.model.device), eval_results['avg_act_preds'].to(model_trainer.model.device), eval_results['avg_val_preds'].to(model_trainer.model.device), eval_results['eval_act_preds'].to(model_trainer.model.device), eval_results['eval_val_preds'].to(model_trainer.model.device))
            targets['act_variance'] = eval_results['act_variances'].to(model_trainer.model.device)
            targets['val_variance'] = eval_results['val_variances'].to(model_trainer.model.device)
            if 'padded_target_acts' in eval_results.return_values:
                targets['padded_individual_annotators_act'] = eval_results['padded_target_acts'].to(model_trainer.model.device)
                targets['padded_individual_annotators_val'] = eval_results['padded_target_vals'].to(model_trainer.model.device)
        # Now need to define the targets for loss functions
        full_val_loss = 0
        loss_err = False
        use_kl_div = len(model_trainer.loss_fns) == 1 # When only task 2 is enabled, we should use this as early stopping, but ignore in other cases
        for loss_fn in model_trainer.loss_fns:
            losses = loss_fn(model_output, targets, None)
            for loss_name in list(losses.keys()):
                if losses[loss_name] is None:
                    del losses[loss_name]
                    loss_err = True
                    print('loss error', loss_name, 'set loss for early stopping to 99')
                    continue # Brent optimisation failed so skip this batch for this loss function
                l = losses[loss_name].item()
                logger.log_scalar(f'validation ({prob_grid_size}x{prob_grid_size}){soft_hist_log_str}', f'{loss_fn.name}-{loss_name}', model_trainer.name, l, epoch)
                # Don't track the kl-div loss values since they can be very large and heavily impact training
                if 'KL-Div' not in loss_name or use_kl_div:
                    # print('using', loss_name, 'in full validation loss')
                    full_val_loss += l
        if loss_err:
            full_val_loss += 99

        logger.log_scalar(f'validation ({prob_grid_size}x{prob_grid_size}){soft_hist_log_str}', f'full loss', model_trainer.name, full_val_loss, epoch)

        early_stopping_metric = full_val_loss

        logger.log_scalar(f'validation ({prob_grid_size}x{prob_grid_size}){soft_hist_log_str}', f'early stopping', model_trainer.name, early_stopping_metric, epoch)

    # Now log early stopping
    model_trainer.continue_training = model_trainer.early_stopping.log_value(early_stopping_metric, model_trainer.model)
    # if epoch < 30: # Train for a minimum of 30 warm up epochs. May end immediately after this
        # model_trainer.continue_training = True
    model_trainer.lr_scheduler.step(early_stopping_metric)
    new_lr = model_trainer.lr_scheduler.get_last_lr()
    if new_lr != model_trainer.current_lr:
        print(f'Learning rate reduced to {new_lr}')
        model_trainer.current_lr = new_lr
    if not model_trainer.continue_training:
        print(model_trainer.name, 'triggered early stopping')

def test_epoch(validation_dataloader, model_trainer, prob_grid_size, soft_hist=True, use_all_annotators=False, test_name='Test (Improv)', fixed_annotators=None, annotator_mappings=None):
    eval_results = eval_epoch(validation_dataloader, model_trainer, prob_grid_size, soft_hist, use_all_annotators)
    y_metric_inputs = {'y_act': eval_results['acts'], 'y_val': eval_results['vals'], 'y_act_bin': eval_results['act_bin'], 'y_val_bin': eval_results['val_bin'], 'y_probabilities': eval_results['probs'], 'y_soft_act_labels': eval_results['soft_act_labels'], 'y_soft_val_labels': eval_results['soft_val_labels']}
    prediction_metric_inputs = {'probability_preds': eval_results['probability_preds'].to(model_trainer.model.device)}
    if not use_all_annotators and not model_trainer.model.args.predict_dist:
        y_metric_inputs['full_annotator_act_preds'] = eval_results['full_act_preds']
        y_metric_inputs['full_annotator_val_preds'] = eval_results['full_val_preds']
        y_metric_inputs['full_act_targets'] = eval_results['full_act_targets']
        y_metric_inputs['full_val_targets'] = eval_results['full_val_targets']
        prediction_metric_inputs['full_act_preds'] = eval_results['avg_act_preds']
        prediction_metric_inputs['full_val_preds'] = eval_results['avg_val_preds']

    soft_hist_log_str = ' soft hist' if soft_hist else ''
    all_evals_str = ' all annotators enabled' if use_all_annotators else ''
    logger = Logger()
    metric_inputs = {**y_metric_inputs, **prediction_metric_inputs}
    early_stopping_metric = log_metrics(f'{test_name} ({prob_grid_size}x{prob_grid_size}){soft_hist_log_str}{all_evals_str}', 0, model_trainer.name, True, **metric_inputs)

    # Now log loss functions
    with torch.no_grad():
        # First need to rebuild model inputs 
        targets = {'act': eval_results['acts'].to(model_trainer.model.device), 'val': eval_results['vals'].to(model_trainer.model.device), 'kde_2d_probability': eval_results['probs'].to(model_trainer.model.device)}
        if model_trainer.model.args.predict_dist:
            model_output = eval_results['probability_logits'].to(model_trainer.model.device)
        else:
            model_output = (eval_results['probability_logits'].to(model_trainer.model.device), eval_results['avg_act_preds'].to(model_trainer.model.device), eval_results['avg_val_preds'].to(model_trainer.model.device), eval_results['eval_act_preds'].to(model_trainer.model.device), eval_results['eval_val_preds'].to(model_trainer.model.device))

            targets['act_variance'] = eval_results['act_variances'].to(model_trainer.model.device)
            targets['val_variance'] = eval_results['val_variances'].to(model_trainer.model.device)
            if 'padded_target_acts' in eval_results.return_values:
                targets['padded_individual_annotators_act'] = eval_results['padded_target_acts'].to(model_trainer.model.device)
                targets['padded_individual_annotators_val'] = eval_results['padded_target_vals'].to(model_trainer.model.device)
        # Now need to define the targets for loss functions
        full_val_loss = 0
        for loss_fn in model_trainer.loss_fns:
            if (loss_fn.name == 'task1' or loss_fn.name == 'task1_mse') and use_all_annotators:
                print('Skipping task 1 loss on test set when using all annotators')
                continue
            if 'task2_per_annotator' in loss_fn.name:
                print('Skipping task 2 loss on test set as not all contain individual annotator mappings')
                continue
            losses = loss_fn(model_output, targets, None)
            for loss_name in losses:
                l = losses[loss_name].item()
                logger.log_scalar(f'{test_name} ({prob_grid_size}x{prob_grid_size}){soft_hist_log_str}', f'{loss_fn.name}-{loss_name}', model_trainer.name, l, 0)
                full_val_loss += l

def cc_test_epoch(validation_dataloader, model_trainer, prob_grid_size, test_name, soft_hist=True, fixed_annotators=None, annotator_mappings=None):
    test_epoch(validation_dataloader, model_trainer, prob_grid_size, soft_hist, use_all_annotators=True, test_name=test_name, fixed_annotators=fixed_annotators, annotator_mappings=None)