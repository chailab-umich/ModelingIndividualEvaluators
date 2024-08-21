import torch
from .base_loss_fns import ccc_loss, probability_loss_no_softmax, probability_loss, kldiv

class LossFunction: # None means these features are disabled for this task
    def __init__(self, name, calculate_kde=False, sparsity=None, after_warmup=False, validation_only=False):
        self.name = name
        self.sparsity = sparsity
        self.after_warmup = after_warmup
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.calculate_kde = calculate_kde
        self.validation_only = validation_only

# Task 1 and Task 1 CCC use almost the exact same code so configure a helper to prevent code repetition
def task1_helper(model_output, targets, masks):
    ind_evals = targets['padded_individual_annotators_act']
    ind_val_evals = targets['padded_individual_annotators_val']

    _, _, _, eval_act_preds, eval_val_preds = model_output
    if len(eval_act_preds.shape) > 2:
        # Mean and squeeze the middle dimension (in case the model output is more than one observation per annotator)
        eval_act_preds = eval_act_preds.nanmean(dim=-1).squeeze()
        eval_val_preds = eval_val_preds.nanmean(dim=-1).squeeze()

    # Model output soft labels will already be lined up with the target labels thanks to batch_collator
    # Now we have target and labels of shape batch_size x num_annotators with nan where samples have no annotators
    # so we just want to flatten these lists into the non-nan values
    m1 = ~eval_act_preds.isnan() #, ~eval_val_preds.isnan(), ~soft_act_labels.isnan(), ~soft_val_labels.isnan()
    act_preds = eval_act_preds[m1]
    val_preds = eval_val_preds[m1]
    act_targets = ind_evals[m1]
    val_targets = ind_val_evals[m1]
    return act_preds, val_preds, act_targets, val_targets

class BaselineLoss(LossFunction):
    def __call__(self, model_output, targets, masks):
        loss = probability_loss(model_output, (targets['kde_2d_probability'],))
        return {'Cross-entropy Loss': loss}

class Task1Loss(LossFunction):
    def __init__(self, *args, **kwargs):
        if kwargs['loss_fn'] == 'CCC':
            self.loss_fn = ccc_loss
        elif kwargs['loss_fn'] == 'MSE':
            self.loss_fn = torch.nn.functional.mse_loss
        else:
            raise ValueError(f'Unknown loss type {kwargs["loss_fn"]}')
        self.loss_string = kwargs['loss_fn']
        del kwargs['loss_fn']
        super().__init__(*args, **kwargs)

    def __call__(self, model_output, targets, masks, helper_outputs=None):
        if helper_outputs is None: # For eval it makes sense to track the task1_helper outputs rather than the inputs to task1_helper so allow this to be overridden
            act_preds, val_preds, act_targets, val_targets = task1_helper(model_output, targets, masks)
        else:
            act_preds, val_preds, act_targets, val_targets = helper_outputs

        act_loss = self.loss_fn(act_preds, act_targets)
        val_loss = self.loss_fn(val_preds, val_targets)

        return {f'Individual Annotator Activation {self.loss_string} Loss': act_loss, f'Individual Annotator Valence {self.loss_string} Loss': val_loss}

class Task2Loss(LossFunction):
    def __call__(self, model_output, targets, masks):
        act = targets['act']
        val = targets['val']

        _, mean_act_preds, mean_val_preds, _, _ = model_output

        # Values will already be of shape batch_size x model_output_size
        if len(mean_act_preds.shape) > 1 and mean_act_preds.shape[-1] > 1:
            # If outputting multiple observations per annotator then mean the observations
            mean_act_preds = mean_act_preds.nanmean(dim=-1)
            mean_val_preds = mean_val_preds.nanmean(dim=-1)

        mean_act_preds = mean_act_preds.squeeze()
        mean_val_preds = mean_val_preds.squeeze()

        act_loss = ccc_loss(mean_act_preds, act)
        val_loss = ccc_loss(mean_val_preds, val)

        return {'Mean Activation CCC Loss': act_loss, 'Mean Valence CCC Loss': val_loss}

class Task3Loss(LossFunction):
    def __call__(self, model_output, targets, masks):
        target_probs = targets['kde_2d_probability']

        kde, _, _, _, _ = model_output

        ce_loss = probability_loss_no_softmax(kde, target_probs)

        return {'Cross-entropy Loss': ce_loss}
