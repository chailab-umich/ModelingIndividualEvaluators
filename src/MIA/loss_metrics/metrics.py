import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import recall_score
from MIA.tb_logging import Logger
import time

def total_variation(inp, tar):
    return (torch.abs(inp-tar).sum(dim=[-1,-2])/2).mean()

def JSD(inp, tar):
    inp[inp==0] = 1e-8
    tar[tar==0] = 1e-8
    M = (inp + tar)/2
    kld_inp_m = inp*(torch.log(inp) - torch.log(M))
    kld_tar_m = tar*(torch.log(tar) - torch.log(M))
    return ((kld_inp_m.sum(dim=[-1,-2])+kld_tar_m.sum(dim=[-1,-2]))*0.5).mean()

def total_variation_1d(inp, tar):
    return torch.pow(inp-tar, 2).sum(dim=-1).mean()

def JSD_1d(inp, tar):
    inp[inp==0] = 1e-8
    tar[tar==0] = 1e-8
    M = (inp + tar)/2
    kld_inp_m = inp*(torch.log(inp) - torch.log(M))
    kld_tar_m = tar*(torch.log(tar) - torch.log(M))
    return ((kld_inp_m.sum(dim=-1)+kld_tar_m.sum(dim=-1))*0.5).mean()

def ccc(prediction, ground_truth):
    if len(prediction.shape) == 1:
        prediction = prediction.view(prediction.shape[0], 1)
    if len(ground_truth.shape) == 1:
        ground_truth = ground_truth.view(ground_truth.shape[0], 1)
    assert prediction.shape[-1] == 1, f'Final dimension of prediction should be of shape 1 for CCC, but found {prediction.shape=}'
    assert prediction.shape == ground_truth.shape and len(prediction.shape) == 2, f'CCC requires 2D inputs of the same shape, but found {prediction.shape=} {ground_truth.shape=}'
    mean_gt = torch.mean(ground_truth, 0)
    mean_pred = torch.mean(prediction, 0)
    var_gt = torch.var(ground_truth, 0)
    var_pred = torch.var(prediction, 0)
    v_pred = prediction - mean_pred
    v_gt = ground_truth - mean_gt
    if not torch.count_nonzero(v_pred): # Add small epsilon to prevent NaN error when predictions or gt are all the same value
        v_pred = v_pred + 1e-8
    if not torch.count_nonzero(v_gt):
        v_gt = v_gt + 1e-8
    cor = torch.sum(v_pred * v_gt) / (torch.sqrt(torch.sum(v_pred ** 2)) * torch.sqrt(torch.sum(v_gt ** 2)))
    sd_gt = torch.std(ground_truth)
    sd_pred = torch.std(prediction)
    numerator=2*cor*sd_gt*sd_pred
    denominator=var_gt+var_pred+(mean_gt-mean_pred)**2
    ccc = numerator/denominator
    return ccc

def cross_entropy_no_softmax(input, target):
    return torch.mean(-torch.sum(target * torch.log(input), 1))

class MetricResults:
    def __init__(self):
        self.results = {}
    
    def log_scalar(self, type, metric_name, model_name, value, step):
        # type and step are not important when running a one time test evaluation 
        if model_name not in self.results:
            self.results[model_name] = {}
        self.results[model_name][metric_name] = value

def log_metrics(type, step, name, log_to_tb, **values):
    assert type != 'train', f'type of logging should be validation/test set, got {type}. log_metrics is too slow to run per training batch.'
    # Get ground truth values from provided values dict for calculation
    y_act, y_val, y_act_bin, y_val_bin, y_probabilities, y_soft_act, y_soft_val = values['y_act'], values['y_val'], values['y_act_bin'], values['y_val_bin'], values['y_probabilities'], values['y_soft_act_labels'], values['y_soft_val_labels']

    # Get model output values from provided values dict
    # If baseline experiment then the model will only output a probability
    probability_preds = values['probability_preds']
    if 'full_act_preds' in values:
        full_act_preds, full_val_preds = values['full_act_preds'], values['full_val_preds']
    else:
        full_act_preds, full_val_preds = None, None # Empty list so that later we will not iterate over them when storing per-annotator metrics

    # Get singleton logger
    logger = Logger() if log_to_tb else MetricResults()

    batch_size, probability_grid_size, _ = probability_preds.shape
    # For UAR we will want to store a few values
    # 1) store UAR based on argmax of the probability distributions
    argmaxes = probability_preds.view(batch_size,-1).argmax(dim=-1)
    # Row is activation bins and column is valence bins
    row, col = (argmaxes // probability_grid_size), (argmaxes % probability_grid_size)

    # Recalculate sizes in the case of experiments where size might change 
    batch_size, probability_grid_size, _ = y_probabilities.shape
    argmaxes = y_probabilities.view(batch_size,-1).argmax(dim=-1)
    y_row, y_col = (argmaxes // probability_grid_size), (argmaxes % probability_grid_size)

    argmax_act_uar = recall_score(y_row.cpu(), row.cpu(), average='macro')
    argmax_val_uar = recall_score(y_col.cpu(), col.cpu(), average='macro')
    logger.log_scalar(type, 'Argmax Activation UAR', name, argmax_act_uar, step)
    logger.log_scalar(type, 'Argmax Valence UAR', name, argmax_val_uar, step)

    # We also should compare the Argmax UAR to the pre-binned UAR. This will be the "normal" uar
    act_uar = recall_score(y_act_bin.cpu(), row.cpu(), average='macro')
    val_uar = recall_score(y_val_bin.cpu(), col.cpu(), average='macro')
    logger.log_scalar(type, 'Activation UAR', name, act_uar, step)
    logger.log_scalar(type, 'Valence UAR', name, val_uar, step)

    # Calculate CCC
    # Calculate CCC from probabilities by multiplying along probability dimensions
    act_dimension = probability_preds.sum(dim=-1)
    val_dimension = probability_preds.sum(dim=-2)
    probability_bins = torch.linspace(-1,1,steps=act_dimension.shape[1], device=act_dimension.device)
    act_from_prob = torch.matmul(act_dimension, probability_bins)
    val_from_prob = torch.matmul(val_dimension, probability_bins)

    y_act_dimension = y_probabilities.sum(dim=-1)
    y_val_dimension = y_probabilities.sum(dim=-2)
    y_probability_bins = torch.linspace(-1,1,steps=y_act_dimension.shape[1], device=y_act_dimension.device)
    y_act_from_prob = torch.matmul(y_act_dimension, y_probability_bins)
    y_val_from_prob = torch.matmul(y_val_dimension, y_probability_bins)

    act_ccc_from_prob = ccc(act_from_prob.cpu(), y_act_from_prob.cpu())
    val_ccc_from_prob = ccc(val_from_prob.cpu(), y_val_from_prob.cpu())

    logger.log_scalar(type, 'Activation CCC From Probability Sum/Matmul', name, act_ccc_from_prob, step)
    logger.log_scalar(type, 'Valence CCC From Probability Sum/Matmul', name, val_ccc_from_prob, step)

    act_ccc_one_prob = ccc(act_from_prob.cpu(), y_act.cpu())
    val_ccc_one_prob = ccc(val_from_prob.cpu(), y_val.cpu())

    logger.log_scalar(type, 'Activation CCC', name, act_ccc_one_prob, step)
    logger.log_scalar(type, 'Valence CCC', name, val_ccc_one_prob, step)

    # We should also compare individual act/val predictions from annotators compared to the real act 
    if full_act_preds is not None: # TODO: THIS IS CURRENTLY MEAN BUT SHOULD NOT BE 
        mean_act_preds = full_act_preds#.squeeze().mean(dim=-1)
        mean_val_preds = full_val_preds#.squeeze().mean(dim=-1)
        act_ccc = ccc(mean_act_preds.cpu(), y_act.cpu())
        val_ccc = ccc(mean_val_preds.cpu(), y_val.cpu())
        logger.log_scalar(type, 'Activation CCC Annotator Mean', name, act_ccc, step)
        logger.log_scalar(type, 'Valence CCC Annotator Mean', name, val_ccc, step)

        # Also compare the mean output to the probability distribution output 
        act_ccc_to_prob = ccc(mean_act_preds.cpu(), y_act_from_prob.cpu())
        val_ccc_to_prob = ccc(mean_val_preds.cpu(), y_val_from_prob.cpu())
        logger.log_scalar(type, 'Activation CCC Annotator Mean/Probabilty', name, act_ccc_to_prob, step)
        logger.log_scalar(type, 'Valence CCC Annotator Mean/Probabilty', name, val_ccc_to_prob, step)

        # Also get the CCC of individual annotator outputs 
        if 'full_annotator_act_preds' in values:
            annotator_act_preds = values['full_annotator_act_preds']
            annotator_val_preds = values['full_annotator_val_preds']
            true_annotator_act = values['full_act_targets']
            true_annotator_val = values['full_val_targets']
            annotator_act_ccc = ccc(annotator_act_preds, true_annotator_act)
            annotator_val_ccc = ccc(annotator_val_preds, true_annotator_val)
            logger.log_scalar(type, 'Annotator Activation CCC', name, annotator_act_ccc, step)
            logger.log_scalar(type, 'Annotator Valence CCC', name, annotator_val_ccc, step)

    # Store JSD and total variation distance
    total_variation_val = total_variation(probability_preds.cpu(), y_probabilities.cpu())
    jsd = JSD(probability_preds.cpu(), y_probabilities.cpu())
    logger.log_scalar(type, 'Total Variation Distance', name, total_variation_val, step)
    logger.log_scalar(type, 'Jensen-Shannon Divergence', name, jsd, step)

    # Also worth calculating the 1d values
    total_variation_1d_act = total_variation_1d(act_dimension.cpu(), y_act_dimension.cpu())
    total_variation_1d_val = total_variation_1d(val_dimension.cpu(), y_val_dimension.cpu())
    jsd_1d_act = JSD_1d(act_dimension.cpu(), y_act_dimension.cpu())
    jsd_1d_val = JSD_1d(val_dimension.cpu(), y_val_dimension.cpu())
    logger.log_scalar(type, 'Total Variation Distance 1D Activation', name, total_variation_1d_act, step)
    logger.log_scalar(type, 'Jensen-Shannon Divergence 1D Activation', name, jsd_1d_act, step)
    logger.log_scalar(type, 'Total Variation Distance 1D Valence', name, total_variation_1d_val, step)
    logger.log_scalar(type, 'Jensen-Shannon Divergence 1D Valence', name, jsd_1d_val, step)
    if log_to_tb:
        return total_variation_val
    else:
        return logger

def get_votes_from_multiple_annotators(prob_grid_size, bin_ranges, full_act_preds, full_val_preds):
    act_votes = torch.zeros((full_act_preds.shape[0], prob_grid_size))
    val_votes = torch.zeros((full_act_preds.shape[0], prob_grid_size))
    # In the event of getting this from all annotators then the output will be batch_Size x num_evals x 1
    # We want to squeeze this, but in the event we are binning a single annotator it will be batch_size x 1 so we don't want to squeeze this
    if len(full_act_preds.shape) == 3:
        full_act_preds = full_act_preds.squeeze()
        full_val_preds = full_val_preds.squeeze()
    if len(full_act_preds.shape) == 1:
        full_act_preds = full_act_preds.unsqueeze(dim=1)
        full_val_preds = full_val_preds.unsqueeze(dim=1)
    assert len(full_val_preds.shape) == 2
    act_votes[:,0] = (full_act_preds < bin_ranges[1]).sum(dim=-1)
    val_votes[:,0] = (full_val_preds < bin_ranges[1]).sum(dim=-1)
    for i in range(1, prob_grid_size-1):
        act_votes[:,i] = ((bin_ranges[i] <= full_act_preds) & (full_act_preds < bin_ranges[i+1])).sum(dim=-1)
        val_votes[:,i] = ((bin_ranges[i] <= full_val_preds) & (full_val_preds < bin_ranges[i+1])).sum(dim=-1)
    act_votes[:,prob_grid_size-1] = (bin_ranges[-2] <= full_act_preds).sum(dim=-1)
    val_votes[:,prob_grid_size-1] = (bin_ranges[-2] <= full_val_preds).sum(dim=-1)
    act_votes = act_votes.argmax(dim=-1)
    val_votes = val_votes.argmax(dim=-1)
    return act_votes, val_votes
