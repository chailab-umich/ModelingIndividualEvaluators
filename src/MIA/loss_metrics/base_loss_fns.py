import torch
from .metrics import ccc, cross_entropy_no_softmax

def probability_loss(model_output, targets):
    if type(model_output) == tuple:
        model_output = model_output[0]
    bs = model_output.shape[0]
    model_output = torch.log_softmax(model_output.view(bs, -1), dim=-1)
    targets = targets[0].view(bs, -1)
    # return torch.nn.functional.cross_entropy(model_output, targets)
    return torch.nn.functional.cross_entropy(model_output, targets)

def probability_loss_no_softmax(model_output, targets):
    # Not sure what to do for this right now 
    # will do what didi did and replace 0 with 1e-8
    if type(model_output) == tuple:
        model_output = model_output[0]
    if model_output is None:
        return None # This means brent optimisation failed so this batch should be skipped for the probability distribution
    pre_prob_conversion = model_output
    bs = model_output.shape[0]
    model_output_prob = model_output.view(bs,-1)
    model_output_prob = (model_output_prob / model_output_prob.sum(dim=-1).unsqueeze(dim=-1)) + 1e-8
    targets = targets.view(bs, -1)
    loss = cross_entropy_no_softmax(model_output_prob, targets)

    if loss.isnan():
        print(pre_prob_conversion, model_output_prob, targets)
        raise ValueError('NaN in cross entropy loss')

    return loss

def ccc_loss(preds, true):
    return torch.tensor(1) - ccc(preds, true)
    # preds = preds.squeeze() + 1
    # true = true.squeeze() + 1
    # return (true - preds).pow(2).sum()/(true*preds).sum()

def kldiv(target_mean, target_var, mean, log_var):
    target_var[target_var<1e-6] = 1e-6 # Add small epsilon when variance in target label is 0 to prevent division by 0 # Assume a variance of 1
    # print()
    # print('input shapes', target_mean.shape, target_var.shape, mean.shape, log_var.shape)
    log_frac = log_var - torch.log(target_var)
    frac = (log_var.exp() + torch.pow(mean-target_mean,2))/(target_var)
    # print((log_frac - frac).shape, (log_frac - frac).mean(), (log_frac - frac).sum())#, log_frac, frac)
    return ((log_frac - frac)*0.5).mean()