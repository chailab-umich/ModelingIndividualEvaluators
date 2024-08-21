import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import dataclasses
import math
from SERDatasets.kde_probability import kde_probability_bs
from AnnotatorLayer import AnnotatorOutputLayer
from .one_hot_layer import OneHotLayer

@dataclasses.dataclass
class ModelArguments:
    dense_layers: int
    layer_size: int
    prob_grid_size: int
    predict_dist: bool
    annotators: set
    training_precision: type = torch.float32
    kde_training_temperature: int = None
    kde_training_grid_size: int = None
    one_hot_annotators: bool = False
    epsilon_size: int = 200

class GenericModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.args = ModelArguments(**kwargs)
        # Predict dist -> directly predicting distribution i.e. output layer is prob_grid_size*prob_grid_size
        if not self.args.predict_dist:
            assert self.args.annotators is not None

        self.skip_random_observations = False
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.test_precision = torch.float64
        self.text_layers = [
            nn.Dropout(p=0.2),
        ]
        self.audio_layers = [
            nn.Dropout(p=0.2),
        ]

        self.audio_layers = nn.ModuleList(self.audio_layers)
        self.text_layers = nn.ModuleList(self.text_layers)

        input_size = 768+512
        if not self.args.predict_dist:
            self.number_of_annotators = len(self.args.annotators)

        if self.args.one_hot_annotators:
            input_size += self.number_of_annotators

        # In one-hot the audio and text will be concatenated at the same time as adding one-hot encoding
        self.combined_layers = [] if self.args.one_hot_annotators else [ConcatLayer()]

        # +1 to keep baseline layers the same count as the new method 
        num_fully_shared = self.args.dense_layers + 1 if self.args.predict_dist else self.args.dense_layers - 1 # -1 as there are 2 separate act/val layer at the end for non-baseline methods
        for _ in range(num_fully_shared):
            self.combined_layers.append(nn.Linear(input_size, self.args.layer_size))
            input_size = self.args.layer_size
            self.combined_layers.append(nn.ReLU())

        self.combined_layers = nn.Sequential(*self.combined_layers)        
        if self.args.predict_dist:
            self.prediction_head = nn.Linear(self.args.layer_size, self.args.prob_grid_size**2)
        else:
            self.act_combined_layers = nn.Sequential(
                nn.Linear(self.args.layer_size, self.args.layer_size),
                nn.ReLU(),
                nn.Linear(self.args.layer_size, self.args.layer_size),
                nn.ReLU(),
            )
            self.val_combined_layers = nn.Sequential(
                nn.Linear(self.args.layer_size, self.args.layer_size),
                nn.ReLU(),
                nn.Linear(self.args.layer_size, self.args.layer_size),
                nn.ReLU(),
            )
            if self.args.one_hot_annotators:
                self.one_hot_encode = OneHotLayer(set_of_annotators=self.args.annotators)
                self.act_head = nn.Linear(self.args.layer_size, 1)
                self.val_head = nn.Linear(self.args.layer_size, 1)
            else:
                self.act_heads = AnnotatorOutputLayer(input_size=256, output_size=1, set_of_annotators=self.args.annotators)
                self.val_heads = AnnotatorOutputLayer(input_size=256, output_size=1, annotator_mapper=self.act_heads.annotator_mapper)

    def forward(self, audio, text, skip_kde=False, soft_hist=False, annotator_masks=None, curr_task=None):
        for layer in self.audio_layers:
            audio = layer(audio)
        for layer in self.text_layers:
            text = layer(text)
        
        if self.args.predict_dist:
            x = self.combined_layers((audio, text))
            return self.prediction_head(x)

        if self.args.one_hot_annotators:
            x, shaped_outputs, output_mask = self.one_hot_encode(torch.cat((audio, text), dim=1), annotator_masks)
            x = self.combined_layers(x)
            act_x = self.act_combined_layers(x)
            val_x = self.val_combined_layers(x)
            act_labels = self.act_head(act_x).squeeze()
            val_labels = self.val_head(val_x).squeeze()
            print('one hot shape', act_labels.shape)
            soft_act_labels, soft_val_labels = shaped_outputs[0].to(x.device, non_blocking=True), shaped_outputs[1].to(x.device, non_blocking=True)
            soft_act_labels[output_mask] = act_labels
            print('one hot final shape', soft_act_labels.shape)
            soft_val_labels[output_mask] = val_labels
        else:
            x = self.combined_layers((audio, text))
            act_x = self.act_combined_layers(x)
            val_x = self.val_combined_layers(x)
            soft_act_labels = self.act_heads(act_x, annotator_masks).squeeze()
            soft_val_labels = self.val_heads(val_x, annotator_masks).squeeze()

        # For single prediction head and head per real annotator this should be the same processing steps
        # Get average of annotators for each sample in batch ignoring nan values
        mean_act = soft_act_labels.nanmean(dim=1)
        mean_val = soft_val_labels.nanmean(dim=1)

        if (not skip_kde and self.training) or not self.training:
            # Don't use soft histogram if during validation step
            precision = self.args.training_precision if self.training else self.test_precision
            use_soft_histogram = self.training or soft_hist
            density_grid_size = self.args.kde_training_grid_size if use_soft_histogram else 512
            kde_act_input = soft_act_labels
            kde_val_input = soft_val_labels
            kde_probs = kde_probability_bs(kde_act_input, kde_val_input, temperature=self.args.kde_training_temperature, density_grid_size=density_grid_size, prob_grid_size=self.args.prob_grid_size, use_soft_histogram=use_soft_histogram, precision=precision) # Returns logits
            return kde_probs, mean_act, mean_val, soft_act_labels, soft_val_labels
        return None, mean_act, mean_val, soft_act_labels, soft_val_labels

class ConcatLayer(nn.Module):
    def forward(self, x):
        # x should be a tuple of tensors to concatenate 
        return torch.cat(x, dim=1)