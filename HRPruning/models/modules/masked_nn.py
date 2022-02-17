# coding=utf-8
# Copyright 2020-present, the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Masked Linear module: A fully connected layer that computes an adaptive binary mask on the fly.
The mask (binary or not) is computed at each forward pass and multiplied against
the weight matrix to prune a portion of the weights.
The pruned weight matrix is then multiplied against the inputs (and if necessary, the bias is added).
"""

import math

import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

from .binarizer import TopKBinarizer

class MaskedLinear(nn.Linear):
    """
    Fully Connected layer with on the fly adaptive mask.
    If needed, a score matrix is created to store the importance of each associated weight.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        mask_init: str = "constant",
        mask_scale: float = 0.0,
        pruning_method: str = "topK",
        head_split: int = -1,
        bias_mask: bool = False,
        head_pruning: bool = False,
        row_pruning: bool= True,
    ):
        """
        Args:
            in_features (`int`)
                Size of each input sample
            out_features (`int`)
                Size of each output sample
            bias (`bool`)
                If set to ``False``, the layer will not learn an additive bias.
                Default: ``True``
            mask_init (`str`)
                The initialization method for the score matrix if a score matrix is needed.
                Choices: ["constant", "uniform", "kaiming"]
                Default: ``constant``
            mask_scale (`float`)
                The initialization parameter for the chosen initialization method `mask_init`.
                Default: ``0.``
            pruning_method (`str`)
                Method to compute the mask.
                Choices: ["topK", "threshold", "sigmoied_threshold", "magnitude", "l0"]
                Default: ``topK``
        """
        super(MaskedLinear, self).__init__(in_features=in_features, out_features=out_features, bias=bias)
        assert pruning_method in ["topK", "threshold", "sigmoied_threshold", "magnitude", "l0"]
        self.pruning_method = pruning_method
        self.head_split = head_split
        self.bias_mask = bias_mask
        self.head_pruning = head_pruning
        self.row_pruning = row_pruning
        
        self.inference_mode = False
        
        self.block_pruning = False  # We will enable this when needed
        self.block_mask_scores = None  # the mask for block wise pruning
        self.threshold_block = None  # the threshold for block wise pruning

        if self.pruning_method in ["topK", "threshold", "sigmoied_threshold", "l0"]:
            self.mask_scale = mask_scale
            self.mask_init = mask_init
            self.mask_scores = nn.Parameter(torch.empty(self.weight.size()))
            self.init_mask(self.mask_scores)
        
        if self.row_pruning:
            self.mask_scores = nn.Parameter(
                torch.Tensor(
                    self.weight.size(0), 
                    1)) # row pruning has score for each row
            self.init_mask(self.mask_scores)
            self.threshold_row = nn.Parameter(torch.zeros(1) + 10.0);
        if self.head_pruning:
            self.head_mask_scores = nn.Parameter(
                torch.Tensor(
                    self.head_split,
                    1)) # row pruning has score for each row
            self.init_mask(self.head_mask_scores)
            self.threshold_head = nn.Parameter(torch.zeros(1) + 10.0);

    def init_mask(self, mask):
        if self.mask_init == "constant":
            init.constant_(mask, val=self.mask_scale)
        elif self.mask_init == "uniform":
            init.uniform_(mask, a=-self.mask_scale, b=self.mask_scale)
        elif self.mask_init == "kaiming":
            init.kaiming_uniform_(mask, a=math.sqrt(5))
            
    def get_mask(self):
        if self.head_pruning:
            mask_head = TopKBinarizer.apply(self.head_mask_scores, self.threshold_head)
        else:
            mask_head = None
        if self.row_pruning:
            mask = TopKBinarizer.apply(self.mask_scores, self.threshold_row)
        else:
            mask = None
        return mask_head, mask 
    
    def make_column_pruning(self, mask):
        self.weight = nn.Parameter(self.weight[:, mask])
        
    def make_inference_pruning(self):
        """
        Set inference mode to True. Approximate threshold for \
            row pruning when inference. Mix head_mask with mask and return the mask.

        Parameters
        ----------

        Returns
        -------
        mask : (`torch.Tensor`)
            Combine mask of head_mask and mask.

        """
        weight_size = self.weight.size()
                
        mask_head, mask = self.get_mask()
        if not self.head_pruning:
            mask_head = torch.ones_like(self.weight[:, 0])\
                            .type('torch.BoolTensor').view(-1)
        else:
            mask_head = mask_head.type('torch.BoolTensor').view(-1)
            mask_head = torch.repeat_interleave(mask_head,
                                                weight_size[0] // self.head_split)
        
        if not self.row_pruning:
            mask = torch.ones_like(self.weight[:, 0])
        mask = mask.type('torch.BoolTensor').view(-1)
        mask = torch.logical_and(mask_head, mask)
        
        self.weight = nn.Parameter(self.weight[mask, :])
        if self.bias_mask:
            self.bias = nn.Parameter(self.bias[mask, :])
        
        # we do not need those parameters!
        self.mask_scores = None
        self.head_mask_scores = None
        self.threshold_head = None
        self.threshold_row = None
        # we need this mask for some Layer O and FC2 pruning
        return mask
    
    def forward(self, input: torch.tensor, threshold: float):
        mask_head, mask = self.get_mask()
        
        weight_shape = self.weight.size()
        bias_shape = self.bias.size()
        if self.head_pruning:
            weight_thresholded = (self.weight.view(self.head_split, -1) * mask_head).view(weight_shape)
            bias_thresholded = (self.bias.view(self.head_split, -1) * mask_head).view(bias_shape)
        else:
            weight_thresholded = self.weight
            bias_thresholded = self.bias
        
        if self.row_pruning:
            weight_thresholded = mask * weight_thresholded
            if self.bias_mask:
                bias_thresholded = mask.view(self.bias.size()) * bias_thresholded
        
        return F.linear(input, weight_thresholded, bias_thresholded)
        
        
        
        
        
        