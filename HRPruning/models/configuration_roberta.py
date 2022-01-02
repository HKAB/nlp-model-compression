# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" MaskedRoBERTa configuration """

from transformers.utils import logging
from transformers.configuration_utils import PretrainedConfig

logger = logging.get_logger(__name__)


class MaskedRobertaConfigForTokenClassification(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.RobertaModel` or a
    :class:`~transformers.TFRobertaModel`. It is used to instantiate a RoBERTa model according to the specified
    arguments, defining the model architecture.
    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.
    The :class:`~transformers.RobertaConfig` class directly inherits :class:`~transformers.BertConfig`. It reuses the
    same defaults. Please check the parent class for more information.
    Examples::
        >>> from transformers import RobertaConfig, RobertaModel
        >>> # Initializing a RoBERTa configuration
        >>> configuration = RobertaConfig()
        >>> # Initializing a model from the configuration
        >>> model = RobertaModel(configuration)
        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = "masked_roberta"

    def __init__(
        self,   
        attention_probs_dropout_prob = 0.1,
        bos_token_id = 0,
        eos_token_id = 2,
        hidden_act = "gelu",
        hidden_dropout_prob = 0.1,
        hidden_size =  768,
        initializer_range = 0.02,
        intermediate_size = 3072,
        layer_norm_eps = 1e-05,
        max_position_embeddings = 514,
        num_attention_heads = 12,
        num_hidden_layers = 12,
        pad_token_id = 1,
        type_vocab_size = 1,
        vocab_size = 50265,
        num_labels = 0,
        classifier_dropout = 0.0,
        pruning_method = "topK",
        mask_init = "constant",
        mask_scale = 0.0,
        head_pruning = True,
        **kwargs
    ):
        """Constructs RobertaConfig."""
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
        
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range
        self.intermediate_size = intermediate_size
        self.layer_norm_eps = layer_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.pad_token_id = pad_token_id
        self.type_vocab_size = type_vocab_size
        self.vocab_size = vocab_size
        self.num_labels = 0
        self.classifier_dropout = 0.0
        self.pruning_method = pruning_method
        self.mask_init = mask_init
        self.mask_scale = mask_scale
        self.head_pruning = head_pruning,