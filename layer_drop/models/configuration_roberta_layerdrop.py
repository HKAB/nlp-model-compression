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
""" RoBERTa configuration """
# from collections import OrderedDict
# from typing import Mapping

# from ...onnx import OnnxConfig
from transformers.utils import logging
from transformers.configuration_utils import PretrainedConfig

logger = logging.get_logger(__name__)

ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "roberta-base": "https://huggingface.co/roberta-base/resolve/main/config.json",
    "roberta-large": "https://huggingface.co/roberta-large/resolve/main/config.json",
    "roberta-large-mnli": "https://huggingface.co/roberta-large-mnli/resolve/main/config.json",
    "distilroberta-base": "https://huggingface.co/distilroberta-base/resolve/main/config.json",
    "roberta-base-openai-detector": "https://huggingface.co/roberta-base-openai-detector/resolve/main/config.json",
    "roberta-large-openai-detector": "https://huggingface.co/roberta-large-openai-detector/resolve/main/config.json",
}


class RobertaConfig(PretrainedConfig):
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
    model_type = "roberta"

    def __init__(self, pad_token_id=1, bos_token_id=0, eos_token_id=2,
                    attention_probs_dropout_prob=0.1,
                    gradient_checkpointing=False,
                    hidden_act="gelu",
                    hidden_dropout_prob=0.1,
                    hidden_size=768,
                    initializer_range=0.02,
                    intermediate_size=3072,
                    layer_norm_eps=1e-05,
                    max_position_embeddings=258,
                    model_type="roberta",
                    num_attention_heads=12,
                    num_hidden_layers=12,
                    position_embedding_type="absolute",
                    tokenizer_class="PhobertTokenizer",
                    transformers_version="4.8.1",
                    type_vocab_size=1,
                    use_cache=True,
                    vocab_size=64001,
                    **kwargs):
        """Constructs RobertaConfig."""
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
        
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.gradient_checkpointing = gradient_checkpointing
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range
        self.intermediate_size = intermediate_size
        self.layer_norm_eps = layer_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.model_type = model_type
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.pad_token_id = pad_token_id
        self.position_embedding_type = position_embedding_type
        self.tokenizer_class = tokenizer_class
        self.transformers_version = transformers_version
        self.type_vocab_size = type_vocab_size
        self.use_cache = use_cache
        self.vocab_size = vocab_size


# class RobertaOnnxConfig(OnnxConfig):
#     @property
#     def inputs(self) -> Mapping[str, Mapping[int, str]]:
#         return OrderedDict(
#             [
#                 ("input_ids", {0: "batch", 1: "sequence"}),
#                 ("attention_mask", {0: "batch", 1: "sequence"}),
#             ]
#         )

#     @property
#     def outputs(self) -> Mapping[str, Mapping[int, str]]:
#         return OrderedDict([("last_hidden_state", {0: "batch", 1: "sequence"}), ("pooler_output", {0: "batch"})])