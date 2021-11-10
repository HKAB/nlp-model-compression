#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 10:41:11 2021

@author: hkab
"""

import torch
import optparse
import os
from models import RobertaForTokenClassification
from models import RobertaConfig
from constants import CUDA_VISIBLE_DEVICES
from constants import RESULT_PATH
import re
from utils import get_model_size

os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def prune_state_dict(state_dict: dict, prune_encoder_layers: list):
    """
    Prune state_dict. Inspired from https://github.com/pytorch/fairseq/issues/1667

    Parameters
    ----------
    state_dict : dict
        state_dict of full model.
    prune_encoder_layers : list
        A list of index number that we want to prune.

    Returns
    -------
    New state_dict.

    """
    new_state_dict = {}

    cur_encoder_index = 0
    for k, v in state_dict.items():
        if 'roberta.encoder.layer' in k:
            layer_number = re.search('roberta\.encoder\.layer.([0-9]+)', k)[1]
            if (int(layer_number) not in prune_encoder_layers):
                new_state_dict[k.replace(str(layer_number), str(cur_encoder_index))] = v
                if (f'roberta.encoder.layer.{layer_number}.output.LayerNorm.bias' in k):
                    cur_encoder_index += 1
        else:
            new_state_dict[k] = v
    return new_state_dict

def main():
    
    parser = optparse.OptionParser()
    
    
    parser.add_option('-f', '--file',
                        action="store", dest="file",
                        help="Full pretrained model", default="")
    
    parser.add_option('-m', '--method',
                      action="store", dest="method", 
                      help="Method: top/last", default="top")
    
    parser.add_option('-l', '--layerdrop',
                      action="store", dest="layerdrop", 
                      help="Probability for drop layer", default=0.5)
    
    parser.add_option('-c', '--classes',
                      action="store", dest="classes", 
                      help="Number of class for last layer", default=32)
    
    
    options, args = parser.parse_args()
    
    if (options.file):
        num_hidden_layers_after_prune = int((1 - float(options.layerdrop))*12)
        print(f'Prune to {num_hidden_layers_after_prune} encoder layers')
        
        num_classes = int(options.classes)
        config = RobertaConfig(num_labels=num_classes, num_hidden_layers=num_hidden_layers_after_prune)
        
        model = RobertaForTokenClassification.from_pretrained("vinai/phobert-base", config=config)
        
        assert os.path.exists(options.file)
        state_dict = torch.load(options.file, map_location=device)
        
        if (options.method == 'top'):
            prune_encoder_layers = range(num_hidden_layers_after_prune, 12)
            new_state_dict = prune_state_dict(state_dict, prune_encoder_layers)
        elif (options.method == 'last'):
            prune_encoder_layers = range(0, num_hidden_layers_after_prune)
            new_state_dict = prune_state_dict(state_dict, prune_encoder_layers)
        else:
            print('Not supported yet')
            exit(0)
        
        model.load_state_dict(new_state_dict)
        
        assert os.path.exists(RESULT_PATH)
        torch.save(model.state_dict(), RESULT_PATH + "/roberta_layerdrop_pruned.pt")
        print('Save model successfully!')
        print(f'Original file: {get_model_size(options.file)} Mb')
        print(f'After pruned file: {get_model_size(RESULT_PATH + "/roberta_layerdrop_pruned.pt")} Mb')
        
        
    
    
if __name__ == "__main__":
    main()