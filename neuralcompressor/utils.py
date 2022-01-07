#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 14:45:44 2022

@author: hkab
"""

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import os
import transformers
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
import time
import random
import torch.nn.functional as F


class PretrainedEmbedding(nn.Embedding):
    def __init__(self, *args, **kwargs):
        super(PretrainedEmbedding, self).__init__(
            *args, norm_type=2, **kwargs)
        self.vocab = {}
        self.i2w = {}
        
    def from_pretrained(self, 
                        tokenizer: transformers.PreTrainedTokenizer, 
                        embedding_matrix: nn.Embedding,
                        freeze: bool = True):
        self.weight = embedding_matrix.weight
        
        for i in range(0, len(tokenizer)):
            self.i2w[i] = tokenizer.convert_ids_to_tokens(i)
            self.vocab[tokenizer.convert_ids_to_tokens(i)] = i
        
        if freeze:
            self.weight.requires_grad = False
        return self
    
class EmbeddingCompressor(nn.Module):
    def __init__(self, embedding_dim: int, 
                         num_codebooks: int, 
                         num_vectors: int, 
                         use_gpu: bool = False):
        super(EmbeddingCompressor, self).__init__()
        self.tau = 1
        self.M = num_codebooks
        self.K = num_vectors
        
        self.use_gpu = use_gpu
        # E(w) -> h_w
        # From the paper: "In our experiments, the hidden layer h w always has a size of MK/2
        self.hidden_layer1 = nn.Sequential(
            nn.Linear(embedding_dim, 
                      num_codebooks*num_vectors//2, 
                      bias = True),
            nn.Tanh()
        )
        
        self.hidden_layer2 = nn.Linear(
            num_codebooks*num_vectors//2, num_codebooks*num_vectors, bias = True)
        # Do not initialize with torch.FloatTensor, element in the matrix can be nan
        # https://discuss.pytorch.org/t/nan-in-torch-tensor/8987
        self.codebook = nn.Parameter(torch.zeros(
            self.M*self.K, embedding_dim), requires_grad = True)
        
    def _encode(self, embeddings):
        # E(w) -> h_w [B, H] -> [B, M*K/2]
        h = self.hidden_layer1(embeddings)
        # h_w -> a_w [B, M*K]
        logits = F.softplus(self.hidden_layer2(h))
        # [B, M * K] -> [B, M, K]
        logits = logits.view(-1, self.M, self.K).contiguous()
        return logits
    
    def _decode(self, gumbel_output):
        return gumbel_output.matmul(self.codebook)
    
    def forward(self, vector):
        # Encoding [B, M, K]
        logits = self._encode(vector)
        
        # Discretization
        D = F.gumbel_softmax(
            logits.view(-1, self.K).contiguous(), tau = self.tau, hard=False)
        gumbel_output = D.view(-1, self.M*self.K).contiguous()
        maxp, _ = D.view(-1, self.M, self.K).max(dim=2)
        
        # Decode
        pred = self._decode(gumbel_output)
        return logits, maxp.data.clone().mean(), pred
    
class Trainer:
    def __init__(self, model, num_embedding, 
                 embedding_dim, model_path, lr=1e-4, 
                 use_gpu=False, batch_size=64):
        self.model = model
        self.embedding = PretrainedEmbedding(num_embedding, embedding_dim)
        self.vocab_size = 0
        self.use_gpu = use_gpu
        self._batch_size = batch_size
        self.optimizer = Adam(model.parameters(), lr=lr)
        self._model_path = model_path
    def load_pretrained_embedding(self, 
                        tokenizer: transformers.PreTrainedTokenizer, 
                        embedding_matrix: nn.Embedding,
                        freeze: bool = True):
        self.embedding = self.embedding.from_pretrained(tokenizer,
                                                       embedding_matrix,
                                                       freeze)
        self.vocab_size = len(self.embedding.vocab)
    def run(self, max_epochs=200):
        torch.manual_seed(3)
        criteration = nn.MSELoss(reduction="sum")
        assert (self.vocab_size > 0)
        valid_ids = torch.from_numpy(np.random.randint(
            0, self.vocab_size, (self._batch_size*10, ))).long()
        
        best_loss = float('inf')
        vocab_list = [x for x in range(self.vocab_size)]
        for epoch in range(max_epochs):
            self.model.train()
            time_start = time.time()
            random.shuffle(vocab_list)
            train_loss_list = []
            train_maxp_list = []
            
            for start_idx in range(0, self.vocab_size, self._batch_size):
            # for start_idx in range(0, 2, self._batch_size):
                word_ids = torch.Tensor(vocab_list[start_idx:start_idx + self._batch_size]).long()
                self.optimizer.zero_grad()
                input_embeds = self.embedding(word_ids)
                if self.use_gpu:
                    input_embeds = input_embeds.cuda()
                logits, maxp, pred = self.model(input_embeds)
                loss = criteration(pred, input_embeds).div(self._batch_size)
                train_loss_list.append(loss.data.clone().item())
                train_maxp_list.append(maxp.cpu() if self.use_gpu else maxp)
                loss.backward()
                clip_grad_norm_(self.model.parameters(), 0.001)
                self.optimizer.step()
            time_elapsed = time.time() - time_start
            train_loss = np.mean(train_loss_list)/2 # why divide by 2? mse loss
            train_maxp = np.mean(train_maxp_list)
            
            self.model.eval()
            val_loss_list = []
            val_maxp_list = []
            
            for start_idx in range(0, len(valid_ids), self._batch_size):
            # for start_idx in range(0, 2, self._batch_size):
                word_ids = valid_ids[start_idx:start_idx + self._batch_size]
                
                oracle = self.embedding(word_ids)
                if self.use_gpu:
                    oracle = oracle.cuda()
                logits, maxp, pred = self.model(oracle)
                loss = criteration(pred, oracle).div(self._batch_size)
                val_loss_list.append(loss.data.clone().item())
                val_maxp_list.append(maxp.cpu() if self.use_gpu else maxp)
                
            val_loss = np.mean(val_loss_list)/2
            val_maxp = np.mean(val_maxp_list)
            
            if train_loss < best_loss*0.99:
                best_loss = train_loss
                print("[epoch {}] train_loss={:.2f}, train_maxp={:.2f}, valid_loss={:.2f}, valid_maxp={:.2f},  bps={:.0f} ".format(
                    epoch, train_loss, train_maxp,
                    val_loss, val_maxp,
                    len(train_loss_list) / time_elapsed
                ))
        print('Training done!')
    def export(self, prefix, result_path):
        assert os.path.exists(self._model_path)
        vocab_list = list(range(self.vocab_size))
        
        codebook = dict(self.model.named_parameters())['codebook'].data
        if self.use_gpu:
            codebook = codebook.cpu()
        
        np.save(result_path + "/" + prefix + ".codebook", codebook.numpy())
    
        with open(result_path + "/" + prefix + ".codes", "w+", encoding='utf-8') as fout:
            vocab_list = list(range(self.vocab_size))
            for start_idx in tqdm(range(0, self.vocab_size, self._batch_size)):
            # for start_idx in tqdm(range(0, 2, self._batch_size)):
                word_ids = torch.Tensor(
                    vocab_list[start_idx:start_idx + self._batch_size]).long() 
                input_embeds = self.embedding(word_ids)
                if self.use_gpu:
                    input_embeds = input_embeds.cuda()
                logits = self.model._encode(input_embeds)
                _, codes = logits.max(dim=2)

                for w_id, code in zip(word_ids, codes):
                    w_id = w_id.item()
                    if self.use_gpu:
                        code = code.data.cpu().tolist()
                    else:
                        code = code.data.tolist()
                    word = self.embedding.i2w[w_id]
                    fout.write(word + "\t" + " ".join(map(str, code)) + "\n")
    def evaluate(self):
        assert os.path.exists(self._model_path)
        vocab_list = list(range(self.vocab_size))
        distances = []
        for start_idx in range(0, len(vocab_list), self._batch_size):
        # for start_idx in range(0, 2, self._batch_size):
            word_ids = torch.Tensor(
                    vocab_list[start_idx:start_idx + self._batch_size]).long()

            input_embeds = self.embedding(word_ids)
            if self.use_gpu:
                input_embeds = input_embeds.cuda()
            _, _, recontructed = self.model(input_embeds)
            
            distances.extend(np.linalg.norm((recontructed - input_embeds).data.cpu(), axis=1).tolist())

        return np.mean(distances)