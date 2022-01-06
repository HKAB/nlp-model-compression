import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoModelForTokenClassification
from transformers import AdamW
import os
import transformers
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
import time
import random
import torch.nn.functional as F
from utils import PretrainedEmbedding, EmbeddingCompressor, Trainer
import optparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    
    parser = optparse.OptionParser()
    
    parser.add_option('-c', '--num_codebooks',
                      action="store", dest="num_codebooks", 
                      help="Number of codebooks", default=32)
    
    parser.add_option('-v', '--vector',
                      action="store", dest="vector", 
                      help="Number of basis vector", default=16)
    
    parser.add_option('-b', '--batch',
                      action="store", dest="batch", 
                      help="Batch size", default=64)
    
    parser.add_option('-e', '--epochs',
                      action="store", dest="epochs", 
                      help="Max epochs", default=200)
    
    parser.add_option('-n', '--model_name',
                      action="store", dest="model_name", 
                      help="Huggingface language model name", default="vinai/phobert-base")
    
    parser.add_option('-m', '--mode',
                  action="store", dest="mode", 
                  help="Mode: train/eval", default="train")
    
    options, args = parser.parse_args()
    
    model = AutoModel.from_pretrained(options.model_name)
    tokenizer = AutoTokenizer.from_pretrained(options.model_name, use_fast=False)
    
    num_embeddings = len(tokenizer)
    embedding_dim = model.embeddings.word_embeddings.weight.shape[1]
    num_codebooks = int(options.num_codebooks)
    num_vectors = int(options.vector)
    batch_size = int(options.batch)
    use_gpu = torch.cuda.is_available()
    max_epochs = int(options.epochs)
    
    if (options.mode == "train"):
        compressor = EmbeddingCompressor(embedding_dim=embedding_dim, 
                                         num_codebooks=num_codebooks, 
                                         num_vectors=num_vectors)

        trainer = Trainer(compressor, 
                          num_embedding=num_embeddings, 
                          embedding_dim=embedding_dim, model_path="embedding_compressed", lr=1e-4, 
                          use_gpu=use_gpu, batch_size=batch_size)
        
        trainer.load_pretrained_embedding(tokenizer, model.embeddings.word_embeddings)
        trainer.run(max_epochs=max_epochs)
        trainer.export("embedding_compressed")
        torch.save(trainer.model.state_dict(), "embedding_compressed.pt")
    elif options.mode == "eval":
        compressor = EmbeddingCompressor(embedding_dim=embedding_dim, 
                                         num_codebooks=num_codebooks, 
                                         num_vectors=num_vectors)
        
        assert os.path.exists("embedding_compressed.pt")
        
        compressor.load_state_dict(torch.load("embedding_compressed.pt"))
        compressor.eval()
        
        trainer = Trainer(compressor, 
                          num_embedding=num_embeddings, 
                          embedding_dim=embedding_dim, model_path="embedding_compressed", lr=1e-4, 
                          use_gpu=use_gpu, batch_size=batch_size)
        trainer.load_pretrained_embedding(tokenizer, model.embeddings.word_embeddings)
        
        evaluate_distance = trainer.evaluate()
        print(f'Evaluate distance: {evaluate_distance}')
    
    
if __name__ == "__main__":
    main()