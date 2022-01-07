import torch
from transformers import AutoModel, AutoTokenizer
import os
from utils import EmbeddingCompressor, Trainer
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
    
    parser.add_option('-r', '--result_path',
                  action="store", dest="result_path", 
                  help="Result path", default=".")
    
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
    result_path = options.result_path
    
    assert os.path.exists(result_path)
    
    if (options.mode == "train"):
        compressor = EmbeddingCompressor(embedding_dim=embedding_dim, 
                                         num_codebooks=num_codebooks, 
                                         num_vectors=num_vectors)

        trainer = Trainer(compressor, 
                          num_embedding=num_embeddings, 
                          embedding_dim=embedding_dim, model_path=result_path + "/embedding_compressed.pt", lr=1e-4, 
                          use_gpu=use_gpu, batch_size=batch_size)
        
        trainer.load_pretrained_embedding(tokenizer, model.embeddings.word_embeddings)
        trainer.run(max_epochs=max_epochs)
        torch.save(trainer.model.state_dict(), result_path + "/embedding_compressed.pt")
        trainer.export("embedding_compressed")
    elif options.mode == "eval":
        compressor = EmbeddingCompressor(embedding_dim=embedding_dim, 
                                         num_codebooks=num_codebooks, 
                                         num_vectors=num_vectors)
        
        assert os.path.exists(result_path + "/embedding_compressed.pt")
        
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