import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoModelForTokenClassification
import optparse
from constants import *
import os.path
import os

os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    
    parser = optparse.OptionParser()
    
    
    parser.add_option('-n', '--name',
        action="store", dest="name",
        help="model name", default="NER_Pretrained_phoBERT")
    
    parser.add_option('-c', '--classes',
        action="store", dest="classes",
        help="number of classes", default=0)
    
    
    options, args = parser.parse_args()
    
    if (options.name):
        print(RESULT_PATH + f"/{options.name}.pt")
        assert os.path.exists(RESULT_PATH + f"/{options.name}.pt")
        
        num_classes = int(options.classes)
        model = AutoModelForTokenClassification.from_pretrained("vinai/phobert-base", 
                                                                num_labels=num_classes).to(device)
        model.load_state_dict(torch.load(RESULT_PATH + f"/{options.name}.pt", map_location=device))
        
        quantized_model = torch.quantization.quantize_dynamic(model, 
                                                              {torch.nn.Linear}, 
                                                              dtype=torch.qint8)
        torch.save(quantized_model.state_dict(), RESULT_PATH + f"/quantized_{options.name}.pt")
        print('Quantize model successfully!')
    
    
if __name__ == "__main__":
    main()