import numpy as np
from tqdm import tqdm
import torch
import torch.nn.utils.prune as prune
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoModelForTokenClassification
from transformers import AdamW
from utils import EarlyStopping, TokenClassificationDataset
import io
from torch.utils.data import DataLoader
import optparse
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
  predictions shape: batch_size x max_seq_length x num_classes
  targets shape: batch_size x seq_length x 1

  example: [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]], [1, 2, -100]
           Return => [1, 1], [1, 2] (ignore -100 class)
"""
def align_prediction(predictions, targets):
    preds = torch.argmax(predictions, axis=2)

    batch_size, seq_len = preds.shape

    true_target_list = [[] for _ in range(batch_size)]
    preds_target_list = [[] for _ in range(batch_size)]

    for i in range(batch_size):
        for j in range(seq_len):
            if (targets[i, j] != nn.CrossEntropyLoss().ignore_index): # -100
                true_target_list[i].append(targets[i, j])
                preds_target_list[i].append(preds[i, j])
    return preds_target_list, true_target_list
  
"""
  predictions shape: batch_size x max_seq_length x num_classes
  targets shape: batch_size x seq_length x 1
"""
def compute_metrics(predictions, targets):
    preds_target_list, true_target_list = align_prediction(predictions, targets)

    results = []
    for preds_target, true_target in zip(preds_target_list, true_target_list):
        results.append((np.array(preds_target) == np.array(true_target)).mean())

    return torch.tensor(results).mean().float()

def loss_fn(output, target):
  # output shape: batch_size x max_seq_length x num_classes
  # target shape: batch_size x max_seq_length
    output = output.permute(0, 2, 1)  
    return nn.CrossEntropyLoss()(predictions, targets.long())


# training procedure
def train_epoch(model, data_iter, optimizer, scheduler):
    model.train()

    losses = []
    accuracy = []

    for data in tqdm(data_iter):
        optimizer.zero_grad()

        ids = data['ids'].to(device, non_blocking=True)
        masks = data['masks'].to(device, non_blocking=True)
        # token_type_ids = data['token_type_ids'].to(device, non_blocking=True)
        labels = data['labels'].to(device, non_blocking=True)

        output = model(ids, masks, labels=labels)
        # print(output)
        # print(loss_fn(output.logits, target))
        loss = output.loss#loss_fn(output.logits, target)

        losses.append(loss.item())
        accuracy.append(compute_metrics(output.logits.detach().cpu(), labels.detach().cpu()))

        loss.backward()
        optimizer.step()

        if scheduler:
            scheduler.step()
    losses = np.mean(losses)
    accuracy = np.mean(accuracy)

    return accuracy, losses

def evaluate_epoch(model, data_iter):
    model.eval()

    accuracy = []

    for data in data_iter:
        # updater.zero_grad()

        ids = data['ids'].to(device, non_blocking=True)
        masks = data['masks'].to(device, non_blocking=True)
        # token_type_ids = data['token_type_ids'].to(device, non_blocking=True)
        labels = data['labels'].to(device, non_blocking=True)

        output = model(ids, masks)

        accuracy.append(compute_metrics(output.logits.detach().cpu(), labels.detach().cpu()))

    accuracy = np.mean(accuracy)

    return accuracy

def train(model, train_iter, val_iter, optimizer, scheduler, epochs):
    train_scores, train_losses = [], []
    val_scores = []
    # best_score = None
    es = EarlyStopping(patience=5, path='results/checkpoint.pt')

    for epoch in range(epochs):
        train_score, train_loss = train_epoch(model, train_iter, optimizer, scheduler)
        train_scores.append(train_score)
        train_losses.append(train_loss)

        val_score = evaluate_epoch(model, val_iter)
        val_scores.append(val_score)

        es(val_score, model)

        print(f'''Epoch {epoch},
            train loss: {train_loss:.2f}, train score: {train_score:.2f}, val score: {val_score:.2f}''')

        if (es.early_stop):
            print('Early stopping')
            break
      
    return train_scores, train_losses, val_scores

def evaluate_test(model, test_iter):
    model.eval()

    pred_labels = []
    true_labels = []
    accuracy = []

    for data in test_iter:
    # updater.zero_grad()

        ids = data['ids'].to(device, non_blocking=True)
        masks = data['masks'].to(device, non_blocking=True)
        # token_type_ids = data['token_type_ids'].to(device, non_blocking=True)
        labels = data['labels'].to(device, non_blocking=True)

        output = model(ids, masks)

        pred_labels.append(output.logits)
        true_labels.append(labels)

        # accuracy.append(compute_metrics(output.logits, labels))
    
    with open("results/pred_labels.txt", "wb+") as fp:
        pickle.dump(pred_labels, fp)
    with open("results/true_labels.txt", "wb+") as fp:
        pickle.dump(true_labels, fp)
    
    for pred_label, true_label in zip(pred_labels, true_labels):
        accuracy.append(compute_metrics(pred_label, true_label))

    accuracy = np.mean(accuracy)

    return accuracy

def main():
    
    parser = optparse.OptionParser()
    
    
    parser.add_option('-b', '--batch',
        action="store", dest="batch",
        help="batch size", default=32)
    
    parser.add_option('-t', '--task',
        action="store", dest="task",
        help="task: 'ner' or 'pos'", default="pos")
    
    parser.add_option('-e', '--epochs',
        action="store", dest="epochs",
        help="number of epochs", default=30)
    
    
    options, args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=True)
    print(f'Loaded tokenizer, using {device}')
    
    if (options.task == "pos"):
        POS_PATH = 'data/POS_data/POS_data'

        target_list = ["N", "Np", "CH", "M", "R", "A", "P", "V", "Nc", "E", "L", "C", "Ny", 
                    "T", "Nb", "Y", "Nu", "Cc", "Vb", "I", "X", "Z", "B", "Eb", "Vy", 
                    "Cb", "Mb", "Pb", "Ab", "Ni", "Xy", "NY"]

        with io.open(POS_PATH + '/VLSP2013_POS_train.txt', encoding='utf-8') as f:
            train_task = f.read()
        with io.open(POS_PATH + '/VLSP2013_POS_test.txt', encoding='utf-8') as f:
            test_task = f.read()
        with io.open(POS_PATH + '/VLSP2013_POS_dev.txt', encoding='utf-8') as f:
            dev_task = f.read()

        train_task = train_task.split('\n\n')
        test_task = test_task.split('\n\n')
        dev_task = dev_task.split('\n\n')
        print('Load POS data sucessfully!')
    else:
        NER_PATH = 'data/NER_data'
                      
        target_list = [
            "O",       # Outside of a named entity
            "B-MISC",  # Beginning of a miscellaneous entity right after another miscellaneous entity
            "I-MISC",  # Miscellaneous entity
            "B-PER",   # Beginning of a person's name right after another person's name
            "I-PER",   # Person's name
            "B-ORG",   # Beginning of an organisation right after another organisation
            "I-ORG",   # Organisation
            "B-LOC",   # Beginning of a location right after another location
            "I-LOC"    # Location
        ]
        
        with io.open(NER_PATH + '/train.txt', encoding='utf-8') as f:
            train_task = f.read()
        with io.open(NER_PATH + '/test.txt', encoding='utf-8') as f:
            test_task = f.read()
        with io.open(NER_PATH + '/dev.txt', encoding='utf-8') as f:
            dev_task = f.read()
        
        train_task = train_task.split('\n\n')
        test_task = test_task.split('\n\n')
        dev_task = dev_task.split('\n\n')
        print('Load NER data sucessfully!')
                      
    assert len(train_task)
    assert len(test_task)
    assert len(dev_task)
    
    train_task = train_task[:-1]
    test_task = test_task[:-1]
    dev_task = dev_task[:-1]
    
#     predictions = torch.randint(low=0, high=9, size=(14, 256, 9))
#     targets = torch.randint(low=0, high=9, size=(14, 256, 1))
#     assert compute_metrics(predictions, targets)
    
    # same as in the paper
    epochs = int(options.epochs)
    lr = 1e-5
    # change this when real train
    batch_size = int(options.batch)

    train_dataset = TokenClassificationDataset(train_task, target_list, tokenizer)
    val_dataset = TokenClassificationDataset(dev_task, target_list, tokenizer)
    
    train_iter = DataLoader(
        train_dataset,
        batch_size = batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_iter = DataLoader(
        val_dataset,
        batch_size = batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    print(f'Loaded dataset')
    
    num_classes = len(target_list)
    model = AutoModelForTokenClassification.from_pretrained("vinai/phobert-base", num_labels=num_classes)
    model.to(device)
    
    print(f'Loaded model')
    
    optimizer = AdamW(model.parameters(), lr)    
    train_scores, train_losses, val_scores = train(model, train_iter, val_iter, 
                                                optimizer, None, epochs)
    
    test_dataset = TokenClassificationDataset(test_task[:], target_list, tokenizer)

    test_iter = DataLoader(
        test_dataset,
        batch_size = batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    test_accuracy = evaluate_test(model, test_iter)
    
    print(f"Test accuracy: {test_accuracy}")
    
if __name__ == "__main__":
    main()