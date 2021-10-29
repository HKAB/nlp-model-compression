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
from constants import *
import os.path
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
# device = torch.device('cpu')
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
    es = EarlyStopping(patience=5, path=CHECKPOINT_PATH)

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
    
    with torch.no_grad():
        for data in tqdm(test_iter):
        # updater.zero_grad()

            ids = data['ids'].to(device, non_blocking=True)
            masks = data['masks'].to(device, non_blocking=True)
            # token_type_ids = data['token_type_ids'].to(device, non_blocking=True)
            labels = data['labels'].to(device, non_blocking=True)

            output = model(ids, masks)

            pred_labels.append(output.logits.detach().cpu())
            true_labels.append(labels.detach().cpu())

        # accuracy.append(compute_metrics(output.logits, labels))
    
#     with open("results/pred_labels.pkl", "wb+") as fp:
#         pickle.dump(pred_labels, fp)
#     with open("results/true_labels.pkl", "wb+") as fp:
#         pickle.dump(true_labels, fp)
    np.save(RESULT_PATH + "/pred_labels", [pred_label.numpy() for pred_label in pred_labels])
    np.save(RESULT_PATH + "/true_labels", [true_label.numpy() for true_label in true_labels])
    
    for pred_label, true_label in zip(pred_labels, true_labels):
        accuracy.append(compute_metrics(pred_label, true_label))

    accuracy = np.mean(accuracy)

    return accuracy

def main():
    
    helptext = """
    For training:               python main.py -m train -t [TASK] -b [BATCH] -e [EPOCH]
    For test:                   python main.py -m test -t [TASK] -b [BATCH] -e [EPOCH]
    For measure inference time: python main.py -m measure_infer -t [TASK]
        (Make sure you change the MEASURE_MODEL_PATH constant to the right model on results folder)\n
    """
    
    optparse.OptionParser.format_epilog = lambda self, formatter: self.epilog
    parser = optparse.OptionParser(epilog=helptext)
    
    parser.add_option('-b', '--batch',
        action="store", dest="batch",
        help="Batch size. Default: 32", default=32)
    
    parser.add_option('-t', '--task',
        action="store", dest="task",
        help="Type of task: 'n' (for NER) or 'p' (for POS). Default: p", default="p")
    
    parser.add_option('-e', '--epochs',
        action="store", dest="epochs",
        help="Number of epochs. Default: 30", default=30)
    
    parser.add_option('-m', '--mode',
        action="store", dest="mode",
        help="train/test/measure_infer mode. Default: train", default='train')
    
    
    options, args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=True)
    print(f'Loaded tokenizer, using {device}')
    
    if (options.mode == "train"):
        # check if it has the results directory for torch.save
        
        assert os.path.exists(RESULT_PATH)
        if (options.task == "p"):
            target_list = POS_TARGET

            with io.open(POS_PATH_TRAIN, encoding='utf-8') as f:
                train_task = f.read()
            with io.open(POS_PATH_DEV, encoding='utf-8') as f:
                dev_task = f.read()

            train_task = train_task.split('\n\n')
            dev_task = dev_task.split('\n\n')
            print('Load POS data sucessfully!')
        else:

            target_list = NER_TARGET

            with io.open(NER_PATH_TRAIN, encoding='utf-8') as f:
                train_task = f.read()
            with io.open(NER_PATH_DEV, encoding='utf-8') as f:
                dev_task = f.read()

            train_task = train_task.split('\n\n')
            dev_task = dev_task.split('\n\n')
            print('Load NER data sucessfully!')

        assert len(train_task)
        assert len(dev_task)

        train_task = train_task[:-1]
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
#         with open("results/train_scores.pkl", "wb+") as fp:
#             pickle.dump(train_scores, fp)
#         with open("results/train_losses.pkl", "wb+") as fp:
#             pickle.dump(train_losses, fp)
#         with open("results/val_scores.pkl", "wb+") as fp:
#             pickle.dump(val_scores, fp)
        np.save(RESULT_PATH + "/train_scores", train_scores)
        np.save(RESULT_PATH + "/train_losses", train_losses)
        np.save(RESULT_PATH + "/val_scores", val_scores)
    elif options.mode == "test":
        if (options.task == "p"):
            target_list = POS_TARGET
            with io.open(POS_PATH_TEST, encoding='utf-8') as f:
                test_task = f.read()
            test_task = test_task.split('\n\n')
            print('Load POS data sucessfully!')
        else:
            target_list = NER_TARGET
            with io.open(NER_PATH_TEST, encoding='utf-8') as f:
                test_task = f.read()
            test_task = test_task.split('\n\n')
            print('Load NER data sucessfully!')
        
        assert len(test_task)

        test_task = test_task[:-1]
    
        test_dataset = TokenClassificationDataset(test_task, target_list, tokenizer)
        
        batch_size = int(options.batch)
        
        test_iter = DataLoader(
            test_dataset,
            batch_size = batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        # check if pretrained model exist
        assert os.path.isfile(CHECKPOINT_PATH)
        
        num_classes = len(target_list)
        model = AutoModelForTokenClassification.from_pretrained("vinai/phobert-base",
                                                                num_labels=num_classes)
        model.to(device)
        if ('quantized' in CHECKPOINT_PATH):
          model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        model.load_state_dict(torch.load(CHECKPOINT_PATH))

        print(f'Loaded model')

        test_accuracy = evaluate_test(model, test_iter)

        print(f"Test accuracy: {test_accuracy}")
        
    elif options.mode == "measure_infer":
        
        def inference_latency(model, inputs, num_samples=100, num_warmups=100):
            with torch.no_grad():
                for _ in range(num_warmups):
                    _ = model(torch.unsqueeze(inputs['ids'].to(device), 0), 
                              torch.unsqueeze(inputs['masks'].to(device), 0))
            torch.cuda.synchronize()

            with torch.no_grad():
                stime = time.time()
                for _ in range(num_samples):
                    _ = model(torch.unsqueeze(inputs['ids'].to(device), 0), 
                              torch.unsqueeze(inputs['masks'].to(device), 0))
                    torch.cuda.synchronize()
                etime = time.time()
            elapsed_time = etime - stime

            return elapsed_time, elapsed_time/num_samples

        if (options.task == "p"):
            target_list = POS_TARGET
            with io.open(POS_PATH_TEST, encoding='utf-8') as f:
                test_task = f.read()
            test_task = test_task.split('\n\n')
            print('Load POS data sucessfully!')
        else:
            target_list = NER_TARGET
            with io.open(NER_PATH_TEST, encoding='utf-8') as f:
                test_task = f.read()
            test_task = test_task.split('\n\n')
            print('Load NER data sucessfully!')
        
        assert len(test_task)
        num_classes = len(target_list)
        model = AutoModelForTokenClassification.from_pretrained("vinai/phobert-base", 
                                                                num_labels=num_classes).to(device)
        if ('quantized' in MEASURE_MODEL_PATH):
          model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        model.load_state_dict(torch.load(MEASURE_MODEL_PATH, map_location=device))

        # we going to feed the model iteratively and then get the average result
        test_task = test_task[:1]
    
        test_dataset = TokenClassificationDataset(test_task, target_list, tokenizer)
        
        total_runtime, avg_runtime = inference_latency(model, 
                                                       test_dataset[0], 
                                                       num_samples=10, 
                                                       num_warmups=10)
        print(f'Total: {total_runtime:.2f}s \nAverage: {avg_runtime:.2f} s/sample')
        
    else:
        print("No such mode exist!")
    
if __name__ == "__main__":
    main()