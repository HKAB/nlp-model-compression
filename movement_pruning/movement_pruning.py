import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AdamW,
    get_linear_schedule_with_warmup
)

from ..global_constants import EarlyStopping, TokenClassificationDataset

from utils import (
    schedule_threshold,
    regularization
)
import io
from torch.utils.data import DataLoader
import optparse
from ..global_constants import CUDA_VISIBLE_DEVICES, POS_TARGET, NER_TARGET, POS_PATH_TRAIN, NER_PATH_TRAIN, \
                                NER_PATH_DEV, POS_PATH_DEV, POS_PATH_TEST, NER_PATH_TEST
from constants import CHECKPOINT_PATH, RESULT_PATH, T_TOTAL, \
                        WARMUP_STEPS, FINAL_THRESHOLD, INITIAL_THRESHOLD, FINAL_WARMUP, \
                        INITIAL_WARMUP, FINAL_LAMBDA, MASKED_SCORE_LEARNING_RATE, \
                        LEARNING_RATE, REGULARIZATION
import os.path
import os
from models.modeling_masked_roberta import MaskedRobertaForTokenClassification
from models.configuration_roberta import MaskedRobertaConfigForTokenClassification

os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
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

# def loss_fn(output, target):
#   # output shape: batch_size x max_seq_length x num_classes
#   # target shape: batch_size x max_seq_length
#     output = output.permute(0, 2, 1)  
#     return nn.CrossEntropyLoss()(predictions, targets.long())


# training procedure
def train_epoch(model, data_iter, optimizer, scheduler):
    global GLOBAL_STEP
    
    model.train()
    
    threshold, regu_lambda = schedule_threshold(
                step=GLOBAL_STEP,
                total_step=T_TOTAL,
                warmup_steps=WARMUP_STEPS,
                final_threshold=FINAL_THRESHOLD,
                initial_threshold=INITIAL_THRESHOLD,
                final_warmup=FINAL_WARMUP,
                initial_warmup=INITIAL_WARMUP,
                final_lambda=FINAL_LAMBDA,
            )

    losses = []
    accuracy = []

    for data in tqdm(data_iter):
        optimizer.zero_grad()

        ids = data['ids'].to(device, non_blocking=True)
        masks = data['masks'].to(device, non_blocking=True)
        # token_type_ids = data['token_type_ids'].to(device, non_blocking=True)
        labels = data['labels'].to(device, non_blocking=True)

        output = model(ids, masks, labels=labels, threshold=threshold)
        # print(output)
        # print(loss_fn(output.logits, target))
        loss = output.loss#loss_fn(output.logits, target)
        
        if (REGULARIZATION):
            regu_ = regularization(model=model, mode=REGULARIZATION)
            loss = loss + regu_lambda * regu_

        losses.append(loss.item())
        accuracy.append(compute_metrics(output.logits.detach().cpu(), labels.detach().cpu()))

        loss.backward()
        optimizer.step()

        if scheduler:
            scheduler.step()
        GLOBAL_STEP += 1
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

        output = model(ids, masks, threshold=FINAL_THRESHOLD)

        accuracy.append(compute_metrics(output.logits.detach().cpu(), labels.detach().cpu()))

    accuracy = np.mean(accuracy)

    return accuracy

def train(model, train_iter, val_iter, optimizer, scheduler, epochs):
    
    # global_step = 1
    
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

            output = model(ids, masks, threshold=FINAL_THRESHOLD)

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
    
    parser = optparse.OptionParser()
    
    
    parser.add_option('-b', '--batch',
        action="store", dest="batch",
        help="batch size", default=32)
    
    parser.add_option('-t', '--task',
        action="store", dest="task",
        help="task: 'ner' or 'pos'", default="p")
    
    parser.add_option('-e', '--epochs',
        action="store", dest="epochs",
        help="number of epochs", default=30)
    
    parser.add_option('-m', '--mode',
        action="store", dest="mode",
        help="train/test mode", default='train')
    
    
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

        train_dataset = TokenClassificationDataset(train_task[:5], target_list, tokenizer)
        val_dataset = TokenClassificationDataset(dev_task[:5], target_list, tokenizer)

        train_iter = DataLoader(
            train_dataset,
            batch_size = batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )

        val_iter = DataLoader(
            val_dataset,
            batch_size = batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        global T_TOTAL
        T_TOTAL = len(train_iter)*epochs
        print(f'Loaded dataset, total optimization steps: {T_TOTAL}')

        num_classes = len(target_list)
#         model = AutoModelForTokenClassification.from_pretrained("vinai/phobert-base", num_labels=num_classes)
        config = MaskedRobertaConfigForTokenClassification.from_pretrained(
                "vinai/phobert-base",
                pruning_method="topK",
                mask_init="constant",
                mask_scale=0.0,
                classifier_dropout=0.0,
                num_labels=num_classes
            )

        model = MaskedRobertaForTokenClassification.from_pretrained(
                "vinai/phobert-base",
                config=config,
            )
        model.to(device)

        print('Loaded model')

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if "mask_score" in n and p.requires_grad],
                    "lr": MASKED_SCORE_LEARNING_RATE,
                },
                {
                    "params": [
                        p
                        for n, p in model.named_parameters()
                        if "mask_score" not in n and p.requires_grad and not any(nd in n for nd in no_decay)
                    ],
                    "lr": LEARNING_RATE,
        #             "weight_decay": args.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in model.named_parameters()
                        if "mask_score" not in n and p.requires_grad and any(nd in n for nd in no_decay)
                    ],
                    "lr": LEARNING_RATE,
                    "weight_decay": 0.0,
                },
            ]
        optimizer = AdamW(model.parameters(), lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=T_TOTAL
        )

        train_scores, train_losses, val_scores = train(model, train_iter, val_iter, 
                                                    optimizer, scheduler, epochs)
#         with open("results/train_scores.pkl", "wb+") as fp:
#             pickle.dump(train_scores, fp)
#         with open("results/train_losses.pkl", "wb+") as fp:
#             pickle.dump(train_losses, fp)
#         with open("results/val_scores.pkl", "wb+") as fp:
#             pickle.dump(val_scores, fp)
        np.save(RESULT_PATH + "/train_scores", train_scores)
        np.save(RESULT_PATH + "/train_losses", train_losses)
        np.save(RESULT_PATH + "/val_scores", val_scores)
    else:
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
    
        test_dataset = TokenClassificationDataset(test_task[:1], target_list, tokenizer)
        
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
        config = MaskedRobertaConfigForTokenClassification.from_pretrained(
                "vinai/phobert-base",
                pruning_method="topK",
                mask_init="constant",
                mask_scale=0.0,
                classifier_dropout=0.0,
                num_labels=num_classes
            )

        model = MaskedRobertaForTokenClassification.from_pretrained(
                "vinai/phobert-base",
                config=config,
            )
        model.to(device)
        model.load_state_dict(torch.load(CHECKPOINT_PATH))

        print('Loaded model')

        test_accuracy = evaluate_test(model, test_iter)

        print(f"Test accuracy: {test_accuracy}")
    
if __name__ == "__main__":
    main()