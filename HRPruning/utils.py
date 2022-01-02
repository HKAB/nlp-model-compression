import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
        
def schedule_threshold(
    step: int,
    total_step: int,
    warmup_steps: int,
    initial_threshold: float,
    final_threshold: float,
    initial_warmup: int,
    final_warmup: int,
    final_lambda: float,
):
    if step <= initial_warmup * warmup_steps:
        threshold = initial_threshold
    elif step > (total_step - final_warmup * warmup_steps):
        threshold = final_threshold
    else:
        spars_warmup_steps = initial_warmup * warmup_steps
        spars_schedu_steps = (final_warmup + initial_warmup) * warmup_steps
        mul_coeff = 1 - (step - spars_warmup_steps) / (total_step - spars_schedu_steps)
        threshold = final_threshold + (initial_threshold - final_threshold) * (mul_coeff ** 3)
    regu_lambda = final_lambda * threshold / final_threshold
    return threshold, regu_lambda


def regularization(model: nn.Module, mode: str):
    regu, counter = 0, 0
    for name, param in model.named_parameters():
        if "mask_scores" in name:
            if mode == "l1":
                regu += torch.norm(torch.sigmoid(param), p=1) / param.numel()
            elif mode == "l0":
                regu += torch.sigmoid(param - 2 / 3 * np.log(0.1 / 1.1)).sum() / param.numel()
            else:
                ValueError("Don't know this mode.")
            counter += 1
    return regu / counter

class TokenClassificationDataset(Dataset):
    """
    text_data's line has the form: WORD\tLABEL
    label_map is a list containing of all target label (e.g ["N", "CH"])

    """
    def __init__(self, text_data, label_map, tokenizer):
        super().__init__()
        self.sentences = []
        self.labels = []
        self.pad_token_label_id = -100
        self.max_seq_length = 256
        self.tokenizer = tokenizer


        for sent in text_data:
            t_label, t_sent = [], ""
            for line in sent.split("\n"):
                t_label.append(label_map.index(line.split("\t")[1]))
                t_sent += line.split("\t")[0] + " "
            self.sentences.append(t_sent.strip())
            self.labels.append(t_label)

    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):

        sentence = self.sentences[idx]
        labels = self.labels[idx]

        tokens = []
        label_ids = []

        # mimic the feature of fast tokenizer in a dumb way
        offset_mapping = [-1] # CLS
        for w, tag in zip(sentence.split(" "), labels):
            word_tokens = self.tokenizer.tokenize(w)
            # -2 for [cls] and end token
            offset_mapping += (list(range(len(word_tokens) - 2)))

            # bert-base-multilingual-cased sometimes output nothing ([]) when calling tokenize with just a space.
            if (len(word_tokens) > 0):
                tokens.extend(word_tokens)
                label_ids.extend([tag] + [self.pad_token_label_id] * (len(word_tokens) - 1))

        # add pad & end token
        offset_mapping += ([-1]*(256 - len(offset_mapping)))
        sent_encode_target = np.ones(len(offset_mapping),dtype=int) * -100
        offset_mapping = np.array(offset_mapping)
        
        # remap class for first token, others token but first will have class -100
        idx = 0
        for i in range(len(sent_encode_target)):
            if (offset_mapping[i] == 0):
                sent_encode_target[i] = labels[idx]
                idx += 1
            

        special_tokens_count = self.tokenizer.num_special_tokens_to_add()
        if len(tokens) > self.max_seq_length - special_tokens_count:
            tokens = tokens[: (self.max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (self.max_seq_length - special_tokens_count)]
        
        # add sep token and its label at the tail
        tokens += ["</s>"]
        label_ids += [self.pad_token_label_id]

        # add [cls] token at the beginning
        tokens = ['<s>'] + tokens
        label_ids = [self.pad_token_label_id] + label_ids

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # padding
        input_mask = [1] * len(input_ids)
        padding_length = self.max_seq_length - len(input_ids)
        input_ids += [1] * padding_length # pad input_ids with 1 token (pad token)
        input_mask += [0] * padding_length
        label_ids += [self.pad_token_label_id] * padding_length

        # {input_ids, token_type_ids, attention_mask}
        # ids = torch.tensor(phobert_sents['input_ids'], dtype=torch.long)
        # masks = torch.tensor(phobert_sents['attention_mask'], dtype=torch.long)
        # token_type_ids = torch.tensor(phobert_sents['token_type_ids'], dtype=torch.long)

        ids = torch.tensor(input_ids, dtype=torch.long)
        masks = torch.tensor(input_mask, dtype=torch.long)
        label_ids = torch.tensor(label_ids, dtype=torch.long)
        # token_type_ids = torch.tensor(phobert_sents['token_type_ids'], dtype=torch.long)
        
        return {
            'ids': ids,
            'masks': masks,
            # 'token_type_ids': token_type_ids,
            'labels': label_ids,
            # 'offset_mapping': torch.tensor(offset_mapping, dtype=torch.long),
        }
    
class EarlyStopping:
    def __init__(self, patience=7, delta=0., path='model_checkpoint.pt'):
        self.patience = patience
        self.delta = delta
        self.path = path
        
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        
        self.val_loss_min = np.Inf
    def __call__(self, val_loss, model):
        
        # change the sign if we want larger score
        score  = val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score > self.best_score:# + self.delta:
            self.counter += 1
            if (self.counter > self.patience):
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(val_loss, model)
    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.path)
        print(f'Validation score decrease {self.val_loss_min} -> {val_loss}. Save model..')
        self.val_loss_min = val_loss