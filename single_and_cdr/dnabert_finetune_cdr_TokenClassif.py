import torch 
import torch.nn as nn
import transformers
import datasets
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup
import argparse
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
import pandas as pd
import sys
from src.data_vdj_cdr import *
from src.model_hg import *
from src.graphic import *
import json
import torch.optim as optim
import torch.nn.functional as F

#from data import TranslationDataset
from transformers import BertTokenizerFast, BertTokenizer
from transformers import BertModel, BertForMaskedLM, BertConfig, EncoderDecoderModel, BertLMHeadModel, AutoModelForSequenceClassification
from sklearn.metrics import roc_auc_score

import sys
import torch
import torch.utils.data as data
from torch.nn.utils.rnn import pad_sequence
import os
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertOnlyMLMHead, SequenceClassifierOutput
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
from typing import List, Optional, Tuple, Union, Any
from transformers.modeling_outputs import ModelOutput
from transformers import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.modeling_utils import PreTrainedModel
#from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
import warnings
from tqdm.auto import tqdm
import evaluate

import matplotlib.pyplot as plt 




def main() :
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--train_dir",
        default=None,
        type=str,
        required=True,
        help="The train data dir. Should contain the .fasta files (or other data files) for the task.",
    )
    parser.add_argument(
        "--kmer",
        default=3,
        type=int,
        help="Determine which dnabert model to load " ,
    )
    parser.add_argument(
        "--batch_size",
        default=16,
        type=int,
        help="batch_size" ,
    )
    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="weight decay" ,
    )
    parser.add_argument(
        "--lr",
        default=0.00001,
        type=float,
        help="learning rate" ,
    )
    parser.add_argument(
        "--epoch",
        default=10, 
        type=int, 
        help="number of epoch",
    )
    parser.add_argument("--freeze_embedding", action="store_true", help="Whether to run training.")
    parser.add_argument(
        "--nb_classes", 
        default=70, 
        type=int, 
        help="number of classes",
    )
    parser.add_argument(
        "--nb_seq_max", 
        default=10000, 
        type=int, 
        help="number of maximum sequences"
    )
    parser.add_argument(
        "--allele", 
        action="store_true",
        help="Whether to consider allele or not."
    )
    parser.add_argument(
        "--save_model", 
        action="store_true", 
        help="Whether to save the model."
    )
    parser.add_argument(
        "--save_name", 
        type=str, 
        default="model_save.pt", 
        help="name of the output save file"
    )
    parser.add_argument(
        "--max_len", 
        default="max", 
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--type", 
        type=str,
        default='V',
        help="Which type of gene to identify : V, D or J"
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="follow run on wandb or not."
    )
    args = parser.parse_args()
    num_labels = args.nb_classes

    if args.wandb :
        # wandb
        import wandb
        wandb.init(project="cdr_detection")
    
    if args.allele :
        allele = True
    else :
        allele = False

    if args.type == 'V' :
        index_class = 1
        rev=False
    elif args.type == 'D' :
        index_class = 2
        rev=False
    elif args.type == 'J' :
        index_class = 3
        #rev = True
        rev = False

    device = torch.device('cuda')
    model = DNABertForSequenceAndTokenClassification.from_pretrained(f"zhihan1996/DNA_bert_{args.kmer}",num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(f"zhihan1996/DNA_bert_{args.kmer}", padding=True, trust_remote_code=True)
    model = model.to(device)


    # train/validation split
    dataset_train = VDJDataset(args.train_dir, tokenizer, device, target_V=None, target_D=None, target_J=None, kmer=args.kmer, allele=allele, nb_seq=args.nb_seq_max, max_len=args.max_len, rev=rev) 
    train_size = int(0.8 * len(dataset_train))
    test_size = len(dataset_train) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset_train, [train_size, test_size])

    def genes_collate_function_ext(batch):

        (seqs, vs, ds, js, va, da, ja, cdr3, head) = zip(*batch)

        raw_seqs = deepcopy(seqs)

        seqs = tokenizer(list(seqs),padding=True, truncation=True, add_special_tokens=True)
        seqs = {k: torch.tensor(v).to(device) for k, v in seqs.items()}#default_collate(peptide)
        
        vs =  default_collate(vs).to(device)
        ds =  default_collate(ds).to(device)
        js =  default_collate(js).to(device)

        va =  default_collate(va).to(device)
        da =  default_collate(da).to(device)
        ja =  default_collate(ja).to(device)

        # resize cdr3 tensor
        # get max length
        max_l = max([x.squeeze().numel() for x in cdr3])
        # pad 
        cdr3 = [torch.nn.functional.pad(x, pad=(0, max_l-x.numel()), mode='constant', value=0) for x in cdr3]

        cdr3 = tuple(cdr3)

        cdr3 = default_collate(cdr3).to(device)

        return seqs, vs, ds, js, va, da, ja, raw_seqs, cdr3, head
    
    # train
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=genes_collate_function_ext)
    # validation
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=genes_collate_function_ext)

    # loss and err history for cls classification
    y_loss = {} 
    y_loss['train'] = []
    y_loss['val'] = []
    y_err = {}
    y_err['train'] = []
    y_err['val'] = []
    # loss and err history for token classification
    y_loss_token = {}
    y_loss_token['train'] = []
    y_loss_token['val'] = []
    y_err_token = {}
    y_err_token['train'] = []
    y_err_token['val'] = []
    # trace epoch
    x_epoch = []

    # wandb track
    wandb_dict = {}


    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 20, 300)
    progress_bar = tqdm(range(args.epoch))


    for epoch in range(args.epoch) :
        x_epoch.append(epoch)
        model.train()

        for batch in train_dataloader :
            outputs = model(**batch[0], labels=batch[index_class], Tokenlabels=batch[8])
            # separate outputs
            cls_output = outputs[0]
            token_output = outputs[1]
            # 2 loss
            loss = cls_output.loss
            token_loss = token_output.loss
            loss.backward(retain_graph=True)
            token_loss.backward(retain_graph=True)
            # optimizer
            optimizer.step()
            optimizer.zero_grad()
        progress_bar.update(1)
        scheduler.step()

        if epoch%5 == 0 :
            model.eval()

            # accuracy
            for dataset in [train_dataloader, test_dataloader] :
                metric = evaluate.load("accuracy")
                token_metric = []
                for batch in dataset :
                    with torch.no_grad() :
                        outputs = model(**batch[0], labels=batch[index_class], Tokenlabels=batch[8])
                    # seperate outputs
                    cls_output = outputs[0]
                    token_output = outputs[1]
                    # logits
                    logits = cls_output.logits
                    token_logits = token_output.logits
                    # accuracy for cls
                    predictions = torch.argmax(logits, dim=-1)
                    metric.add_batch(predictions=predictions, references=batch[index_class])
                    # accuracy for token
                    token_predictions = torch.argmax(token_logits, dim=-1)
                    for pred, ref in zip(token_predictions, batch[8]) :
                        token_metric.append(torch.eq(pred,ref).sum()/pred.shape[0])
                token_metric = [value.cpu().item() for value in token_metric]
                token_metric = np.sum(token_metric)/len(token_metric)
                epoch_acc = metric.compute()['accuracy']
                epoch_token_acc = token_metric
                if dataset == train_dataloader :
                    y_err['train'].append(epoch_acc)
                    y_err_token['train'].append(epoch_token_acc)
                else :
                    y_err['val'].append(epoch_acc)
                    y_err_token['val'].append(epoch_token_acc)

            # loss
            for dataset in [train_dataloader, test_dataloader] :
                running_loss, running_loss_token = 0.0, 0.0
                for batch in dataset : 
                    with torch.no_grad() :
                        outputs = model(**batch[0], labels=batch[index_class], Tokenlabels=batch[8])
                    cls_output, token_output = outputs[0], outputs[1]
                    loss = cls_output.loss
                    token_loss = token_output.loss
                    running_loss += loss.item()
                    running_loss_token += token_loss.item()
                epoch_loss = running_loss/len(dataset)
                epoch_loss_token = running_loss_token/len(dataset)
                if dataset == train_dataloader :
                    y_loss['train'].append(epoch_loss)
                    y_loss_token['train'].append(epoch_loss_token)
                else :
                    y_loss['val'].append(epoch_loss)
                    y_loss_token['val'].append(epoch_loss_token)

            # wandb
            wandb_dict['epoch'] = epoch
            # CLS
            wandb_dict['train_acc'] = y_err['train'][-1]
            wandb_dict['train_loss'] = y_loss['train'][-1]
            wandb_dict['test_acc'] = y_err['val'][-1]
            wandb_dict['test_loss'] = y_loss['val'][-1]
            # Token
            wandb_dict['token_train_acc'] = y_err_token['train'][-1]
            wandb_dict['token_train_loss'] = y_loss_token['train'][-1]
            wandb_dict['token_test_acc'] = y_err_token['val'][-1]
            wandb_dict['token_test_loss'] = y_loss_token['val'][-1]
            if args.wandb :
                wandb.log(wandb_dict)

        print(
            f"Epoch {epoch}:\n",
            f"CLS : Train : Loss:{y_loss['train'][-1]}, Accuracy : {y_err['train'][-1]} \n \t Validation : Loss : {y_loss['val'][-1]}, Accuracy : {y_err['val'][-1]}",
            f"Token : Train : Loss:{y_loss_token['train'][-1]}, Accuracy : {y_err_token['train'][-1]} \n \t Validation : Loss : {y_loss_token['val'][-1]}, Accuracy : {y_err_token['val'][-1]}"
        )

        


        # save the best model
        acc_ref = 0
        if args.save_model :
            # best model criteria is based on test accuracy
            if y_err['val'][-1] > acc_ref :
                acc_ref = y_err['val'][-1]
                torch.save(model, args.save_name)

    draw_curve(x_epoch, y_loss, y_err, 'curves.png')
    draw_curve(x_epoch, y_loss_token, y_err_token, 'curve_token.png')


"""    for batch in test_dataloader :
        with torch.no_grad() :
            outputs = model(**batch[0])

        logits1 = outputs[0].logits
        logits2 = outputs[1].logits

        print(logits1[0], logits2[0])
        print(len(logits2[0]), len(logits2[0]))"""

if __name__ == '__main__' :
    main()