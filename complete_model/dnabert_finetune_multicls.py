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
from src.data_vdj_multicls import *
from src.model_hg_multicls import *
from src.graphic import *
import json
import torch.optim as optim
import torch.nn.functional as F

#from data import TranslationDataset
from transformers import BertTokenizerFast, BertTokenizer
from transformers import BertModel, BertForMaskedLM, BertConfig, EncoderDecoderModel, BertLMHeadModel, AutoModelForSequenceClassification
from sklearn.metrics import roc_auc_score
from tokenizers.processors import TemplateProcessing

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
        "--nb_classes_v", 
        default=70, 
        type=int, 
        help="number of classes",
    )
    parser.add_argument(
        "--nb_classes_d", 
        default=70, 
        type=int, 
        help="number of classes",
    )
    parser.add_argument(
        "--nb_classes_j", 
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
        "--wandb",
        action="store_true",
        help="follow run on wandb or not."
    )
    args = parser.parse_args()

    if args.wandb :
        # wandb
        import wandb
        wandb.init(project="cdr_detection")
    
    if args.allele :
        allele = True
    else :
        allele = False


    device = torch.device('cuda')
    model = DNABertForSequenceAndTokenClassification.from_pretrained(f"zhihan1996/DNA_bert_{args.kmer}", num_labels_v=args.nb_classes_v, num_labels_d=args.nb_classes_d, num_labels_j=args.nb_classes_j)
    tokenizer = AutoTokenizer.from_pretrained(f"zhihan1996/DNA_bert_{args.kmer}", padding=True, trust_remote_code=True)
    tokenizer._tokenizer.post_processor = TemplateProcessing(
       single="<CLS> <CLS> <CLS> $A <EOS>",
       special_tokens=[
          ("<EOS>", 2),
          ("<CLS>", 1)
       ],
    )
    model = model.to(device)


    # train/validation split
    dataset_train = VDJDataset(args.train_dir, tokenizer, device, target_V=None, target_D=None, target_J=None, kmer=args.kmer, allele=allele, nb_seq=args.nb_seq_max, max_len=args.max_len) 
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
    # CLS0
    y_loss0 = {} 
    y_loss0['train'] = []
    y_loss0['val'] = []
    y_err0 = {}
    y_err0['train'] = []
    y_err0['val'] = []
    # CLS1
    y_loss1 = {} 
    y_loss1['train'] = []
    y_loss1['val'] = []
    y_err1 = {}
    y_err1['train'] = []
    y_err1['val'] = []
    # CLS2
    y_loss2 = {} 
    y_loss2['train'] = []
    y_loss2['val'] = []
    y_err2 = {}
    y_err2['train'] = []
    y_err2['val'] = []
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
            outputs = model(**batch[0], labels_V=batch[1], labels_D=batch[2], labels_J=batch[3], Tokenlabels=batch[8])
            # separate outputs
            cls0_output = outputs[0]
            cls1_output = outputs[1]
            cls2_output = outputs[2]
            token_output = outputs[3]
            # 4 loss
            loss_cls0 = cls0_output.loss
            loss_cls1 = cls1_output.loss
            loss_cls2 = cls2_output.loss
            token_loss = token_output.loss
            loss_cls0.backward(retain_graph=True)
            loss_cls1.backward(retain_graph=True)
            loss_cls2.backward(retain_graph=True)
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
                metric_cls0 = evaluate.load("accuracy")
                metric_cls1 = evaluate.load("accuracy")
                metric_cls2 = evaluate.load("accuracy")
                token_metric = []
                for batch in dataset :
                    with torch.no_grad() :
                        outputs = model(**batch[0], labels_V=batch[1], labels_D=batch[2], labels_J=batch[3], Tokenlabels=batch[8])
                    # seperate outputs
                    cls0_output = outputs[0]
                    cls1_output = outputs[1]
                    cls2_output = outputs[2]
                    token_output = outputs[3]
                    # logits
                    logits_cls0 = cls0_output.logits
                    logits_cls1 = cls1_output.logits
                    logits_cls2 = cls2_output.logits
                    token_logits = token_output.logits
                    # accuracy for cls
                    # CLS1
                    predictions_cls0 = torch.argmax(logits_cls0, dim=-1)
                    metric_cls0.add_batch(predictions=predictions_cls0, references=batch[1])
                    # CLS2
                    predictions_cls1 = torch.argmax(logits_cls1, dim=-1)
                    metric_cls1.add_batch(predictions=predictions_cls1, references=batch[2])
                    # CLS3
                    predictions_cls2 = torch.argmax(logits_cls2, dim=-1)
                    metric_cls2.add_batch(predictions=predictions_cls2, references=batch[3])
                    # accuracy for token
                    token_predictions = torch.argmax(token_logits, dim=-1)
                    for pred, ref in zip(token_predictions, batch[8]) :
                        token_metric.append(torch.eq(pred,ref).sum()/pred.shape[0])
                token_metric = [value.cpu().item() for value in token_metric]
                token_metric = np.sum(token_metric)/len(token_metric)
                epoch_acc_cls0 = metric_cls0.compute()['accuracy']
                epoch_acc_cls1 = metric_cls1.compute()['accuracy']
                epoch_acc_cls2 = metric_cls2.compute()['accuracy']
                epoch_token_acc = token_metric
                if dataset == train_dataloader :
                    y_err0['train'].append(epoch_acc_cls0)
                    y_err1['train'].append(epoch_acc_cls1)
                    y_err2['train'].append(epoch_acc_cls2)
                    y_err_token['train'].append(epoch_token_acc)
                else :
                    y_err0['val'].append(epoch_acc_cls0)
                    y_err1['val'].append(epoch_acc_cls1)
                    y_err2['val'].append(epoch_acc_cls2)
                    y_err_token['val'].append(epoch_token_acc)

            # loss
            for dataset in [train_dataloader, test_dataloader] :
                running_loss_cls0, running_loss_cls1, running_loss_cls2, running_loss_token = 0.0, 0.0, 0.0, 0.0
                for batch in dataset : 
                    with torch.no_grad() :
                        outputs = model(**batch[0], labels_V=batch[1], labels_D=batch[2], labels_J=batch[3], Tokenlabels=batch[8])
                    cls0_output = outputs[0]
                    cls1_output = outputs[1]
                    cls2_output = outputs[2]
                    token_output = outputs[3]
                    loss0 = cls0_output.loss
                    loss1 = cls1_output.loss
                    loss2 = cls2_output.loss
                    token_loss = token_output.loss
                    running_loss_cls0 += loss0.item()
                    running_loss_cls1 += loss1.item()
                    running_loss_cls2 += loss2.item()
                    running_loss_token += token_loss.item()
                epoch_loss0 = running_loss_cls0/len(dataset)
                epoch_loss1 = running_loss_cls1/len(dataset)
                epoch_loss2 = running_loss_cls2/len(dataset)
                epoch_loss_token = running_loss_token/len(dataset)
                if dataset == train_dataloader :
                    y_loss0['train'].append(epoch_loss0)
                    y_loss1['train'].append(epoch_loss1)
                    y_loss2['train'].append(epoch_loss2)
                    y_loss_token['train'].append(epoch_loss_token)
                else :
                    y_loss0['val'].append(epoch_loss0)
                    y_loss1['val'].append(epoch_loss1)
                    y_loss2['val'].append(epoch_loss2)
                    y_loss_token['val'].append(epoch_loss_token)

            """# wandb
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
                wandb.log(wandb_dict)"""

        print(
            f"Epoch {epoch}:\n",
            f"CLS0 : Train : Loss:{y_loss0['train'][-1]}, Accuracy : {y_err0['train'][-1]} \n \t Validation : Loss : {y_loss0['val'][-1]}, Accuracy : {y_err0['val'][-1]}\n",
            f"CLS1 : Train : Loss:{y_loss1['train'][-1]}, Accuracy : {y_err1['train'][-1]} \n \t Validation : Loss : {y_loss1['val'][-1]}, Accuracy : {y_err1['val'][-1]}\n",
            f"CLS2 : Train : Loss:{y_loss2['train'][-1]}, Accuracy : {y_err2['train'][-1]} \n \t Validation : Loss : {y_loss2['val'][-1]}, Accuracy : {y_err2['val'][-1]}\n",
            f"Token : Train : Loss:{y_loss_token['train'][-1]}, Accuracy : {y_err_token['train'][-1]} \n \t Validation : Loss : {y_loss_token['val'][-1]}, Accuracy : {y_err_token['val'][-1]}"
        )

        


        # save the best model
        acc_ref = 0
        if args.save_model :
            # best model criteria is based on test accuracy
            if y_err0['val'][-1] > acc_ref :
                acc_ref = y_err0['val'][-1]
                torch.save(model, args.save_name)

    draw_curve(x_epoch, y_loss0, y_err0, 'CLS0.png')
    draw_curve(x_epoch, y_loss1, y_err1, 'CLS1.png')
    draw_curve(x_epoch, y_loss2, y_err2, 'CLS2.png')
    draw_curve(x_epoch, y_loss_token, y_err_token, 'Token.png')


"""    for batch in test_dataloader :
        with torch.no_grad() :
            outputs = model(**batch[0])

        logits1 = outputs[0].logits
        logits2 = outputs[1].logits

        print(logits1[0], logits2[0])
        print(len(logits2[0]), len(logits2[0]))"""

if __name__ == '__main__' :
    main()