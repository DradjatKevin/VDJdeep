import torch 
import torch.nn as nn
import torchmetrics
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
from scipy.special import softmax
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
from Bio import pairwise2
from Bio.Seq import Seq
import warnings
from tqdm.auto import tqdm
import evaluate
import time


def main() :
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default=None,
        type=str,
        required=True,
        help="The model data file path. Should be a .pt file (or pth).",
    )
    parser.add_argument(
        "--test_dir",
        default=None,
        type=str,
        required=True,
        help="The test data dir. Should contain the .fasta files (or other data files) for the task.",
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
        "--allele", 
        action="store_true", 
        help="Whether consider allele or not."
    )
    parser.add_argument(
        "--nb_seq_max", 
        default=10000,
        type=int, 
        help="number of maximum sequences"
    )
    parser.add_argument(
        "--output",
        default='output.txt',
        type=str,
        help='name of the output file'
    )

    args = parser.parse_args()


    # Preprocessing
    start = time.time()

    device = torch.device('cuda')
    tokenizer = AutoTokenizer.from_pretrained(f"zhihan1996/DNA_bert_{args.kmer}", padding=True, trust_remote_code=True)
    tokenizer._tokenizer.post_processor = TemplateProcessing(
       single="<CLS> <CLS> <CLS> $A <EOS>",
       special_tokens=[
          ("<EOS>", 2),
          ("<CLS>", 1)
       ],
    )
    model = torch.load(args.model)
    model = model.to(device)

    dataset_test = VDJDataset(args.test_dir, tokenizer, device, target_V=None, target_D=None, target_J=None, kmer=args.kmer, allele=args.allele, nb_seq=args.nb_seq_max)
    test_dataloader = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False, collate_fn=dataset_test.genes_collate_function)

    end = time.time()

    print(f'Preprocessing Time: {end-start}')

    # Model
    start = time.time()

    metric_cls0 = evaluate.load("accuracy")
    metric_cls1 = evaluate.load("accuracy")
    metric_cls2 = evaluate.load("accuracy")
    token_metric = []
    model.eval()

    outfile = open(args.output, 'w')
    cdr_file = open(f'cdr3_{args.output}', 'w')

    # import dictionnaries
    if args.allele :
        v_genes_dict = np.load('data/dict/v_alleles_276.npy', allow_pickle='True').item()
        d_genes_dict = np.load('data/dict/d_alleles_37.npy', allow_pickle='True').item()
        j_genes_dict = np.load('data/dict/j_alleles_11.npy', allow_pickle='True').item()
    else :
        v_genes_dict = np.load('data/dict/v_genes_75.npy', allow_pickle='True').item()
        d_genes_dict = np.load('data/dict/d_genes_30.npy', allow_pickle='True').item()
        j_genes_dict = np.load('data/dict/j_genes_6.npy', allow_pickle='True').item()

    for batch in test_dataloader :
        c = 0
        with torch.no_grad() :
            outputs = model(**batch[0], labels_V=batch[1], labels_D=batch[2], labels_J=batch[3], Tokenlabels=batch[8])
        cls0_output = outputs[0]
        cls1_output = outputs[1]
        cls2_output = outputs[2]
        token_output = outputs[3]
        logits_cls0 = cls0_output.logits
        logits_cls1 = cls1_output.logits
        logits_cls2 = cls2_output.logits
        token_logits = token_output.logits
        # accuracy for cls1
        predictions_cls0 = torch.argmax(logits_cls0, dim=-1)
        metric_cls0.add_batch(predictions=predictions_cls0, references=batch[1])
        # accuracy for cls2
        predictions_cls1 = torch.argmax(logits_cls1, dim=-1)
        metric_cls1.add_batch(predictions=predictions_cls1, references=batch[2])
        # accuracy for cls3
        predictions_cls2 = torch.argmax(logits_cls2, dim=-1)
        metric_cls2.add_batch(predictions=predictions_cls2, references=batch[3])
        # accuracy for token
        token_predictions = torch.argmax(token_logits, dim=-1)
        for pred, ref in zip(token_predictions, batch[8]) :
            token_metric.append(torch.eq(pred,ref).sum()/pred.shape[0])
            cdr_file.write(f'{pred}\n{ref}\n \n')

        # write files
        # output file
        for vect_v, vect_d, vect_j, token_pred in zip(logits_cls0, logits_cls1, logits_cls2, token_predictions) :
            token_pred = token_pred.detach().cpu().numpy()
            pos_list = []
            for i in range(len(token_pred)-1) :
                if token_pred[i] != token_pred[i+1] and pos_list == []: pos_list.append(i+1)
                elif token_pred[i] != token_pred[i+1] and pos_list != []: pos_list.append(i)
            vect_v, vect_d, vect_j = vect_v.detach().cpu().numpy(), vect_d.detach().cpu().numpy(), vect_j.detach().cpu().numpy()
            vect_v, vect_d, vect_j = softmax(vect_v), softmax(vect_d), softmax(vect_j)
            pred_v = [u for u, v in v_genes_dict.items() if v == int(np.argmax(vect_v))][0]
            pred_d = [u for u, v in d_genes_dict.items() if v == int(np.argmax(vect_d))][0]
            pred_j = [u for u, v in j_genes_dict.items() if v == int(np.argmax(vect_j))][0]
            outfile.write(f'>{batch[9][c]}|{pred_v}|{pred_d}|{pred_j}|{pos_list[0]-2}|{pos_list[1]-2}\n{kmer2seq(batch[7][c])}\n')

    token_metric = [value.cpu().item() for value in token_metric]
    token_metric = np.sum(token_metric)/len(token_metric)

    
    end = time.time()

    print(f'CLS0 acc : {metric_cls0.compute()}')
    print(f'CLS1 acc : {metric_cls1.compute()}')
    print(f'CLS2 acc : {metric_cls2.compute()}')
    print(f'token acc : {token_metric}')

    print(f'Evaluation Time : {end-start}')


if __name__ == "__main__" :
    main()
