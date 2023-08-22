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
from scipy.special import softmax

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
        "--type",
        type=str,
        default='V',
        help="Which type of gene to consider"
    )

    args = parser.parse_args()

    if args.type == 'V' :
        index_class = 1
        rev=False
    elif args.type == 'D' :
        index_class = 2
        rev=False
    elif args.type == 'J' :
        rev = True
        index_class = 3


    # Preprocessing
    start = time.time()

    device = torch.device('cuda')
    tokenizer = AutoTokenizer.from_pretrained(f"zhihan1996/DNA_bert_{args.kmer}", padding=True, trust_remote_code=True)
    model = torch.load(args.model)
    model = model.to(device)

    dataset_test = VDJDataset(args.test_dir, tokenizer, device, target_V=None, target_D=None, target_J=None, kmer=args.kmer, allele=args.allele, nb_seq=args.nb_seq_max, rev=rev)
    test_dataloader = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False, collate_fn=dataset_test.genes_collate_function)

    end = time.time()

    print(f'Preprocessing Time: {end-start}')

    # Model
    start = time.time()

    metric = evaluate.load('accuracy')
    token_metric = []
    model.eval()

    for batch in test_dataloader :
        with torch.no_grad() :
            outputs = model(**batch[0], labels=batch[index_class], Tokenlabels=batch[8])
        cls_output = outputs[0]
        token_output = outputs[1]
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
    
    end = time.time()

    print(f'acc : {metric.compute()}')
    print(f'token acc : {token_metric}')

    print(f'Evaluation Time : {end-start}')


if __name__ == "__main__" :
    main()
