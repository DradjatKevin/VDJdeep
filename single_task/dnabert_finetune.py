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
from src.data_vdj_old import *
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
from src.graphic import *






def main():
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
    parser.add_argument(
        "--nb_frozen_layers", 
        default=1, 
        type=int, 
        help="number of frozen layers", 
    )
    parser.add_argument("--freeze_embedding", action="store_true", help="Whether to run training.")
    parser.add_argument(
        "--nb_classes", 
        default=76, 
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
        "--figure_name", 
        type=str, 
        default='metrics.png', 
        help="name of the metrics figure"
    )
    parser.add_argument(
        "--all_test", 
        action="store_true", 
        help="Whether to run tests on 5 datasets"
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
        wandb.init(project="vdj_detection")

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
        rev = True

    device = torch.device('cuda')
    model = AutoModelForSequenceClassification.from_pretrained("zhihan1996/DNA_bert_"+str(args.kmer),num_labels=num_labels, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNA_bert_"+str(args.kmer), trust_remote_code=True)
    model = model.to(device)

    # train/validation split
    dataset_train = VDJDataset(args.train_dir, tokenizer, device, target_V=None, target_D=None, target_J=None, kmer=args.kmer, allele=allele, nb_seq=args.nb_seq_max, max_len=args.max_len, rev=rev) 
    train_size = int(0.8 * len(dataset_train))
    test_size = len(dataset_train) - train_size
    # random indices
    #train_index = np.random.choice(len(dataset_train), train_size, replace=False)
    #train_dataset, test_dataset = dataset_train[train_index], dataset_train[~train_index]
    train_dataset, test_dataset = torch.utils.data.random_split(dataset_train, [train_size, test_size])
    def genes_collate_function_ext(batch):

        (seqs, vs, ds, js, va, da, ja) = zip(*batch)

        raw_seqs = deepcopy(seqs)

        seqs = tokenizer(list(seqs),padding="longest", truncation=True, add_special_tokens=True)
        seqs = {k: torch.tensor(v).to(device) for k, v in seqs.items()}#default_collate(peptide)
        
        vs =  default_collate(vs).to(device)
        ds =  default_collate(ds).to(device)
        js =  default_collate(js).to(device)

        va =  default_collate(va).to(device)
        da =  default_collate(da).to(device)
        ja =  default_collate(ja).to(device)

        return seqs, vs, ds, js, va, da, ja, raw_seqs


    # train
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=genes_collate_function_ext)
    # validation
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=genes_collate_function_ext)


    # Tests datasets
    if args.all_test :
        dataset_test0 = VDJDataset('data/nika_data/simple_plus_indels.fasta', tokenizer, device, target_V=None, target_D=None, target_J=None, kmer=args.kmer, allele=allele, nb_seq=args.nb_seq_max, max_len=args.max_len, rev=rev)
        dataset_test5 = VDJDataset('data/nika_data/simple_plus_indels_5Mut_out.fasta', tokenizer, device, target_V=None, target_D=None, target_J=None, kmer=args.kmer, allele=allele, nb_seq=args.nb_seq_max, max_len=args.max_len, rev=rev) 
        dataset_test10 = VDJDataset('data/nika_data/simple_plus_indels_10Mut_out.fasta', tokenizer, device, target_V=None, target_D=None, target_J=None, kmer=args.kmer, allele=allele, nb_seq=args.nb_seq_max, max_len=args.max_len, rev=rev) 
        dataset_test20 = VDJDataset('data/nika_data/simple_plus_indels_20Mut_out.fasta', tokenizer, device, target_V=None, target_D=None, target_J=None, kmer=args.kmer, allele=allele, nb_seq=args.nb_seq_max, max_len=args.max_len, rev=rev) 
        dataset_test40 = VDJDataset('data/nika_data/simple_plus_indels_40Mut_out.fasta', tokenizer, device, target_V=None, target_D=None, target_J=None, kmer=args.kmer, allele=allele, nb_seq=args.nb_seq_max, max_len=args.max_len, rev=rev) 
        dataset_test80 = VDJDataset('data/nika_data/simple_plus_indels_80Mut_out.fasta', tokenizer, device, target_V=None, target_D=None, target_J=None, kmer=args.kmer, allele=allele, nb_seq=args.nb_seq_max, max_len=args.max_len, rev=rev)
        test_dataloader0 = torch.utils.data.DataLoader(dataset=dataset_test0, batch_size=args.batch_size, shuffle=True, collate_fn=dataset_test0.genes_collate_function)
        test_dataloader5 = torch.utils.data.DataLoader(dataset=dataset_test5, batch_size=args.batch_size, shuffle=True, collate_fn=dataset_test5.genes_collate_function)
        test_dataloader10 = torch.utils.data.DataLoader(dataset=dataset_test10, batch_size=args.batch_size, shuffle=True, collate_fn=dataset_test10.genes_collate_function)
        test_dataloader20 = torch.utils.data.DataLoader(dataset=dataset_test20, batch_size=args.batch_size, shuffle=True, collate_fn=dataset_test20.genes_collate_function)
        test_dataloader40 = torch.utils.data.DataLoader(dataset=dataset_test40, batch_size=args.batch_size, shuffle=True, collate_fn=dataset_test40.genes_collate_function)
        test_dataloader80 = torch.utils.data.DataLoader(dataset=dataset_test80, batch_size=args.batch_size, shuffle=True, collate_fn=dataset_test80.genes_collate_function)
        test_dataloader_list = [test_dataloader0, test_dataloader5, test_dataloader10, test_dataloader20, test_dataloader40, test_dataloader80]
        dataset_names = ['0mut', '5mut', '10mut', '20mut', '40mut', '80mut']

    # freeze layers
    if args.freeze_embedding :
        for param in model.bert.embeddings.parameters(): #params have requires_grad=True by default
            param.requires_grad = False
        if args.nb_frozen_layers > 1 :
            for i, layer in enumerate(model.bert.encoder.layer) :
                if i < args.nb_frozen_layers :
                    for param in layer.parameters() :
                        param.requires_grad = False

    # loss and err history
    y_loss = {} 
    y_loss['train'] = []
    y_loss['val'] = []
    y_err = {}
    y_err['train'] = []
    y_err['val'] = []
    x_epoch = []

    # wandb track
    wandb_dict = {}



    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 20, 300)
    progress_bar = tqdm(range(args.epoch))
    
    for epoch in range(args.epoch):
        #print('Epoch :', epoch)
        x_epoch.append(epoch)
        # training 
        model.train()

        for batch in train_dataloader:
            outputs =  model(**batch[0], labels=batch[index_class])
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        progress_bar.update(1)
        scheduler.step()


        # compute metrics
        model.eval()

        # accuracy
        metric = evaluate.load("accuracy")
        for dataset in [train_dataloader, test_dataloader] :
            for batch in dataset :
                with torch.no_grad():
                    outputs = model(**batch[0], labels=batch[index_class])
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                metric.add_batch(predictions=predictions, references=batch[index_class])
            epoch_acc = metric.compute()['accuracy']
            if dataset == train_dataloader :
                y_err['train'].append(epoch_acc)
            else :
                y_err['val'].append(epoch_acc)
        
        # loss
        for dataset in [train_dataloader, test_dataloader] :
            running_loss = 0.0
            for batch in dataset :
                with torch.no_grad():
                    outputs = model(**batch[0], labels=batch[index_class])
                loss = outputs.loss
                running_loss += loss.item() 
            epoch_loss = running_loss / len(dataset)
            if dataset == train_dataloader :
                y_loss['train'].append(epoch_loss)
            else :
                y_loss['val'].append(epoch_loss)

        # tests on datasets
        metric = evaluate.load("accuracy")       
        if args.all_test :
            if epoch%10 == 0 :
                for dataset, name in zip(test_dataloader_list, dataset_names) :
                    for batch in dataset :
                        with torch.no_grad() :
                            outputs = model(**batch[0], labels=batch[index_class])
                        # accuracy
                        logits = outputs.logits
                        predictions = torch.argmax(logits, dim=-1)
                        metric.add_batch(predictions=predictions, references=batch[index_class])
                    epoch_acc = metric.compute()['accuracy']
                    # wandb 
                    wandb_dict[f'{name}_acc'] = epoch_acc




        print(f"Epoch {epoch} \n Train : Loss : {y_loss['train'][-1]}, Accuracy : {y_err['train'][-1]} \n Validation : Loss : {y_loss['val'][-1]}, Accuracy : {y_err['val'][-1]}")


        # wandb
        wandb_dict['epoch'] = epoch
        wandb_dict['train_acc'] = y_err['train'][-1]
        wandb_dict['train_loss'] = y_loss['train'][-1]
        wandb_dict['test_acc'] = y_err['val'][-1]
        wandb_dict['test_loss'] = y_loss['val'][-1]
        if args.wandb :
            wandb.log(wandb_dict)


        # save the best model
        acc_ref = 0
        if args.save_model :
            # best model criteria is based on test accuracy
            if y_err['val'][-1] > acc_ref :
                acc_ref = y_err['val'][-1]
                torch.save(model, args.save_name)

    # draw figure at the end
    draw_curve(x_epoch, y_loss, y_err, args.figure_name)

    # final results
    print("final results :\n")
    for name in dataset_names :
        print(name, wandb_dict[f'{name}_acc'])



if __name__ == "__main__":
    main()
















# python dnabert_finetune.py --train_dir "/home/barthelemy/Datasets/vdj/train.fasta" --test_dir "/home/barthelemy/Datasets/vdj/test.fasta" --kmer 3 --batch_size 32 --weight_decay 0.0






