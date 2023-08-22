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
from src.data_vdj import *
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



def main():
    parser = argparse.ArgumentParser()

    # Required parameters
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
    parser.add_argument(
        "--align",
        action="store_true",
        help="Apply alignment to precise alignment"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="threshold to apply alignment"
    )
    parser.add_argument(
        "--error_file",
        type=str,
        default="error_file.txt",
        help="name of the file to store misclassification"
    )
    parser.add_argument(
        "--precise",
        action="store_true",
        help="Apply alignment after classification or not"
    )
    parser.add_argument(
        "--output",
        type=str,
        default='output.txt',
        help='name of the output file'
    )
    parser.add_argument(
        "--het",
        action="store_true",
        help="For heterozygous/one patient repertoire"
    )
    parser.add_argument(
        "--nolabel",
        action="store_true",
        help="data with no VDJ in headers"
    )
    parser.add_argument(
        "--cluster",
        action='store_true', 
        help='Apply clustering before classification'
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
    elif args.type == 'cdr3_start' :
        rev = False
        index_class = 8
    elif args.type == 'cdr3_end' :
        rev = False
        index_class = 9

    
    # clustering option
    if args.cluster :
        start2 = time.time()
        # execute mmseq2 for clustering
        os.system('mkdir results_clust/')
        os.system(f'mmseqs easy-linclust {args.test_dir} results_clust/results tmp')
        
        # execute function to get clusters informations from files 
        nb_cluster, cluster_names, len_cluster_list, rep_seq_list, seq_cluster_dict = get_clusterInfo('results_clust/results_all_seqs.fasta')

        # load model
        device = torch.device('cuda')
        tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNA_bert_"+str(args.kmer), trust_remote_code=True)
        model = torch.load(args.model)
        model = model.to(device)
        # load data from mmseqs2 output
        dataset_test = VDJDataset_cluster('results_clust/results_rep_seq.fasta', tokenizer, device, target_V=None, target_D=None, target_J=None, kmer=args.kmer, allele=args.allele, nb_seq=args.nb_seq_max, rev=rev)
        test_dataloader = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False, collate_fn=dataset_test.genes_collate_function)
        
        
        # pass to the model
        if args.type != 'cdr3_start'  and args.type != 'cdr3_end' :
            # accuracy and output file
            if args.type == 'V' :
                if args.allele :
                    genes_dict = np.load('data/dict/v_alleles_276.npy', allow_pickle='True').item()
                else :
                    genes_dict = np.load('data/dict/v_genes_75.npy', allow_pickle='True').item()
            elif args.type == 'D' :
                if args.allele :
                    genes_dict = np.load('data/dict/d_alleles_37.npy', allow_pickle='True').item()
                else :
                    genes_dict = np.load('data/dict/d_genes_30.npy', allow_pickle='True').item()
            else :
                if args.allele :
                    genes_dict = np.load('data/dict/j_alleles_11.npy', allow_pickle='True').item()
                else :
                    genes_dict = np.load('data/dict/j_genes_6.npy', allow_pickle='True').item()
            metric = evaluate.load("accuracy")
            model.eval()
            with open(f'{args.output[:len(args.output)-4]}_cluster.txt', 'w') as file :
                k = 0
                for batch in test_dataloader:
                    c = 0
                    with torch.no_grad():
                        outputs = model(**batch[0])

                    logits = outputs.logits
                    predictions = torch.argmax(logits, dim=-1)

                    # output file
                    for vect in logits :
                        vect = vect.detach().cpu().numpy()
                        vect = softmax(vect)
                        pred = [u for u, v in genes_dict.items() if v == int(np.argmax(vect))][0]
                        file.write(f'>{batch[2][c]}|{pred}\n{kmer2seq(batch[1][c])}\n')
                        c += 1
                        k += 1  
        
        # delete results_clust folder
        os.system('rm -r results_clust/')

        # treat vdjdeep output to create cluster_dict {cluster : pred}
        cluster_dict = {}
        cluster, seq = str, str
        with open(f'{args.output[:len(args.output)-4]}_cluster.txt', 'r') as vdjfile :
            for line in vdjfile.readlines() :
                if line[0] == '>' : 
                    cluster, pred = '|'.join(line.split('|')[0:-1]), line.split('|')[-1].split('\n')[0]
                    cluster_dict[cluster] = pred
        
        # take original file to create final output
        c = 0
        with open(args.test_dir, 'r') as file :
            with open(args.output, 'w') as outfile :
                for line in file.readlines() :
                    if line[0] != '>' :
                        seq = line.split('\n')[0].upper()
                        cluster = seq_cluster_dict[seq]
                        pred = cluster_dict[cluster]
                        outfile.write(f'>S{c} {cluster[1:]}|{pred}\n{seq}\n')
                        c+=1


        end2 = time.time()
        print(f'Computation Time: {end2-start2}')


    elif not args.cluster :

        start1 = time.time()

        device = torch.device('cuda')
        tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNA_bert_"+str(args.kmer), trust_remote_code=True)
        #special_tokens_dict = {'additionnal_special_tokens': ['[C1]','[C2]']}
        #num_added_tokens = tokenizer.add_tokens(['[C1]','[C2]'], special_tokens=True)
        #print(tokenizer.convert_tokens_to_ids('[C1]'))
        #print(tokenizer.convert_tokens_to_ids('[C2]'))
        model = torch.load(args.model)
        #model.resize_token_embeddings(len(tokenizer))
        model = model.to(device)
        if args.nolabel :
            dataset_test = VDJDataset_nolabel(args.test_dir, tokenizer, device, target_V=None, target_D=None, target_J=None, kmer=args.kmer, allele=args.allele, nb_seq=args.nb_seq_max, rev=rev)
        elif not args.nolabel :
            dataset_test = VDJDataset(args.test_dir, tokenizer, device, target_V=None, target_D=None, target_J=None, kmer=args.kmer, allele=args.allele, nb_seq=args.nb_seq_max, rev=rev)
        test_dataloader = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False, collate_fn=dataset_test.genes_collate_function)

        end1 = time.time()
        print(f'Preprocessing time : {end1-start1}')


        # one patient option
        if args.het and args.allele :
            # pass all sequences first time to get allele proportions
            if args.type == 'V' : alleles_dict = np.load('data/dict/v_alleles_276.npy', allow_pickle='True').item()
            elif args.type == 'D' : alleles_dict = np.load('data/dict/d_alleles_37.npy', allow_pickle='True').item()
            elif args.type == 'J' : alleles_dict = np.load('data/dict/j_alleles_11.npy', allow_pickle='True').item()

            metric = evaluate.load("accuracy")
            model.eval()
            prop_dict = {}  # dictionnary to store proportions of each allele
            for gene in alleles_dict.keys() :
                prop_dict[gene] = 0

            for batch in test_dataloader :
                with torch.no_grad() :
                    outputs = model(**batch[0], labels=batch[index_class])
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                metric.add_batch(predictions=predictions, references=batch[index_class])
                
                # fill the proportion dict
                for vect in logits :
                    vect = vect.detach().cpu().numpy()
                    vect = softmax(vect)
                    pred = [u for u, v in alleles_dict.items() if v == int(np.argmax(vect))][0]
                    prop_dict[pred] += 1
            print('acc fisrt pass :', metric.compute())

            # get patient alleles from proportion dict
            # import genes dict
            if args.type == 'V' : genes_dict = np.load('data/dict/v_genes_75.npy', allow_pickle='True').item()
            elif args.type == 'D' : genes_dict = np.load('data/dict/d_genes_30.npy', allow_pickle='True').item()
            else : genes_dict = np.load('data/dict/j_genes_6.npy', allow_pickle='True').item()
            alleles_patient = []

            for gene in genes_dict.keys() : 
                allele_prop_list = [[u,v] for u,v in prop_dict.items() if u.split('*')[0] == gene]
                # if the gene has two or less alleles
                if len(allele_prop_list) <= 2 and allele_prop_list != [] :
                    for [u,v] in allele_prop_list :
                        alleles_patient.append(u)
                elif len(allele_prop_list) > 2 and allele_prop_list != [] :
                    prop_list = [v for [u,v] in allele_prop_list]
                    a, b = sorted(range(len(prop_list)), key=lambda i: prop_list[i])[-2:][0], sorted(range(len(prop_list)), key=lambda i: prop_list[i])[-2:][1]
                    alleles_patient.append(allele_prop_list[a][0])
                    alleles_patient.append(allele_prop_list[b][0])

            # second pass according to patient's alleles
            model.eval()
            for batch in test_dataloader :
                with torch.no_grad() :
                    outputs = model(**batch[0], labels=batch[index_class])
                logits = outputs.logits
                # replace prob of alleles that are not in patient's allele by zero
                for i, vect in enumerate(logits) :
                    for j, val in enumerate(vect) :
                        if j in alleles_dict.values() :
                            allele = [u for u, v in alleles_dict.items() if v == int(j)][0]
                            if allele not in alleles_patient :
                                logits[i][j] == 0
                
                predictions = torch.argmax(logits, dim=-1)
                metric.add_batch(predictions=predictions, references=batch[index_class])
            print('acc second pass :', metric.compute())           
                
        # independant sequences
        elif not args.het :
            start2 = time.time()
            # accuracy and output file
            if args.type == 'V' :
                if args.allele :
                    genes_dict = np.load('data/dict/v_alleles_276.npy', allow_pickle='True').item()
                else :
                    genes_dict = np.load('data/dict/v_genes_75.npy', allow_pickle='True').item()
            elif args.type == 'D' :
                if args.allele :
                    genes_dict = np.load('data/dict/d_alleles_37.npy', allow_pickle='True').item()
                else :
                    genes_dict = np.load('data/dict/d_genes_30.npy', allow_pickle='True').item()
            else :
                if args.allele :
                    genes_dict = np.load('data/dict/j_alleles_11.npy', allow_pickle='True').item()
                else :
                    genes_dict = np.load('data/dict/j_genes_6.npy', allow_pickle='True').item()
            metric = evaluate.load("accuracy")
            model.eval()

            # with labels
            if not args.nolabel :
                if args.type != 'cdr3_start' and args.type != 'cdr3_end' : # V, D, J
                    with open(args.output, 'w') as file :
                        k = 0
                        for batch in test_dataloader:
                            c = 0
                            with torch.no_grad():
                                outputs = model(**batch[0], labels=batch[index_class])

                            logits = outputs.logits
                            predictions = torch.argmax(logits, dim=-1)
                            metric.add_batch(predictions=predictions, references=batch[index_class])

                            # output file
                            for vect in logits :
                                vect = vect.detach().cpu().numpy()
                                vect = softmax(vect)
                                pred = [u for u, v in genes_dict.items() if v == int(np.argmax(vect))][0]
                                file.write(f'>{batch[10][c]}|{pred}\n{kmer2seq(batch[7][c])}\n')
                                c += 1
                                k += 1
                    print("acc :", metric.compute())
                    end2 = time.time()
                    print(f'Evaluation time : {end2-start2}')
                    print(f'Total time : {end2-start1}')

                    # top k acc
                    top3_acc = []
                    model.eval()
                    for batch in test_dataloader :
                        with torch.no_grad() :
                            outputs = model(**batch[0], labels=batch[index_class])
                        logits = outputs.logits
                        logits = logits.to(device)
                        metric = torchmetrics.Accuracy(task='multiclass', num_classes=logits.shape[1], top_k=2)
                        metric = metric.to(device)
                        top3_acc.append(metric(logits, batch[index_class]).item())
                    top3_acc = np.mean(top3_acc)
                    print("top2_acc :", top3_acc)


                    # error file 
                    with open(args.error_file, 'w') as file :
                        file.write(f'pred_class\ttrue_class\tdiff_top2\n')
                        # import genes_dict
                        if args.type == 'V' :
                            if args.allele :
                                genes_dict = np.load('data/dict/v_alleles_276.npy', allow_pickle='True').item()
                            else :
                                genes_dict = np.load('data/dict/v_genes_75.npy', allow_pickle='True').item()
                        elif args.type == 'D' :
                            if args.allele :
                                genes_dict = np.load('data/dict/d_alleles_37.npy', allow_pickle='True').item()
                            else :
                                genes_dict = np.load('data/dict/d_genes_30.npy', allow_pickle='True').item()
                        else :
                            if args.allele :
                                genes_dict = np.load('data/dict/j_alleles_11.npy', allow_pickle='True').item()
                            else :
                                genes_dict = np.load('data/dict/j_genes_6.npy', allow_pickle='True').item()
                        for batch in test_dataloader :
                            c = 0
                            with torch.no_grad() :
                                outputs = model(**batch[0], labels=batch[index_class])
                            logits = outputs.logits
                            for vect in logits :
                                vect = vect.detach().cpu().numpy()
                                vect = softmax(vect)
                                # top-2 
                                ind = np.argpartition(vect, -2)[-2:]
                                if np.argmax(vect) != batch[index_class][c].item() :
                                    pred = [k for k, v in genes_dict.items() if v == int(np.argmax(vect))][0]
                                    true = [k for k, v in genes_dict.items() if v == int(batch[index_class][c].item())][0]
                                    diff = np.abs(vect[ind[0]] - vect[ind[1]]) 
                                    file.write(f'{pred}\t{true}\t{diff}\n')
                                c += 1
                else : # cdr3 
                    with open(args.output, 'w') as file :
                        k = 0
                        for batch in test_dataloader:
                            c = 0
                            with torch.no_grad():
                                outputs = model(**batch[0], labels=batch[index_class])

                            logits = outputs.logits
                            predictions = torch.argmax(logits, dim=-1)
                            metric.add_batch(predictions=predictions, references=batch[index_class])

                            # output file
                            for vect in logits :
                                vect = vect.detach().cpu().numpy()
                                vect = softmax(vect)
                                pred = int(np.argmax(vect))
                                file.write(f'>s{k}|{pred}\n{kmer2seq(batch[7][c])}\n')
                                c += 1
                                k += 1
                    print("acc :", metric.compute())
                    end2 = time.time()
                    print(f'Evaluation time : {end2-start2}')
                    print(f'Total time : {end2-start1}')

                    # top k acc
                    top3_acc = []
                    model.eval()
                    for batch in test_dataloader :
                        with torch.no_grad() :
                            outputs = model(**batch[0], labels=batch[index_class])
                        logits = outputs.logits
                        logits = logits.to(device)
                        metric = torchmetrics.Accuracy(task='multiclass', num_classes=logits.shape[1], top_k=2)
                        metric = metric.to(device)
                        top3_acc.append(metric(logits, batch[index_class]).item())
                    top3_acc = np.mean(top3_acc)
                    print("top2_acc :", top3_acc)
            # no labels
            else :
                if args.type != 'cdr3_start'  and args.type != 'cdr3_end' :
                    with open(args.output, 'w') as file :
                        k = 0
                        for batch in test_dataloader:
                            c = 0
                            with torch.no_grad():
                                outputs = model(**batch[0])

                            logits = outputs.logits
                            predictions = torch.argmax(logits, dim=-1)

                            # output file
                            for vect in logits :
                                vect = vect.detach().cpu().numpy()
                                vect = softmax(vect)
                                print(vect)
                                pred = [u for u, v in genes_dict.items() if v == int(np.argmax(vect))][0]
                                #file.write(f'>s{k}|{pred}\n{kmer2seq(batch[1][c])}\n')
                                file.write(f'>s{k}|{pred}\n{kmer2seq(batch[1][c])}\n')
                                c += 1
                                k += 1
                    end2 = time.time()  
                    print(f'Evaluation time : {end2-start2}')
                    print(f'Total time : {end2-start1}')   
                # cdr3 no labels
                else :  
                    with open(args.output, 'w') as file :
                        k = 0
                        for batch in test_dataloader:
                            c = 0
                            with torch.no_grad():
                                outputs = model(**batch[0])

                            logits = outputs.logits
                            predictions = torch.argmax(logits, dim=-1)

                            # output file
                            for vect in logits :
                                vect = vect.detach().cpu().numpy()
                                vect = softmax(vect)
                                pred = int(np.argmax(vect))
                                file.write(f'>s{k}|{pred}\n{kmer2seq(batch[1][c])}\n')
                                c += 1
                                k += 1
                    end2 = time.time()
                    print(f'Evaluation time : {end2-start2}')
                    print(f'Total time : {end2-start1}')

                    # top k acc
                    top3_acc = []
                    model.eval()
                    for batch in test_dataloader :
                        with torch.no_grad() :
                            outputs = model(**batch[0], labels=batch[index_class])
                        logits = outputs.logits
                        logits = logits.to(device)
                        metric = torchmetrics.Accuracy(task='multiclass', num_classes=logits.shape[1], top_k=2)
                        metric = metric.to(device)
                        top3_acc.append(metric(logits, batch[index_class]).item())
                    top3_acc = np.mean(top3_acc)
                    print("top2_acc :", top3_acc)


        # apply alignment after gene assignment (not allele)
        if args.precise and not args.allele :
            metric = evaluate.load("accuracy")

            # import dict
            v_genes_dict = np.load('data/dict/v_genes_75.npy', allow_pickle='True').item()   # {gene_name : label}
            v_allele_dict = np.load('data/dict/v_alleles_276.npy', allow_pickle='True').item()  # {allele_name : label}
            v_genes_to_alleles = np.load('data/v_genes_to_alleles.npy', allow_pickle='True').item()  # {gene_name : [allele_name]}
            allele_dict = np.load('data/v_alleles_seq.npy', allow_pickle='True').item()  # {allele_name : seq}
            seq_dict = np.load('data/v_genes_seq.npy', allow_pickle='True').item()  # {gene_name : first allele seq}
            #seq_dict = np.load('data/v_genes_all_alleles.npy', allow_pickle='True').item()  # {gene_name : [all alleles seq]}

            count_filter = []
            for batch in test_dataloader :
                c = 0
                new_pred =[]
                with torch.no_grad() :
                    outputs = model(**batch[0], labels=batch[1])
                logits = outputs.logits

                for vect in logits :
                    seq = kmer2seq(batch[7][c])
                    vect = vect.detach().cpu().numpy()
                    vect = softmax(vect)
                    ind = np.argpartition(vect, -2)[-2:]
                    if np.abs(vect[ind[0]] - vect[ind[1]]) < args.threshold : 
                        count_filter.append(1)
                        fam, gene_order = [], []
                        # get gene names
                        for i in ind :
                            fam.append([n for n, m in v_genes_dict.items() if m == i][0])
                            gene_order.append([k for k, v in v_genes_dict.items() if v == i][0])
                        # get alleles name
                        for i in range(len(fam)) :
                            fam[i] = v_genes_to_alleles[fam[i]].copy()
                        # convert to seq (only fam) 
                        for i in range(len(fam)) :
                            for j in range(len(fam[i])) :
                                fam[i][j] = allele_dict[fam[i][j]]
                        for i in range(len(fam)) :
                            for j in range(len(fam[i])) :
                                fam[i][j] = Seq(fam[i][j].upper())
                        seq = Seq(seq)

                        # align
                        scores = []
                        for i in range(len(fam)) :
                            score_sublist = []
                            for j in range(len(fam[i])) :
                                alignment = pairwise2.align.localms(seq[:305], fam[i][j], 1, -1, -3, -1)
                                score_sublist.append(alignment[0][2])
                            #scores.append(alignment[0][2])
                            scores.append(score_sublist)
                        # max of each sublist
                        for i in range(len(scores)) :
                            scores[i] = np.max(scores[i])
                        # case of equality
                        if scores[0] == scores[1] :
                            name = np.argmax(vect)
                        else :    
                            name = gene_order[int(np.argmax(scores))]
                            # get label
                            name = v_genes_dict[name]
                        new_pred.append(name)

                        pred = [k for k, v in genes_dict.items() if v == name][0]
                        true = [k for k, v in genes_dict.items() if v == int(batch[1][c].item())][0]
                        if name == int(batch[1][c].item()) :
                            print('good :', scores, pred, true)
                        else :
                            print('wrong :', scores, pred, true)

                    else :
                        count_filter.append(0)
                        new_pred.append(np.argmax(vect))

                    c += 1
                new_pred = torch.Tensor(new_pred)
                metric.add_batch(predictions=new_pred, references=batch[1])
            print("acc after alignment :", metric.compute())
            print("Pourcentage of aligned sequences :", np.sum(count_filter)/len(count_filter))

        # apply alignment after allele assignment (--allele option)
        if args.precise and args.allele :
            metric = evaluate.load("accuracy")
            # import dict
            v_allele_dict = np.load('data/dict/v_alleles_276.npy', allow_pickle='True').item()  # {allele_name : label}
            allele_dict = np.load('data/seq_dict/v_alleles_seq.npy', allow_pickle='True').item()  # {allele_name : seq}

            count_filter = []

            for batch in test_dataloader :
                c = 0
                new_pred = []
                with torch.no_grad() :
                    outputs = model(**batch[0], labels=batch[1])
                logits = outputs.logits

                for vect in logits :
                    seq = kmer2seq(batch[7][c])
                    vect = vect.detach().cpu().numpy()
                    vect = softmax(vect)
                    ind = np.argpartition(vect, -2)[-2:]
                    if np.abs(vect[ind[0]] - vect[ind[1]]) < args.threshold :
                        count_filter.append(1)
                        fam, allele_order = [], []
                        # get allele name
                        for i in ind :
                            fam.append([n for n, m in v_allele_dict.items() if m == i][0])
                            allele_order.append([n for n, m in v_allele_dict.items() if m == i][0])
                        # convert to seq (only fam)
                        for i in range(len(fam)) :
                            fam[i] = allele_dict[fam[i]]
                        for i in range(len(fam)) :
                            fam[i] = Seq(fam[i].upper())
                        seq = Seq(seq)

                        # align 
                        scores = []
                        for i in range(len(fam)) :
                            alignment = pairwise2.align.localms(seq[:305], fam[i], 1, -1, -3, -1)
                            scores.append(alignment[0][2])
                        if scores[0] == scores[1] :
                            name = np.argmax(vect)
                        else :
                            name = allele_order[int(np.argmax(scores))]
                            # get label
                            name = v_allele_dict[name]
                        new_pred.append(name)

                    else :
                        count_filter.append(0)
                        new_pred.append(np.argmax(vect))
                    
                    c += 1
                new_pred = torch.Tensor(new_pred)
                metric.add_batch(predictions=new_pred, references=batch[1])
            print("acc after alingment :", metric.compute())
            print("Pourcentage of aligned sequences :", np.sum(count_filter)/len(count_filter))
                    
            




        # alignment to find allele
        if args.align :
            metric = evaluate.load("accuracy")
            # import genes_dict
            v_genes_dict = np.load('data/dict/v_genes_75.npy', allow_pickle='True').item()   # {gene_name : label}
            v_allele_dict = np.load('data/dict/v_alleles_276.npy', allow_pickle='True').item()  # {allele_name : label}

            v_genes_to_alleles = np.load('data/v_genes_to_alleles.npy', allow_pickle='True').item()  # {gene_name : [allele_name]}

            allele_dict = np.load('data/v_alleles_seq.npy', allow_pickle='True').item()  # {allele_name : seq}

            #seq_dict = np.load('data/v_genes_seq.npy', allow_pickle='True').item()
            seq_dict = np.load('data/v_genes_all_alleles.npy', allow_pickle='True').item()  # {gene_name : seq}
            count_filter = []
            for batch in test_dataloader :
                c = 0
                new_pred = []
                with torch.no_grad() :
                    outputs = model(**batch[0], labels=batch[1])
                logits = outputs.logits
                for vect in logits :
                    seq = kmer2seq(batch[7][c])
                    vect = vect.detach().cpu().numpy()
                    vect = softmax(vect)
                    ind = np.argpartition(vect, -2)[-2:]
                    # condition : consider only top-2
                    ind_cond = np.argpartition(vect, -2)[-2:]
                    if np.abs(vect[ind_cond[0]] - vect[ind_cond[1]]) < args.threshold : 
                        count_filter.append(1)
                        # get gene names
                        fam = []
                        gene_order = []
                        for i in ind :
                            fam.append([n for n, m in v_genes_dict.items() if m == i][0])
                            gene_order.append([k for k, v in v_genes_dict.items() if v == i][0])
                        # get alleles from genes {gene_name : [all allele_name]}
                        for i in range(len(fam)) :
                            fam[i] = v_genes_to_alleles[fam[i]].copy()
                            gene_order[i] = v_genes_to_alleles[gene_order[i]]
                        # convert to sequences (only fam) with allele_dict {allele_name : seq}
                        #print(fam)
                        for i in range(len(fam)) :
                            #fam[i] = allele_dict[fam[i]]
                            for j in range(len(fam[i])) :
                                fam[i][j] = allele_dict[fam[i][j]]
                        for i in range(len(fam)) :
                            for j in range(len(fam[i])) :
                                fam[i][j] = Seq(fam[i][j].upper())
                        # convert to Seq
                        seq = Seq(seq)
                        # flat fam and gene_order
                        fam = [seq for sublist in fam for seq in sublist]
                        gene_order = [allele for sublist in gene_order for allele in sublist]
                        # align 
                        scores = []
                        for i in range(len(fam)) :
                            alignment = pairwise2.align.localxx(seq, fam[i])
                            scores.append(alignment[0][2])

                        name = gene_order[int(np.argmax(scores))]
                        # get allele label
                        name = v_allele_dict[name]
                        new_pred.append(name)

                    else :
                        count_filter.append(0)
                        fam, gene_order = 0, 0
                        # gene order
                        gene_order = [k for k, v in v_genes_dict.items() if v == np.argmax(vect)][0]
                        gene_order = v_genes_to_alleles[gene_order]

                        # find gene
                        fam = [n for n, m in v_genes_dict.items() if m == np.argmax(vect)][0]
                        
                        # get alleles from gene
                        fam = v_genes_to_alleles[fam].copy()
                        # convert to Seq (only fam)
                        for i in range(len(fam)) :
                            fam[i] = allele_dict[fam[i]]
                        for i in range(len(fam)) :
                            fam[i] = Seq(fam[i].upper())
                        seq = Seq(seq)

                        # align
                        scores = []
                        for i in range(len(fam)) :
                            alignment = pairwise2.align.localxx(seq, fam[i])
                            scores.append(alignment[0][2])
                        name = gene_order[int(np.argmax(scores))]
                        # get label
                        name = v_allele_dict[name]
                        new_pred.append(name)
                    c += 1
                new_pred = torch.Tensor(new_pred)
                metric.add_batch(predictions=new_pred, references=batch[1+3])
            print("acc after alignment :", metric.compute())
            print("Pourcentage of aligned sequences :", np.sum(count_filter)/len(count_filter))

    
"""    # pure alignment for IGHD case
    if args.type == 'D' and not args.allele :
        metric = evaluate.load("accuracy")
        # import dict

        for batch in test_dataloader :
            c = 0 
            new_pred = []
            with torch.no_grad() :
                outputs = model(**batch[0], labels=batch[2])
            logits = outputs.logits

            # get sequence
            seq = batch[7][c]
            seq = Seq(seq)
"""
            




if __name__ == "__main__":
    main()

    
    
















# python dnabert_finetune.py --train_dir "/home/barthelemy/Datasets/vdj/train.fasta" --test_dir "/home/barthelemy/Datasets/vdj/test.fasta" --kmer 3 --batch_size 32 --weight_decay 0.0






