from torch.utils.data._utils.collate import default_collate
import sys
import torch
import torch.utils.data as data
import numpy as np
import pandas as pd
from copy import deepcopy
import random
import math


def parse_fasta(filename, a3m=False, stop=10000, max_len='max', rev=False):
  '''function to parse fasta file'''
  
  if a3m:
    # for a3m files the lowercase letters are removed
    # as these do not align to the query sequence
    rm_lc = str.maketrans(dict.fromkeys(string.ascii_lowercase))
    
  header, sequence = [],[]
  lines = open(filename, "r")
  for line in lines :
    line = line.rstrip()
    if len(line) > 0:
      if line[0] == ">":
        if len(header) == stop:
          break
        else:
          header.append(line[1:])
          sequence.append([])
      else:
        if a3m: 
          line = line.translate(rm_lc)
        else: 
          line = line.upper()
          if max_len != 'max' :
            line = line[:int(max_len)]
        if rev :
          line = line[::-1]
        sequence[-1].append(line)
  lines.close()
  sequence = [''.join(seq) for seq in sequence]
  
  return header, sequence


def keep_genes_in_dict_v2(header, sequence, allele=False) :
  new_header, new_sequence = [], []
  v_genes_dict = np.load('data/dict/v_alleles_276.npy', allow_pickle='True').item()
  d_genes_dict = np.load('data/dict/d_alleles_37.npy', allow_pickle='True').item()
  j_genes_dict = np.load('data/dict/j_alleles_11.npy', allow_pickle='True').item()

  # keep V, D, J genes that are in the allele dict (we want same alleles on test and train)
  for i in range(len(header)) :
    v_gene_allele = header[i].split('|')[1]
    d_gene_allele = header[i].split('|')[2]
    j_gene_allele = header[i].split('|')[3]

    if v_gene_allele in v_genes_dict.keys() and d_gene_allele in d_genes_dict.keys() and j_gene_allele in j_genes_dict.keys() :
      new_header.append(header[i])
      new_sequence.append(sequence[i])

  return new_header, new_sequence


def parse_name(name, allele=False):
  V, D, J = name.split("|")[1], name.split("|")[2], name.split("|")[3]
  #V = int(V[4])
  #D = int(D[4])
  #J = int(J[4])

  v_allele_dict = np.load('data/dict/v_alleles_276.npy', allow_pickle='True').item()
  d_allele_dict = np.load('data/dict/d_alleles_37.npy', allow_pickle='True').item()
  j_allele_dict = np.load('data/dict/j_alleles_11.npy', allow_pickle='True').item()
  
  if allele :
    V = v_allele_dict[V]
    D = d_allele_dict[D]
    J = j_allele_dict[J]

  else :
    v_genes_dict = np.load('data/dict/v_genes_75.npy', allow_pickle='True').item()
    d_genes_dict = np.load('data/dict/d_genes_30.npy', allow_pickle='True').item()
    j_genes_dict = np.load('data/dict/j_genes_6.npy', allow_pickle='True').item()

    V = v_genes_dict[V.split('*')[0]]
    D = d_genes_dict[D.split('*')[0]]
    J =  j_genes_dict[J.split('*')[0]]

  Va, Da, Ja = v_allele_dict[name.split("|")[1]], d_allele_dict[name.split("|")[2]], j_allele_dict[name.split("|")[3]]
  
  cdr3_start, cdr3_end = name.split("|")[5], name.split("|")[6]


  return V, D, J, Va, Da, Ja, int(cdr3_start), int(cdr3_end)

def seq2kmer(seq, k):
    """
    Convert original sequence to kmers
    
    Arguments:
    seq -- str, original sequence.
    k -- int, kmer of length k specified.
    
    Returns:
    kmers -- str, kmers separated by space
    """
    kmer = [seq[x:x+k] for x in range(len(seq)+1-k)]
    kmers = " ".join(kmer)
    return kmers

def kmer2seq(kmers) :
    list_kmers = kmers.split(' ')
    seq = []
    seq.append(list_kmers[0][0])
    seq.append(list_kmers[0][1])
    for kmer in list_kmers :
      seq.append(kmer[2])
    seq = ''.join(seq)
    return seq


class VDJDataset(data.Dataset):

    def __init__(self, fasta_file, tokenizer, device, target_V=None, target_D=None, target_J=None, kmer=1, allele=False, nb_seq=10000, max_len='max', rev=False):#, alpha_maxlength, beta_maxlength, epitope_maxlength):
        self.device=device
        self.tokenizer = tokenizer
#         self.alpha_maxlength = alpha_maxlength
#         self.beta_maxlength = beta_maxlength
#         self.epitope_maxlength = epitope_maxlength

        print("Loading and Tokenizing the data ...")

        names, seqs = parse_fasta(fasta_file, stop=nb_seq, max_len=max_len, rev=rev)
        names, seqs = keep_genes_in_dict_v2(names, seqs, allele=allele)
        self.V = []
        self.J = []
        self.D = []
        self.Va = []
        self.Da = []
        self.Ja = []
        self.sequences = []
        self.cdr3 = []
        self.heads = []

        for name, seq in zip(names, seqs):
          v,d,j, va, da, ja, cdr3_start, cdr3_end = parse_name(name, allele=allele)
          self.V.append(v)
          self.J.append(j)
          self.D.append(d)
          self.Va.append(va)
          self.Da.append(da)
          self.Ja.append(ja)
          # create cdr binary vector
          cdr_arr = np.array([0 for i in range(len(seq))])
          cdr_arr[cdr3_start-1 : cdr3_end+1] = 1  
          self.cdr3.append(torch.tensor(cdr_arr))

          self.heads.append(name)
          if kmer==1:
            self.sequences.append(seq)
          else:
            self.sequences.append(seq2kmer(seq.upper(), kmer))
        
        assert len(self.V)==len(self.J)==len(self.D)==len(self.sequences), "unconsistent dimension"
        
        # resize cdr3 tensor
        # get max length
        #max_l = max([x.squeeze().numel() for x in self.cdr3])
        # pad 
        #self.cdr3 = [torch.nn.functional.pad(x, pad=(0, max_l-x.numel()), mode='constant', value=0) for x in self.cdr3]
        # stack
        #self.cdr3 = torch.stack(self.cdr3)

        self.V = torch.tensor(self.V)
        self.D = torch.tensor(self.D)
        self.J = torch.tensor(self.J)
        self.Va = torch.tensor(self.Va)
        self.Da = torch.tensor(self.Da)
        self.Ja = torch.tensor(self.Ja)
        #self.cdr3 = torch.tensor(self.cdr3)
        self.sequences = np.array(self.sequences)
        self.heads = np.array(self.heads)

        if target_V:
          mask = self.V == target_V
          self.V = self.V[mask]
          self.D = self.D[mask]
          self.J = self.J[mask]
          self.sequences = self.sequences[mask]

    def __getitem__(self, offset):

        v = self.V[offset]
        d = self.D[offset]
        j = self.J[offset]
        va = self.Va[offset]
        da = self.Da[offset]
        ja = self.Ja[offset]
        cdr3 = self.cdr3[offset]
        seq = self.sequences[offset]
        head = self.heads[offset]
        return seq, v, d, j, va, da, ja, cdr3, head

    def __len__(self):
        return len(self.V)

    def genes_collate_function(self, batch):

        (seqs, vs, ds, js, va, da, ja, cdr3, head) = zip(*batch)

        raw_seqs = deepcopy(seqs)

        seqs = self.tokenizer(list(seqs),padding=True, truncation=True, add_special_tokens=True)
        seqs = {k: torch.tensor(v).to(self.device) for k, v in seqs.items()}#default_collate(peptide)
        
        vs =  default_collate(vs).to(self.device)
        ds =  default_collate(ds).to(self.device)
        js =  default_collate(js).to(self.device)

        va =  default_collate(va).to(self.device)
        da =  default_collate(da).to(self.device)
        ja =  default_collate(ja).to(self.device)

        # resize cdr3 tensor
        # get max length
        max_l = max([x.squeeze().numel() for x in cdr3])
        # pad 
        cdr3 = [torch.nn.functional.pad(x, pad=(0, max_l-x.numel()), mode='constant', value=0) for x in cdr3]

        cdr3 = tuple(cdr3)

        cdr3 = default_collate(cdr3).to(self.device)

        #head = default_collate(head).to(self.device)

        return seqs, vs, ds, js, va, da, ja, raw_seqs, cdr3, head