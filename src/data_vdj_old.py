from torch.utils.data._utils.collate import default_collate
import sys
import torch
import torch.utils.data as data
import numpy as np
import pandas as pd
from copy import deepcopy
import random
import math

def add_ground_truth(filename) :
  with open(filename, 'r') as file :
    with open(filename[:len(filename)-6]+'_gt.fasta', 'w') as outfile :
      for line in file.readlines() :
        if line[0] == '>' :
          outfile.write(line.split('\n')[0] + f"|IGHV1-69*01|IGHD3-22*01|IGHJ4*02|\n")
        else :
          outfile.write(line)
  return 

def csv_to_fasta(filename) :
  file_df = pd.read_csv(filename, sep='\t')
  c = 0
  print(file_df['v_call'])
  with open(str(filename)[:len(filename)-4]+'.fasta', 'w') as outfile :
     for i in range(len(file_df)) :
      V, D, J = file_df['v_call'][i], file_df['d_call'][i], file_df['j_call'][i]
      outfile.write(f'>S{c}|{V}|{D}|{J}|mut'+'\n')
      outfile.write(file_df['sequence'][i].upper()+'\n')
      c += 1
  return

# add random nucleotides at start of the sequence (from tsv file)
def csv_to_fasta_insertions(filename) :
  file_df = pd.read_csv(filename, sep='\t')
  c = 0
  with open(str(filename)[:len(filename)-4]+'_insertions'+'.fasta', 'w') as outfile :
    for i in range(len(file_df)) :
      V, D, J = file_df['v_call'][i], file_df['d_call'][i], file_df['j_call'][i]

      # generate random prefix
      seq_length = random.randint(4, 12)
      random_seq = []
      nt_list = ['A', 'C', 'G', 'T']
      for i in range(seq_length) :
        random_index = random.randint(0,3)
        random_seq.append(nt_list[random_index])
      random_seq = ''.join(random_seq)

      outfile.write(f'>S{c}|{V}|{D}|{J}|mut'+'\n')
      outfile.write(random_seq + file_df['sequence'][i].upper()+'\n')

      c += 1
  return


# add random nt at start of the sequence (from fasta file)
def insertions_fasta(filename) :
  c = 0
  with open(filename, 'r') as file :
    with open(str(filename)[:len(filename)-6]+'_insertions'+'.fasta', 'w') as outfile :
      for line in file.readlines() :
        if line[0] == '>' :
          outfile.write(line)
        else :
          # generate random prefix
          seq_length = random.randint(20, 40)
          random_seq = []
          nt_list = ['A', 'C', 'G', 'T']
          for i in range(seq_length) :
            random_index = random.randint(0,3)
            random_seq.append(nt_list[random_index])
          random_seq = ''.join(random_seq)

          outfile.write(random_seq + line)
  return


def insertions_fasta_end(filename) :
  c = 0
  with open(filename, 'r') as file :
    with open(str(filename)[:len(filename)-6]+'_insertions_end'+'.fasta', 'w') as outfile :
      for line in file.readlines() :
        if line[0] == '>' :
          outfile.write(line)
        else :
          # generate random suffix
          seq_length = random.randint(0, 10)
          random_seq = []
          nt_list = ['A', 'C', 'G', 'T']
          for i in range(seq_length) :
            random_index = random.randint(0,3)
            random_seq.append(nt_list[random_index])
          random_seq = ''.join(random_seq)

          outfile.write(line.split('\n')[0] + random_seq+'\n')
  return

# tsv file as input
def preprocess_imgt_bert(filename) :
  file_df = pd.read_csv(filename, sep='\t')
  c = 0
  with open(str(filename)[:len(filename)-4]+'.fasta', 'w') as outfile :
     for i in range(len(file_df)) :
      V, D, J = file_df['v_call'][i], file_df['d_call'][i], file_df['j_call'][i]
      if type(V) != float and type(D) != float and type(J) != float :
        #print(V, D, J)
        # remove 'Homsap' and 'or'
        V = V.split(' ')[1]
        D = D.split(' ')[1]
        J = J.split(' ')[1]
        outfile.write(f'>S{c}|{V}|{D}|{J}|mut'+'\n')
        # sequence : 
        outfile.write(file_df['sequence'][i].upper()+'\n')
        c += 1
  return

# tsv file as input
def preprocess_imgt_cnn(filename) :
  file_df = pd.read_csv(filename, sep='\t')
  c = 0
  with open(str(filename)[:len(filename)-4]+'.fasta', 'w') as outfile :
     for i in range(len(file_df)) :
      V, D, J = file_df['v_call'][i], file_df['d_call'][i], file_df['j_call'][i]
      if type(V) != float and type(D) != float and type(J) != float :
        # remove 'Homsap' and 'or'
        V = V.split(' ')[1]
        D = D.split(' ')[1]
        J = J.split(' ')[1]
        outfile.write(f'>S{c}|{V}|{D}|{J}|mut'+'\n')
        # sequence : transform N to A
        seq = file_df['sequence'][i]
        new_seq = []
        for i in range(len(seq)) :
          if seq[i] == 'n' :
            new_seq.append('a')
          else :
            new_seq.append(seq[i])
        new_seq = ''.join(new_seq)
        outfile.write(new_seq.upper()+'\n')
        c += 1
  return


# remove identical sequences : fasta as input
def remove_identical_seq(filename) :
  c = 0
  sequences, headers = [], []
  with open(filename, 'r') as file :
    for line in file.readlines() :
      if line[0] != '>' :
        sequences.append(line.split('\n')[0])
      else :
        headers.append(line.split('\n')[0])
  # write
  new_seq = []
  with open(filename+'removeid', 'w') as outfile :
    for head, seq in zip(headers, sequences) :
      if seq not in new_seq :
        new_seq.append(seq)
        outfile.write(f'{head}\n{seq}\n')
  return

# tsv as input
def get_D_part(filename) :
  file_df = pd.read_csv(filename, sep='\t')
  c = 0
  with open(str(filename)[:len(filename)-4]+'_onlyD.fasta', 'w') as outfile :
     for i in range(len(file_df)) :
      seq = file_df['d_sequence'][i].upper()
      if len(seq) >= 3 :
        V, D, J = file_df['v_call'][i], file_df['d_call'][i], file_df['j_call'][i]
        outfile.write(f'>S{c}|{V}|{D}|{J}|mut'+'\n')
        outfile.write(seq+'\n')
        c += 1
  return


# preprocess IMPLAntS output simulator
def preprocess_implants(filename) :
  k = 1
  with open(filename, 'r') as file :
    with open(filename+'_pp.fasta', 'w') as outfile :
      for line in file.readlines() :
        if line[0] == '>' :
          V, D, J = line.split(':')[0][1:], line.split(':')[1], line.split(':')[2]
          outfile.write(f'>S{k}|{V}|{D}|{J}\n')
          k += 1
        else :
          outfile.write(line)
  return



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


def keep_genes_in_dict(header, sequence, allele=False) :
  new_header, new_sequence = [], []
  if allele :
    v_genes_dict = np.load('data/dict/v_alleles_276.npy', allow_pickle='True').item()
    d_genes_dict = np.load('data/dict/d_alleles_37.npy', allow_pickle='True').item()
    j_genes_dict = np.load('data/dict/j_alleles_11.npy', allow_pickle='True').item()
  else :
    v_genes_dict = np.load('data/dict/v_genes_75.npy', allow_pickle='True').item()
    d_genes_dict = np.load('data/dict/d_genes_30.npy', allow_pickle='True').item()
    j_genes_dict = np.load('data/dict/j_genes_6.npy', allow_pickle='True').item()
  # keep V, D, J genes that are in dict 
  for i in range(len(header)) :
    v_gene_allele = header[i].split('|')[1]
    d_gene_allele = header[i].split('|')[2]
    j_gene_allele = header[i].split('|')[3]
    if allele :
      v_gene = v_gene_allele
      d_gene = d_gene_allele
      j_gene = j_gene_allele
    else : 
      v_gene = v_gene_allele.split('*')[0]
      d_gene = d_gene_allele.split('*')[0]
      j_gene = j_gene_allele.split('*')[0]
    if v_gene in v_genes_dict.keys() and d_gene in d_genes_dict.keys() and j_gene in j_genes_dict.keys() :
      new_header.append(header[i])
      new_sequence.append(sequence[i])
  return new_header, new_sequence



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
  

  return V, D, J, Va, Da, Ja
    




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
        self.cdr3_S = []
        self.cdr3_E = []

        for name, seq in zip(names, seqs):
          v,d,j, va, da, ja= parse_name(name, allele=allele)
          self.V.append(v)
          self.J.append(j)
          self.D.append(d)
          self.Va.append(va)
          self.Da.append(da)
          self.Ja.append(ja)
          if kmer==1:
            self.sequences.append(seq)
          else:
            self.sequences.append(seq2kmer(seq.upper(), kmer))
        
        assert len(self.V)==len(self.J)==len(self.D)==len(self.sequences), "unconsistent dimension"

        self.V = torch.tensor(self.V)
        self.D = torch.tensor(self.D)
        self.J = torch.tensor(self.J)
        self.Va = torch.tensor(self.Va)
        self.Da = torch.tensor(self.Da)
        self.Ja = torch.tensor(self.Ja)
        self.sequences = np.array(self.sequences)
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
        seq = self.sequences[offset]
        return seq, v, d, j, va, da, ja

    def __len__(self):
        return len(self.V)

    # def set_reweight(self,alpha):
    #     freq = self.df["peptide"].value_counts()/self.df["peptide"].value_counts().sum()
    #     alpha = alpha
    #     freq = alpha*freq + (1-alpha)/len(self.df["peptide"].value_counts())
    #     self.weights = (1/torch.tensor(list(self.df.apply(lambda x: freq[x["peptide"]],1 ))))/len(self.df["peptide"].value_counts())
    #     self.reweight = True



    def genes_collate_function(self, batch):

        (seqs, vs, ds, js, va, da, ja) = zip(*batch)

        raw_seqs = deepcopy(seqs)

        seqs = self.tokenizer(list(seqs),padding="longest", truncation=True, add_special_tokens=True)
        seqs = {k: torch.tensor(v).to(self.device) for k, v in seqs.items()}#default_collate(peptide)
        
        vs =  default_collate(vs).to(self.device)
        ds =  default_collate(ds).to(self.device)
        js =  default_collate(js).to(self.device)

        va =  default_collate(va).to(self.device)
        da =  default_collate(da).to(self.device)
        ja =  default_collate(ja).to(self.device)


        return seqs, vs, ds, js, va, da, ja, raw_seqs




### no label ###
class VDJDataset_nolabel(data.Dataset):

    def __init__(self, fasta_file, tokenizer, device, target_V=None, target_D=None, target_J=None, kmer=1, allele=False, nb_seq=10000, max_len='max', rev=False):#, alpha_maxlength, beta_maxlength, epitope_maxlength):
        self.device=device
        self.tokenizer = tokenizer
#         self.alpha_maxlength = alpha_maxlength
#         self.beta_maxlength = beta_maxlength
#         self.epitope_maxlength = epitope_maxlength

        print("Loading and Tokenizing the data ...")

        names, seqs = parse_fasta(fasta_file, stop=nb_seq, max_len=max_len, rev=rev)
        self.sequences = []

        for name, seq in zip(names, seqs):
          if kmer==1:
            self.sequences.append(seq)
          else:
            self.sequences.append(seq2kmer(seq, kmer))

        self.sequences = np.array(self.sequences)

    def __getitem__(self, offset):
        seq = self.sequences[offset]
        return seq

    def __len__(self):
        return len(self.sequences)
    

    def genes_collate_function(self, batch):

        seqs = batch

        raw_seqs = deepcopy(seqs)

        seqs = self.tokenizer(list(seqs),padding="longest", truncation=True, add_special_tokens=True)
        seqs = {k: torch.tensor(v).to(self.device) for k, v in seqs.items()}#default_collate(peptide)
        

        return seqs, raw_seqs