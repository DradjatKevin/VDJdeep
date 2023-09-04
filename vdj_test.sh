#!/bin/bash
#
#SBATCH --partition=gpu                       # partition
#SBATCH --gres=gpu:7g.40gb:1
#SBATCH -N 1                         # nombre de nœuds
#SBATCH -n 1                         # nombre de cœurs
#SBATCH --mem 100GB                    # mémoire vive pour l'ensemble des cœurs
#SBATCH -t 0-24:00                    # durée maximum du travail (D-HH:MM)
#SBATCH -o slurm.%N.%j.MultiCLS.IG_alleles.out           # STDOUT
#SBATCH -e slurm.%N.%j.MultiCLS.IG_alleles.err           # STDERR

module load python/3.9
module load mmseqs2/14.7e284

### Multi_CLS
echo 'IG alleles'
for file in data/IG/cdr3/*.fasta
do
echo $file
filename=$(basename "$file")
python dnabert_finetune_eval_multicls.py --model 'models/model_multiCLS_alleles_50epoch.pt' --allele --test_dir $file --kmer 3 --batch_size 32 --weight_decay 0.0 --nb_seq_max 200000 --output "output_alleles_$filename.txt"
done

### Simple Models
#echo 'AMR1 details'
#for t in $(seq 0.1 0.1 0.6)
#do
#for file in data/fastBCR/*.fasta
#do
#echo $file 
#filename=$(basename "$file")
#python dnabert_finetune_eval.py --model 'models/model_shmCompleteInsertionsStartEnd_Vallele_newdict.pt' --type 'V' --test_dir $file --kmer 3 --batch_size 256 --weight_decay 0.0 --nb_seq_max 150000 --output "output_Valleles_$filename.txt" --nolabel --allele --cluster
#python dnabert_finetune_eval.py --model 'models/model_shmCompleteInsertionsStartEnd_Jalleles_rev.pt' --type 'J' --test_dir $file --kmer 3 --batch_size 256 --weight_decay 0.0 --nb_seq_max 150000 --output "output_Jalleles_$filename.txt" --nolabel --allele --cluster
#python dnabert_finetune_eval.py --model 'models/model_shmCompleteInsertionStartEnd_Dalleles.pt' --type 'D' --test_dir $file --kmer 3 --batch_size 256 --weight_decay 0.0 --nb_seq_max 150000 --output "output_Dalleles_$filename.txt" --nolabel --allele --cluster 
#echo ' '
#done
#done

