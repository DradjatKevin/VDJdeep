#!/bin/bash
#
#SBATCH --partition=gpu                       # partition
#SBATCH --gres=gpu:7g.40gb:1
#SBATCH -N 1                         # nombre de nœuds
#SBATCH -n 1                         # nombre de cœurs
#SBATCH --mem 100GB                    # mémoire vive pour l'ensemble des cœurs
#SBATCH -t 3-0:00                    # durée maximum du travail (D-HH:MM)
#SBATCH -o slurm.%N.%j.MultiCLS.ASTrain_alleles.out           # STDOUT
#SBATCH -e slurm.%N.%j.MultiCLS.ASTrain_alleles.err           # STDERR


module load python/3.9

echo 'Multi CLS and Token classification'
python dnabert_finetune_multicls.py --train_dir 'data/airrship/merge_shm_complete_cdr3v1.fasta' --kmer 3 --batch_size 8 --weight_decay 0.0 --epoch 50 --allele --nb_classes_v 270 --nb_classes_d 38 --nb_classes_j 12 --nb_seq_max 200000 --save_model --save_name 'model_multiCLS_alleles_50epoch.pt' 

#echo 'V allele classification'
#python dnabert_finetune.py --train_dir 'data/airrship/merge_shm_complete_insertions_start_end.fasta' --kmer 3 --batch_size 32 --weight_decay 0.0 --epoch 100 --allele --nb_classes 270 --nb_seq_max 200000 --type 'V' --save_model --save_name 'model_shmCompleteInsertionsSatrtEnd_Vallele_newdict.pt'

#echo 'D gene classification'
#python dnabert_finetune.py --train_dir 'data/airrship/onlyD/merge_onlyD.fasta' --kmer 3 --batch_size 32 --weight_decay 0.0 --epoch 100 --nb_classes 31 --nb_seq_max 150000 --type 'D' --save_model --save_name 'model_onlyD_D_genes.pt' --wandb

#echo 'D allele classification'
#python dnabert_finetune.py --train_dir 'data/airrship/merge_shm_complete_insertions_start_end.fasta' --kmer 3 --batch_size 32 --weight_decay 0.0 --epoch 100 --allele --nb_classes 38 --nb_seq_max 150000 --type 'D' --save_model --save_name "model_shmCompleteInsertionStartEnd_Dalleles.pt" 

#echo 'J gene classification'
#python dnabert_finetune.py --train_dir 'data/airrship/merge_shm_complete_insertions_start_end.fasta' --kmer 3 --batch_size 32 --weight_decay 0.0 --epoch 50 --nb_classes 7 --nb_seq_max 150000 --type 'J' --save_model --save_name 'model_shmInsertionsStartEnd_J_genes.pt' 

#echo 'J allele classification'
#python dnabert_finetune.py --train_dir 'data/airrship/merge_shm_complete_insertions_start_end.fasta' --kmer 3 --batch_size 32 --weight_decay 0.0 --epoch 100 --allele --nb_classes 12 --nb_seq_max 200000 --type 'J' --save_model --save_name "model_shmCompleteInsertionsStartEnd_Jalleles_rev.pt" 

#echo '10Knomut 20Kmut'
#echo 'D gene classification'
#python dnabert_finetune.py --train_dir 'data/airrship/merge_data_10Knomut_20Kmut.fasta' --kmer 3 --batch_size 32 --weight_decay 0.0 --epoch 50 --nb_classes 31 --nb_seq_max 150000 --all_test --type 'D'

#echo 'D allele classification'
#python dnabert_finetune.py --train_dir 'data/airrship/merge_data_10Knomut_20Kmut.fasta' --kmer 3 --batch_size 32 --weight_decay 0.0 --epoch 50 --allele --nb_classes 38 --nb_seq_max 150000 --all_test --type 'D'

#echo 'J gene classification'
#python dnabert_finetune.py --train_dir 'data/airrship/merge_data_10Knomut_20Kmut.fasta' --kmer 3 --batch_size 32 --weight_decay 0.0 --epoch 50 --nb_classes 7 --nb_seq_max 150000 --all_test --type 'J'

#echo 'J allele classification'
#python dnabert_finetune.py --train_dir 'data/airrship/merge_data_10Knomut_20Kmut.fasta' --kmer 3 --batch_size 32 --weight_decay 0.0 --epoch 50 --allele --nb_classes 12 --nb_seq_max 150000 --all_test --type 'J'

#v1
#echo 'cdr3 start'
#python dnabert_finetune_cdr.py --train_dir 'data/airrship/merge_shm_complete_cdr3v1.fasta' --kmer 3 --batch_size 32 --weight_decay 0.0 --epoch 80 --nb_classes 430 --nb_seq_max 200000 --type 'cdr3_start' --save_model --save_name 'model_shmComplete_cdr3Startv1.pt' --wandb


#v2
#echo 'cdr3 start'
#python dnabert_finetune_cdr.py --train_dir 'data/airrship/merge_shm_complete_cdr3v2.fasta' --kmer 3 --batch_size 32 --weight_decay 0.0 --epoch 80 --nb_classes 400 --nb_seq_max 200000 --type 'cdr3_start' --save_model --save_name 'model_shmComplete_cdr3Startv2.pt'

#echo 'cdr3 end'
#python dnabert_finetune_cdr.py --train_dir 'data/airrship/merge_shm_complete_cdr3v2.fasta' --kmer 3 --batch_size 32 --weight_decay 0.0 --epoch 80 --nb_classes 150 --nb_seq_max 200000 --type 'cdr3_end' --save_model --save_name 'model_shmComplete_cdr3Endv2.pt' 
