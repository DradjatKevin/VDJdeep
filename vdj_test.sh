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

echo 'IG alleles'
for file in data/IG/cdr3/*.fasta
do
echo $file
filename=$(basename "$file")
python dnabert_finetune_eval_multicls.py --model 'models/model_multiCLS_alleles_50epoch.pt' --allele --test_dir $file --kmer 3 --batch_size 32 --weight_decay 0.0 --nb_seq_max 200000 --output "output_alleles_$filename.txt"
#python dnabert_finetune_eval.py --model 'model_shmComplete_cdr3End.pt' --test_dir $file --type 'cdr3_end' --kmer 3 --batch_size 256 --weight_decay 0.0 --nb_seq_max 200000 --output "output_cdr3End_$filename.txt"
done

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


#echo 'fastBCR alleles cluster'
#for file in data/fastBCR/*no_noise.fasta
#do
#echo $file
#filename=$(basename "$file")
#python dnabert_finetune_eval.py --model 'models/model_shmComplete_Vallele_newdict.pt' --type 'V' --test_dir $file --kmer 3 --batch_size 256 --weight_decay 0.0 --nb_seq_max 150000 --output "output_Valleles_$filename.txt" --allele --nolabel --cluster
#python dnabert_finetune_eval.py --model 'models/model_shmComplete_Dalleles.pt' --type 'D' --test_dir $file --kmer 3 --batch_size 256 --weight_decay 0.0 --nb_seq_max 150000 --output "output_Dalleles_$filename.txt" --allele --nolabel --cluster
#python dnabert_finetune_eval.py --model 'models/model_shmComplete_Jalleles_rev.pt' --type 'J' --test_dir $file --kmer 3 --batch_size 256 --weight_decay 0.0 --nb_seq_max 150000 --output "output_Jalleles_$filename.txt" --allele --nolabel --cluster
#done



#echo 'gene test AS'
#for t in $(seq 0.1 0.1 0.6)
#do
#for file in data/airrship/test_shm_fasta/*.fasta
#do
#echo $file $t
#python dnabert_finetune_eval.py --model 'model_shmcomplete_gene.pt' --test_dir $file --kmer 3 --batch_size 32 --weight_decay 0.0 --nb_seq_max 120000 --align --threshold $t
#done
#done

#echo 'test IMPlAntS'
#python dnabert_finetune_eval.py --model 'models/model_shmComplete_Vgene_newdict.pt' --test_dir 'data/implants/implants_10K.fasta' --kmer 3 --batch_size 32 --weight_decay 0.0 --nb_seq_max 200000 --error_file "error_file_imgt.txt"

#python dnabert_finetune_eval.py --model 'model_shmcomplete_gene.pt' --test_dir $file --kmer 3 --batch_size 32 --weight_decay 0.0 --nb_seq_max 120000 


#echo 'real data V genes'
#for file in data/real/*.fasta
#do 
#echo $file
#filename=$(basename "$file")
#python dnabert_finetune_eval.py --model 'model_shmInsertionsStartEnd_V_genes.pt' --type 'V' --test_dir $file --kmer 3 --batch_size 256 --weight_decay 0.0 --nb_seq_max 5000000 --error_file "error_$filename.txt" --output "outputStartEnd_$filename.txt"
#echo ' '
#done
#done



#echo 'real data J genes'
#for file in data/real/*.fasta
#do
#echo $file
#filename=$(basename "$file")
#python dnabert_finetune_eval.py --model 'model_shmInsertionsStartEnd_J_genes.pt' --type 'J' --test_dir $file --kmer 3 --batch_size 32 --weight_decay 0.0 --nb_seq_max 5000000 --error_file "error_$filename.txt" --output "outputStartEnd_Jgenes_$filename.txt"
#echo ' '
#done
#done



#echo 'onlyD genes eval AS'
#for file in data/airrship/onlyD/all*.fasta
#do 
#echo $file
#filename=$(basename "$file")
#python dnabert_finetune_eval.py --model 'model_onlyD_D_genes.pt' --type 'D' --test_dir $file --kmer 3 --batch_size 32 --weight_decay 0.0 --nb_seq_max 5000000 --error_file "error_$filename.txt"
#echo ' '
#done
#done



#echo 'het test V alleles'
#echo 'no mut'
#python dnabert_finetune_eval.py --model 'models/model_shmComplete_Vallele_newdict.pt' --type 'V' --test_dir 'data/airrship/het/one_patient_nomut.fasta' --kmer 3 --batch_size 512 --weight_decay 0.0 --nb_seq_max 150000 --het --allele
#echo 'shm'
#python dnabert_finetune_eval.py --model 'models/model_shmComplete_Vallele_newdict.pt' --type 'V' --test_dir 'data/airrship/het/one_patient_shm.fasta' --kmer 3 --batch_size 512 --weight_decay 0.0 --nb_seq_max 150000 --het --allele
