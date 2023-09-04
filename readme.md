# VDJdeep
# BCR sequence annotation using pre-trained DNABERT model

## Requirements
The models run on Pytorc, with various deep learning libraires.  
We provide a .yml file that contains all the tools and dependancies.

## Installation
We recommand using conda and creating a separate environment or run it on a cluster.  
You can create an appropriate environment with the command line :
```
conda env create -f torch.yml
```

## Usage
Several scripts are available depending on the task to be performed.

### Fine-tuning DNABERT (train a new model)
If you want to train a new model with your own dataset, there are are two possibilities depending on the approach: the classical by training 3 separate models or the Multi-CLS approach by training one model.  

- Classical approach :
You need to launch the file dnabert_finetune.py by specifying the training dataset, which type of gene you want to identify, the number of classes, and whether or not you want to consider alleles, etc. The training dataset has to be a .fasta. An example of the appropriate format is available [here](https://github.com/DradjatKevin/VDJdeep/blob/main/data/train/merge_shm_complete_cdr3v1.fasta)  
If you want to build a complete pipeline for the V(D)J assignment and CDR3 detection, you have to train 3 models; one for each type of gene.  
Examples are availables on the script [vdj_detection.sh](https://github.com/DradjatKevin/VDJdeep/blob/main/vdj_detection.sh) to launch the training of each type of model.  

- Multi-CLS approach : 
You need to launch the file dnabert_finetune_multicls.py by specifying the training dataset and whether or not you want to consider alleles.  
One training is sufficient to build a model for the V(D)J assignment and CDR3 detection.  
An example is available on the script [vdj_detection.sh](https://github.com/DradjatKevin/VDJdeep/blob/main/vdj_detection.sh) to launch the training of a model.

### Inference (use the trained model)
Depedning on the model used (Multi-CLS or simple), different scripts need to be used.  

- Simple model :
You need to launch the file dnabert_finetune_eval.py by specifying the model used and the testing dataset and whether or not you consider the alleles. You can also scpecify the output file.  
The testing file can be as the same format as the training dataset. If it is the case, the accuracy is computed directly. Else, you have to specify --nolabel in the command line.  
You can also choose to apply clustering before passing the sequences into the model by adding --cluster.  
An example is given in the script [vdj_test.py](https://github.com/DradjatKevin/VDJdeep/blob/main/vdj_test.sh).

- Multi-CLS model :
You need to launch the file dnabert_finetune_multicls_eval.py by specifying the model used and the testing dataset and whether or not you consider the alleles. You can also scpecify the output file.  
The same remarks can be applied to the input testing dataset as previously.
An example is given in the script [vdj_test.py](https://github.com/DradjatKevin/VDJdeep/blob/main/vdj_test.sh).



