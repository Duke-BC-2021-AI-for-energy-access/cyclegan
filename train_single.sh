#!/bin/bash
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=yl708@duke.edu     # Where to send mail	
#SBATCH --partition=scavenger-gpu
#SBATCH --exclusive
#SBATCH --time='7-0'

source ~/.bashrc
source ~/.bash_profile

conda activate torch

# if [[ $SLURM_ARRAY_TASK_ID -eq 1 ]]
# then
# 	rm folders.txt
# 	cp folders.txt.bak folders.txt
# fi

PYTHONPATH="${PYTHONPATH}:/work/yl708/bio-image-unet" python train_single.py
