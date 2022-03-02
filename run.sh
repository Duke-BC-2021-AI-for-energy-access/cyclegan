#!/bin/bash
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=yl708@duke.edu     # Where to send mail 
#SBATCH --partition=scavenger-gpu
#SBATCH --exclusive
#SBATCH --time='7-0'
#SBATCH --chdir='/work/yl708/bass/cyclegan'
#SBATCH --mem=0

source ~/.bashrc
source ~/.bash_profile
cd /work/yl708/bass/cyclegan

conda activate cyclegan

# python train.py --source SW --target EM --source_num_imgs 182 --target_num_imgs 182
# python train.py --source EM --target SW --source_num_imgs 182 --target_num_imgs 182

python predict.py --source SW --target EM
python predict.py --source EM --target SW
