"""
Runs training script across all pairs of domains among `domains`
"""
import os

# domains = ['EM', 'NE', 'NW', 'SW']

domain_pairs = [('EM', 'EM'), ('SW', 'SW'), ('NW', 'SW'), ('NW', 'EM')]
img_size_pair = (182, 182)

for s_domain, t_domain in domain_pairs:
        src_n, targ_n = img_size_pair
        # os.system(f'python train.py --source {s_domain} --target {t_domain}')
        s ="""#!/bin/bash
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

"""

        s += f'python train.py --source {s_domain} --target {t_domain} --source_num_imgs {src_n} --target_num_imgs {targ_n}\n'

        with open(f'training_scripts/{s_domain}_{t_domain}_{src_n}_{targ_n}.sh', 'w') as script:
            script.write(s)
