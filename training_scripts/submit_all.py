import os
import glob

for script in glob.glob('*.sh'):
    os.system(f'sbatch {script}')
