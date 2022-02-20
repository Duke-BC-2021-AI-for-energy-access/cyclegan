"""
Runs training script across all pairs of domains among `domains`
"""
import os

domains = ['EM', 'NE', 'NW', 'SW']

for s_domain in domains:
    for t_domain in domains:
        if s_domain == t_domain:
            continue
        os.system(f'python train.py --source {s_domain} --target {t_domain}')
