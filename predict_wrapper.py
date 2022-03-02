import os

domains = ['EM', 'NE', 'NW', 'SW']

for s_domain in domains:
    for t_domain in domains:
        if s_domain == t_domain:
            continue
        if not (s_domain == 'NE' and t_domain == 'NW'):
            continue
        os.system(f'python predict.py --source {s_domain} --target {t_domain}')
