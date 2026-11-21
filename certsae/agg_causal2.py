"""Aggregate causal2 results across folds (merging chunk files) -> results/causal2_summary.{txt,json}"""
import json, glob, os, re, collections
import numpy as np
R = os.path.join(os.environ.get('CERTSAE_ROOT', './outputs'), 'results')
MET = ['flip', 'acc', 'swap_follow', 'sst_flip', 'alp_kl']
SHOW = ['md_task|5', 'md_task|20', 'md_task|50', 'anova|5', 'anova|20', 'anova|50', 'ols|5', 'ols|20', 'ols|50',
        'somp_cov|1', 'somp_cov|2', 'somp_cov|5', 'somp_cov|10', 'somp_cov|20', 'somp_cov|50', 'somp_cov|100',
        'somp_both|20', 'rand_atom|50', 'gauss|50', 'dense_cov|0', 'dense_beta|0', 'dense_both|0', 'leace_cov|0',
        'leace_somp|5', 'leace_somp|20', 'leace_somp|50', 'leace_somp|100']
out_txt, out_json = [], {}
for d in sorted(glob.glob(f'{R}/causal2_*')):
    model = os.path.basename(d).replace('causal2_', '')
    runs = collections.defaultdict(lambda: collections.defaultdict(dict))  # layer -> fold -> methods
    clean = collections.defaultdict(dict)
    for f in glob.glob(f'{d}/L*_f*.json'):
        m = re.match(r'L(\d+)_f(\d+)(?:_c\d+of\d+)?\.json', os.path.basename(f))
        if not m: continue
        L, fo = int(m.group(1)), int(m.group(2))
        j = json.load(open(f))
        runs[L][fo].update(j['methods']); clean[L][fo] = j['clean_acc']
    for L in sorted(runs):
        folds = sorted(runs[L])
        out_txt.append(f'\n#### {model} L{L}  folds={folds}  clean_acc={np.mean([clean[L][f] for f in folds]):.3f}')
        out_txt.append(f"{'method':15s} " + ' '.join(f'{m:>13s}' for m in MET) + '  nfold')
        keys = [k for k in SHOW if any(k in runs[L][f] for f in folds)]
        for k in keys:
            vals = {m: [runs[L][f][k][m] for f in folds if k in runs[L][f]] for m in MET}
            nf = len(vals['flip'])
            out_json.setdefault(model, {}).setdefault(str(L), {})[k] = {m: (float(np.mean(v)), float(np.std(v))) for m, v in vals.items()} | {'nfold': nf}
            out_txt.append(f'{k:15s} ' + ' '.join(f'{np.mean(v):6.3f}±{np.std(v):5.3f}' for v in vals.values()) + f'  {nf}')
open(f'{R}/causal2_summary.txt', 'w').write('\n'.join(out_txt))
json.dump(out_json, open(f'{R}/causal2_summary.json', 'w'), indent=1)
print('\n'.join(out_txt))
