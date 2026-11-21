"""Compact text summary of all pilot results -> results/pilot_summary.txt"""
import json, glob, os, re, collections
R = os.path.join(os.environ.get('CERTSAE_ROOT', './outputs'), 'results')
out = []
METH = ['md_task', 't_task', 'anova', 'ols', 'somp_cov', 'somp_beta', 'somp_both', 'rand_atom', 'gauss']
for d in sorted(glob.glob(f'{R}/pilot_*')):
    files = glob.glob(f'{d}/*.json')
    by_t = collections.defaultdict(list)
    for f in files:
        j = json.load(open(f)); by_t[j['target']].append(j)
    for t, js in sorted(by_t.items()):
        out.append(f'\n#### {os.path.basename(d)}  target={t}   cells: rho_in/rho_out (k=5 | 20 | 50)')
        for j in sorted(js, key=lambda x: x['layer']):
            m = j['methods']
            out.append(f"L{j['layer']:>2} full={j['full']:.3f} base={j['base']:.3f} " +
                       ' '.join(f"{k}:dim{m[k+'|0']['dim']} {m[k+'|0']['rho_in']:.2f}/{m[k+'|0']['rho_out']:.2f}" for k in ['dense_cov', 'dense_beta', 'dense_both']))
            for meth in METH:
                cells = []
                for k in [5, 20, 50]:
                    r = m.get(f'{meth}|{k}')
                    cells.append(f"{r['rho_in']:5.2f}/{r['rho_out']:5.2f}" if r else '   -   ')
                r50 = m.get(f'{meth}|50', {})
                out.append(f"    {meth:9s} {' | '.join(cells)}   leak50={r50.get('leak', float('nan')):.2f} cover50={r50.get('cover', float('nan')):.2f}")
# synthetic
f = f'{R}/synth_theory.json'
if os.path.exists(f):
    import numpy as np
    res = json.load(open(f)); agg = collections.defaultdict(list)
    for r in res: agg[(r['cfg']['spread'], r['cfg']['nuis'])].append(r)
    out.append('\n#### synthetic (s=8): rho_in/rho_out/support-recovery at k=s')
    for key, rs in sorted(agg.items()):
        s = rs[0]['cfg']['s']; line = f"spread={key[0]:4} nuis={key[1]:3} mu={np.mean([r['coherence'] for r in rs]):.2f} full={np.mean([r['full'] for r in rs]):.2f}"
        for meth in ['md_task', 'anova', 'ols', 'somp_cov', 'somp_beta', 'somp_both']:
            k = f'{meth}|{s}'
            line += f" | {meth} {np.mean([r['methods'][k]['rho_in'] for r in rs]):.2f}/{np.mean([r['methods'][k]['rho_out'] for r in rs]):.2f}/{np.mean([r['support_recovery'].get(k, 0) for r in rs]):.2f}"
        out.append(line)
open(f'{R}/pilot_summary.txt', 'w').write('\n'.join(out))
print('\n'.join(out))
