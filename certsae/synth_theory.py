"""Controlled synthetic study to test the theory:
 - dictionary with tunable coherence via feature-splitting clusters
 - class signal carried by a sparse support S (|S| = s atoms)
 - nuisance activity on atoms *correlated* with S (makes Sigma non-diagonal)
Compare univariate ranking (md / anova) vs joint greedy (ols / somp_*) on:
 support recovery, rho_in, rho_out, and the certificate quantities.
Runs on CPU with many processes (joblib)."""
import os, json, itertools, argparse
import numpy as np, torch
from joblib import Parallel, delayed

def make_dict(d, m, n_clusters, spread, rng):
    C = rng.standard_normal((n_clusters, d)); C /= np.linalg.norm(C, axis=1, keepdims=True)
    lab = rng.integers(0, n_clusters, m)
    W = C[lab] + spread * rng.standard_normal((m, d)) / np.sqrt(d)
    W /= np.linalg.norm(W, axis=1, keepdims=True)
    return W, lab

def one(cfg):
    torch.set_num_threads(1)
    import analyze, saes
    d, m, n, c, s = cfg['d'], cfg['m'], cfg['n'], cfg['c'], cfg['s']
    rng = np.random.default_rng(cfg['seed'])
    W, lab = make_dict(d, m, cfg['nclust'], cfg['spread'], rng)
    G = W @ W.T; np.fill_diagonal(G, 0); mu = float(np.abs(G).max())
    S = rng.choice(m, s, replace=False)
    y = rng.integers(0, c, n)
    M = rng.standard_normal((c, s)) * cfg['snr']
    codes = M[y] + rng.standard_normal((n, s))
    # nuisance: atoms in the same clusters as S, strongly active but class-independent,
    # plus partially shared with support activity (creates off-diagonal covariance)
    nu = np.array([j for j in range(m) if lab[j] in set(lab[S]) and j not in set(S)])
    if len(nu) == 0:
        nu = rng.choice(m, 20, replace=False)
    nu = nu[:60]
    shared = rng.standard_normal((n, 1))
    ncodes = cfg['nuis'] * (rng.standard_normal((n, len(nu))) + shared)
    H = codes @ W[S] + ncodes @ W[nu] + cfg['noise'] * rng.standard_normal((n, d))
    # "baseline" distribution: generic activity without the task support
    B = (cfg['nuis'] * rng.standard_normal((n // 2, len(nu)))) @ W[nu] + cfg['noise'] * rng.standard_normal((n // 2, d))
    Ht = torch.tensor(H, dtype=torch.float32); Bt = torch.tensor(B, dtype=torch.float32); Wt = torch.tensor(W, dtype=torch.float32)
    sae = dict(W_dec=Wt, W_enc=Wt.T.clone(), b_enc=torch.zeros(m), thr=None, in_scale=1.0, pre_bias=None, topk=None)
    analyze.KS = [s, 2 * s]
    res = analyze.run(Ht, torch.tensor(y), Bt, Wt, sae, c, kmax=2 * s, seed=cfg['seed'])
    out = analyze.summarize(res)
    rec = {}
    for key, r in out['methods'].items():
        if r['sel']:
            rec[key] = len(set(r['sel'][:s]) & set(S.tolist())) / s
    out.update(cfg=cfg, coherence=mu, support_recovery=rec)
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default=os.path.join(os.environ.get('CERTSAE_ROOT', './outputs'), 'results', 'synth_theory.json'))
    ap.add_argument('--jobs', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    a = ap.parse_args()
    grid = []
    for spread, nuis, seed in itertools.product([0.3, 0.6, 1.0, 2.0, 8.0], [0.0, 1.0, 3.0], range(5)):
        grid.append(dict(d=128, m=1024, n=2000, c=4, s=8, nclust=64, spread=spread, nuis=nuis, snr=0.7, noise=0.3, seed=seed))
    res = Parallel(n_jobs=a.jobs, verbose=5)(delayed(one)(g) for g in grid)
    json.dump(res, open(a.out, 'w'), default=float)
    # compact summary
    import collections
    agg = collections.defaultdict(list)
    for r in res:
        key = (r['cfg']['spread'], r['cfg']['nuis'])
        agg[key].append(r)
    for key, rs in sorted(agg.items()):
        line = f"spread={key[0]:4} nuis={key[1]:3} mu={np.mean([r['coherence'] for r in rs]):.2f} |"
        for meth in ['md_task', 'anova', 'ols', 'somp_cov', 'somp_both']:
            k = f'{meth}|{rs[0]["cfg"]["s"]}'
            ri = np.mean([r['methods'][k]['rho_in'] for r in rs]); ro = np.mean([r['methods'][k]['rho_out'] for r in rs])
            sr = np.mean([r['support_recovery'].get(k, 0) for r in rs])
            line += f" {meth}: in={ri:.2f} out={ro:.2f} rec={sr:.2f} |"
        print(line, flush=True)

if __name__ == '__main__':
    main()
