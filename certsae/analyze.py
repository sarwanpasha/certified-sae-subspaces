"""Pilot: joint vs univariate selection of SAE decoder atoms.

For each (target, layer): 5-fold CV. ALL selection is done on the training
fold only (no leakage). For each method we select k_max atoms once (greedy
methods are nested, so prefixes give smaller k) and evaluate, for k in KS:
  acc_in  : ridge-classifier accuracy on Q^T h  (Q = orthonormal basis of span W_S)
  acc_out : ridge-classifier accuracy on (I - QQ^T) h
  leak    : ||(I-QQ^T) Sigma_hy||_F / ||Sigma_hy||_F on the TEST fold
  cover   : ||(I-QQ^T) beta||_F / ||beta||_F   (beta from test-fold ridge)
Methods:
  md_task   top-k by mean SAE activation (task - alpaca)       [prior work]
  t_task    top-k by two-sample t (task vs alpaca)             [prior work]
  anova     top-k by univariate ANOVA F of SAE code vs target  [supervised univariate]
  ols       greedy orthogonal least squares over atoms on the probe target (joint, sufficiency)
  somp_cov  greedy simultaneous OMP of the cross-covariance Sigma_hy by atoms (necessity)
  somp_beta greedy SOMP of ridge readout beta (sufficiency, closed form)
  somp_both greedy SOMP of [Sigma_hy, beta]
  rand_atom k random SAE atoms; gauss: k-dim random Gaussian subspace
  dense_cov / dense_beta / dense_both : non-sparse references (rank <= 2(c-1))
"""
import os, json, argparse, time, math
import numpy as np, torch
import saes

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
KS = [1, 2, 5, 10, 20, 50, 100, 200]

# ---------------------------------------------------------------- probes
def ridge_acc(Xtr, ytr, Xte, yte, ncls):
    """Kernel (dual) ridge classifier on standardised features, alpha by LOO."""
    Xtr = Xtr.double(); Xte = Xte.double()
    mu = Xtr.mean(0); sd = Xtr.std(0).clamp_min(1e-6)
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    Y = torch.nn.functional.one_hot(ytr, ncls).double(); ybar = Y.mean(0); Yc = Y - ybar
    K = Xtr @ Xtr.T
    s, U = torch.linalg.eigh(K); s = s.clamp_min(0)
    UtY = U.T @ Yc
    scale = s.mean().item() + 1e-12
    best = None
    for a in np.logspace(-4, 3, 22) * scale:
        d = s / (s + a)
        Hdiag = (U * U * d).sum(1)
        fit = U @ (d[:, None] * UtY)
        loo = Yc - (Yc - fit) / (1 - Hdiag).clamp_min(1e-6)[:, None]
        err = ((loo + ybar).argmax(1) != ytr).double().mean().item()
        mse = ((Yc - fit) / (1 - Hdiag).clamp_min(1e-6)[:, None]).pow(2).mean().item()
        if best is None or (err, mse) < best[0]:
            best = ((err, mse), a)
    a = best[1]
    coef = U @ (UtY / (s + a)[:, None])          # dual coefficients
    pred = (Xte @ Xtr.T @ coef + ybar).argmax(1)
    return (pred == yte).double().mean().item()

def ridge_beta(Hc, Yc):
    """Primal readout beta (d x c) on raw centred activations, alpha by GCV."""
    Hc = Hc.double(); Yc = Yc.double()
    K = Hc @ Hc.T; s, U = torch.linalg.eigh(K); s = s.clamp_min(0); UtY = U.T @ Yc
    n = Hc.shape[0]; best = None
    for a in np.logspace(-4, 3, 22) * (s.mean().item() + 1e-12):
        d = s / (s + a); fit = U @ (d[:, None] * UtY)
        gcv = (Yc - fit).pow(2).sum().item() / (n * (1 - d.sum().item() / n) ** 2)
        if best is None or gcv < best[0]:
            best = (gcv, a)
    coef = U @ (UtY / (s + best[1])[:, None])
    return (Hc.T @ coef).float()

# ---------------------------------------------------------------- selection
def topk(score, k):
    return torch.topk(score, k).indices.tolist()

def sel_ols(Hc, Yc, W, kmax):
    """Greedy orthogonal least squares in data space over atom features a_i = Hc w_i."""
    A = Hc @ W.T                      # n x m
    E = Yc.clone().float()
    nrm0 = A.norm(dim=0).clamp_min(1e-8)
    S = []
    for _ in range(kmax):
        nr = A.norm(dim=0)
        sc = (E.T @ A).pow(2).sum(0) / nr.pow(2).clamp_min(1e-12)
        sc[nr < 1e-4 * nrm0] = -1; sc[S] = -1
        j = int(sc.argmax()); S.append(j)
        u = A[:, j] / A[:, j].norm()
        A -= torch.outer(u, u @ A); E -= torch.outer(u, u @ E)
    return S

def sel_somp(T, W, kmax):
    """Greedy (orthogonal-matching) simultaneous sparse approximation of the columns of T (d x p)."""
    T = T / T.norm(dim=0, keepdim=True).clamp_min(1e-12)
    R = W.clone(); Tr = T.clone(); S = []
    for _ in range(kmax):
        nr = R.norm(dim=1)
        sc = (R @ Tr).pow(2).sum(1) / nr.pow(2).clamp_min(1e-12)
        sc[nr < 1e-4] = -1; sc[S] = -1
        j = int(sc.argmax()); S.append(j)
        q = R[j] / R[j].norm()
        R -= torch.outer(R @ q, q); Tr -= torch.outer(q, q @ Tr)
    return S

def basis(V):
    """orthonormal basis (d x r) for columns of V (d x k), dropping tiny directions."""
    Q, Rr = torch.linalg.qr(V.double())
    keep = Rr.diagonal().abs() > 1e-6 * Rr.diagonal().abs().max()
    return Q[:, keep].float()

# ---------------------------------------------------------------- main
def run(task_H, y, alp_H, W, sae, ncls, kmax, seed=0, nfold=5):
    n, d = task_H.shape
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    folds = [perm[i::nfold] for i in range(nfold)]
    res = {}
    Z_task = saes.encode(sae, task_H); Z_alp = saes.encode(sae, alp_H)
    for fi, te in enumerate(folds):
        tr = torch.cat([folds[j] for j in range(nfold) if j != fi])
        Htr, Hte, ytr, yte = task_H[tr], task_H[te], y[tr], y[te]
        mu = Htr.mean(0); Hc = Htr - mu
        Ytr = torch.nn.functional.one_hot(ytr, ncls).float(); Yc = Ytr - Ytr.mean(0)
        Hte_c = Hte - Hte.mean(0); Yte = torch.nn.functional.one_hot(yte, ncls).float(); Yte_c = Yte - Yte.mean(0)
        Sxy_te = Hte_c.T @ Yte_c / len(te)
        beta_te = ridge_beta(Hte_c, Yte_c)
        Sxy = Hc.T @ Yc / len(tr); beta = ridge_beta(Hc, Yc)
        base = torch.bincount(yte, minlength=ncls).max().item() / len(te)
        full = ridge_acc(Htr, ytr, Hte, yte, ncls)
        res.setdefault('_full', []).append(full); res.setdefault('_base', []).append(base)
        # selections
        Zt = Z_task[tr]; za_m, za_v = Z_alp.mean(0), Z_alp.var(0)
        zt_m, zt_v = Zt.mean(0), Zt.var(0)
        cm = torch.stack([Zt[ytr == c].mean(0) if (ytr == c).any() else zt_m for c in range(ncls)])
        cnt = torch.bincount(ytr, minlength=ncls).float()[:, None]
        ssb = (cnt * (cm - zt_m).pow(2)).sum(0); ssw = ((Zt - cm[ytr]).pow(2)).sum(0)
        anova = (ssb / (ncls - 1)) / (ssw / (len(tr) - ncls) + 1e-8)
        anova[Zt.abs().sum(0) == 0] = -1
        sels = {
            'md_task': topk(zt_m - za_m, kmax),
            't_task': topk((zt_m - za_m) / torch.sqrt(zt_v / len(tr) + za_v / len(Z_alp) + 1e-8), kmax),
            'anova': topk(anova, kmax),
            'ols': sel_ols(Hc, Yc, W, kmax),
            'somp_cov': sel_somp(Sxy, W, kmax),
            'somp_beta': sel_somp(beta, W, kmax),
            'somp_both': sel_somp(torch.cat([Sxy / Sxy.norm(), beta / beta.norm()], 1), W, kmax),
            'rand_atom': torch.randperm(W.shape[0], generator=g)[:kmax].tolist(),
        }
        G = torch.randn(d, kmax, generator=g).to(task_H)
        subspaces = {}
        for name, S in sels.items():
            for k in KS:
                if k <= kmax:
                    subspaces[(name, k)] = (basis(W[S[:k]].T), S[:k])
        for k in KS:
            if k <= kmax:
                subspaces[('gauss', k)] = (basis(G[:, :k]), None)
        subspaces[('dense_cov', 0)] = (basis(Sxy), None)
        subspaces[('dense_beta', 0)] = (basis(beta), None)
        subspaces[('dense_both', 0)] = (basis(torch.cat([Sxy, beta], 1)), None)
        for (name, k), (Q, S) in subspaces.items():
            Xin_tr, Xin_te = Htr @ Q, Hte @ Q
            a_in = ridge_acc(Xin_tr, ytr, Xin_te, yte, ncls)
            Xo_tr = Htr - (Htr @ Q) @ Q.T; Xo_te = Hte - (Hte @ Q) @ Q.T
            a_out = ridge_acc(Xo_tr, ytr, Xo_te, yte, ncls)
            leak = ((Sxy_te - Q @ (Q.T @ Sxy_te)).norm() / Sxy_te.norm()).item()
            cover = ((beta_te - Q @ (Q.T @ beta_te)).norm() / beta_te.norm()).item()
            r = res.setdefault(f'{name}|{k}', dict(acc_in=[], acc_out=[], leak=[], cover=[], dim=Q.shape[1], sel=[]))
            r['acc_in'].append(a_in); r['acc_out'].append(a_out); r['leak'].append(leak); r['cover'].append(cover)
            if S is not None and fi == 0:
                r['sel'] = S
    return res

def summarize(res):
    full = np.mean(res['_full']); base = np.mean(res['_base'])
    out = dict(full=full, base=base, methods={})
    for key, r in res.items():
        if key.startswith('_'):
            continue
        ai, ao = np.array(r['acc_in']), np.array(r['acc_out'])
        out['methods'][key] = dict(acc_in=ai.mean(), acc_in_sd=ai.std(), acc_out=ao.mean(), acc_out_sd=ao.std(),
                                   rho_in=(ai.mean() - base) / (full - base + 1e-12),
                                   rho_out=(ao.mean() - base) / (full - base + 1e-12),
                                   leak=float(np.mean(r['leak'])), cover=float(np.mean(r['cover'])), dim=r['dim'], sel=r['sel'])
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.environ.get('CERTSAE_ROOT', './outputs'))
    ap.add_argument('--tag', default='gemma9b'); ap.add_argument('--family', default='gemma9b')
    ap.add_argument('--layers', type=int, nargs='+', required=True)
    ap.add_argument('--targets', default='mmlu_pred,mmlu_gold,mmlu_super,sst2_label,sst2_pred')
    ap.add_argument('--kmax', type=int, default=200)
    a = ap.parse_args()
    D = f'{a.root}/data/{a.tag}'; O = f'{a.root}/results/pilot_{a.tag}_{a.family}'; os.makedirs(O, exist_ok=True)
    mm = torch.load(f'{D}/mmlu.pt'); alp = torch.load(f'{D}/alpaca.pt')
    sst = torch.load(f'{D}/sst2.pt') if os.path.exists(f'{D}/sst2.pt') else None
    for L in a.layers:
        sae = saes.load(a.family, L)
        sae = {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in sae.items()}
        W = sae['W_dec']
        # GemmaScope layer L SAE reads the output of block L = hidden_states[L+1]
        A_H = alp['h'][:, L + 1].float().to(DEV)
        for t in a.targets.split(','):
            f = f'{O}/{t}_L{L}.json'
            if os.path.exists(f):
                continue
            t0 = time.time()
            if t.startswith('mmlu'):
                H = mm['h'][:, L + 1].float().to(DEV); y = mm[{'mmlu_pred': 'pred', 'mmlu_gold': 'gold', 'mmlu_super': 'supercat'}[t]].long().to(DEV); nc = 4
            else:
                if sst is None:
                    continue
                H = sst['h'][:, L + 1].float().to(DEV); y = sst['label' if t == 'sst2_label' else 'pred'].long().to(DEV); nc = 2
            res = run(H, y, A_H, W, sae, nc, a.kmax)
            s = summarize(res); s.update(target=t, layer=L, sae=sae['name'], secs=time.time() - t0)
            json.dump(s, open(f, 'w'), indent=1, default=float)
            m = s['methods']
            print(f"[{t} L{L}] full={s['full']:.3f} base={s['base']:.3f} ({s['secs']:.0f}s)", flush=True)
            for meth in ['md_task', 't_task', 'anova', 'ols', 'somp_cov', 'somp_beta', 'somp_both', 'rand_atom', 'gauss']:
                row = ' '.join(f"k{k}:{m[f'{meth}|{k}']['rho_in']:.2f}/{m[f'{meth}|{k}']['rho_out']:.2f}" for k in [5, 20, 50, 200] if f'{meth}|{k}' in m)
                print(f'   {meth:10s} rho_in/rho_out {row}', flush=True)
            for meth in ['dense_cov', 'dense_beta', 'dense_both']:
                r = m[f'{meth}|0']; print(f"   {meth:10s} dim={r['dim']} rho_in={r['rho_in']:.2f} rho_out={r['rho_out']:.2f}", flush=True)

if __name__ == '__main__':
    main()
