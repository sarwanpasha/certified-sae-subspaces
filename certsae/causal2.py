"""Causal round 2: full 5-fold CV, more k, LEACE-style oblique eraser, specificity controls.

For (model, layer, fold): select atoms on the TRAIN split (same selection code as analyze.py),
then on the held-out MMLU prompts of that fold apply interventions at layer L:
  abl   : h <- E(h) at ALL token positions (E = orthogonal removal or LEACE oblique eraser)
  swap  : last token only, replace the subspace component by that of a source prompt whose
          clean prediction differs; success = model outputs the source's letter
Specificity (same 'abl' intervention, all positions):
  sst2  : flip rate of the model's SST-2 sentiment prediction (300 prompts)
  alpaca: mean KL(clean || ablated) of the full next-token distribution (200 prompts)
Erasers:
  orth  : E(h) = h - QQ^T h
  leace : E(h) = h - W^+ P_{W S} W (h - mu),  W = (Sigma + lam I)^{-1/2}, S = Cov(h, Y)
          (Belrose et al. 2023); swap for leace: h + W^+ P W (h_src - h)
"""
import os, json, argparse, random, time
import torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import saes, analyze
from extract import mmlu_prompt

KS = [1, 2, 5, 10, 20, 50, 100]

def basis(V):
    Q, R = torch.linalg.qr(V.double()); d = R.diagonal().abs()
    return Q[:, d > 1e-6 * d.max()].float()

def leace_ops(H, Y):
    """returns (mu, A) with eraser E(h) = h - (h - mu) @ A^T, A = W^+ P W."""
    H = H.double(); mu = H.mean(0); Hc = H - mu
    Yc = Y.double() - Y.double().mean(0)
    S = Hc.T @ Yc / len(H)                      # d x c
    C = Hc.T @ Hc / len(H)
    lam = 1e-3 * C.trace() / C.shape[0]
    ev, U = torch.linalg.eigh(C + lam * torch.eye(C.shape[0], dtype=C.dtype, device=C.device))
    W = U @ torch.diag(ev.rsqrt()) @ U.T; Wi = U @ torch.diag(ev.sqrt()) @ U.T
    Qs = basis(W @ S).double()
    A = Wi @ Qs @ Qs.T @ W
    return mu.float(), A.float()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.environ.get('CERTSAE_ROOT', './outputs'))
    ap.add_argument('--model', required=True); ap.add_argument('--tag', required=True); ap.add_argument('--family', required=True)
    ap.add_argument('--layer', type=int, required=True); ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--chunk', type=int, default=0); ap.add_argument('--nchunks', type=int, default=1)
    ap.add_argument('--methods', default='md_task,anova,ols,somp_cov,somp_both,rand_atom')
    a = ap.parse_args()
    L, DEV = a.layer, 'cuda'
    outd = f'{a.root}/results/causal2_{a.tag}_{a.family}'; os.makedirs(outd, exist_ok=True)
    outf = f'{outd}/L{L}_f{a.fold}.json' if a.nchunks == 1 else f'{outd}/L{L}_f{a.fold}_c{a.chunk}of{a.nchunks}.json'
    if os.path.exists(outf):
        print('exists', outf); return
    t0 = time.time()
    D = f'{a.root}/data/{a.tag}'
    mm = torch.load(f'{D}/mmlu.pt'); alp = torch.load(f'{D}/alpaca.pt'); sst = torch.load(f'{D}/sst2.pt')
    hi = saes.hs_index(a.family, L)
    n = mm['h'].shape[0]; g = torch.Generator().manual_seed(0); perm = torch.randperm(n, generator=g)
    folds = [perm[i::5] for i in range(5)]
    te = folds[a.fold]; tr = torch.cat([folds[j] for j in range(5) if j != a.fold])
    sae = saes.load(a.family, L); sae = {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in sae.items()}
    W = sae['W_dec']
    Htr = mm['h'][tr, hi].float().to(DEV); ytr = mm['pred'][tr].long().to(DEV)
    Ytr = torch.nn.functional.one_hot(ytr, 4).float(); Yc = Ytr - Ytr.mean(0); Hc = Htr - Htr.mean(0)
    Sxy = Hc.T @ Yc / len(tr); beta = analyze.ridge_beta(Hc, Yc)
    # ---- selections on train split
    kmax = max(KS)
    Zt = saes.encode(sae, Htr); Za = saes.encode(sae, alp['h'][:, hi].float().to(DEV))
    cm = torch.stack([Zt[ytr == c].mean(0) for c in range(4)]); zm = Zt.mean(0)
    cnt = torch.bincount(ytr, minlength=4).float()[:, None]
    anova = ((cnt * (cm - zm) ** 2).sum(0) / 3) / (((Zt - cm[ytr]) ** 2).sum(0) / (len(tr) - 4) + 1e-8)
    anova[Zt.abs().sum(0) == 0] = -1
    sels = {}
    for meth in a.methods.split(','):
        if meth == 'md_task': sels[meth] = analyze.topk(zm - Za.mean(0), kmax)
        elif meth == 'anova': sels[meth] = analyze.topk(anova, kmax)
        elif meth == 'ols': sels[meth] = analyze.sel_ols(Hc, Yc, W, kmax)
        elif meth == 'somp_cov': sels[meth] = analyze.sel_somp(Sxy, W, kmax)
        elif meth == 'somp_both': sels[meth] = analyze.sel_somp(torch.cat([Sxy / Sxy.norm(), beta / beta.norm()], 1), W, kmax)
        elif meth == 'rand_atom': sels[meth] = torch.randperm(W.shape[0], generator=torch.Generator().manual_seed(a.fold))[:kmax].tolist()
    del Zt, Za
    erasers = {}   # name -> ('orth', Q) or ('leace', mu, A)
    for meth, S in sels.items():
        for k in KS:
            erasers[f'{meth}|{k}'] = ('orth', basis(W[S[:k]].T))
    G = torch.randn(W.shape[1], 50, generator=torch.Generator().manual_seed(100 + a.fold)).to(DEV)
    for k in [5, 50]:
        erasers[f'gauss|{k}'] = ('orth', basis(G[:, :k]))
    erasers['dense_cov|0'] = ('orth', basis(Sxy)); erasers['dense_beta|0'] = ('orth', basis(beta))
    erasers['dense_both|0'] = ('orth', basis(torch.cat([Sxy, beta], 1)))
    mu_l, A_l = leace_ops(Htr, Ytr); erasers['leace_cov|0'] = ('leace', mu_l, A_l)
    # sparse-atom LEACE: oblique erasure restricted to the span of covariance-matched atoms
    for k in [5, 20, 50, 100]:
        Qk = basis(W[sels['somp_cov'][:k]].T).double() if 'somp_cov' in sels else None
        if Qk is not None:
            # A_k = Q (Q^T W^-2... ) : LEACE computed in the k-dim coordinates Q^T h, lifted back
            muk, Ak = leace_ops(Htr @ Qk.float(), Ytr)
            erasers[f'leace_somp|{k}'] = ('leace', (Qk.float() @ muk), (Qk.float() @ Ak @ Qk.float().T))
    if a.nchunks > 1:
        keys = list(erasers.keys())[a.chunk::a.nchunks]
        erasers = {k: erasers[k] for k in keys}
    print(f'selections done {time.time()-t0:.0f}s, {len(erasers)} erasers (chunk {a.chunk}/{a.nchunks})', flush=True)

    # ---- model + prompts
    tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = 'left'
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    kw = dict(torch_dtype=torch.bfloat16, device_map='cuda')
    if 'gemma' in a.model.lower(): kw['attn_implementation'] = 'eager'
    model = AutoModelForCausalLM.from_pretrained(a.model, **kw).eval()
    letter_ids = [tok.encode(' ' + c, add_special_tokens=False)[-1] for c in 'ABCD']
    pn = [tok.encode(' ' + w, add_special_tokens=False)[0] for w in ['negative', 'positive']]
    ds = load_dataset('cais/mmlu', 'all', split='test')
    mm_prompts = [mmlu_prompt(ds[mm['idx'][i]]) for i in te.tolist()]; gold = mm['gold'][te]
    sds = load_dataset('nyu-mll/glue', 'sst2', split='validation')
    rs = random.Random(a.fold); sidx = rs.sample(range(len(sds)), 300)
    sst_prompts = [f"Review: {sds[i]['sentence'].strip()}\nSentiment (positive or negative):" for i in sidx]
    ads = load_dataset('tatsu-lab/alpaca', split='train'); aidx = rs.sample(range(len(ads)), 200)
    alp_prompts = [ads[i]['instruction'] + ('\n' + ads[i]['input'] if ads[i]['input'] else '') for i in aidx]
    layer_mod = model.model.layers[L]; st = {}

    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        e = st.get('e')
        if e is None: return out
        hf = h.float()
        if st['mode'] == 'abl':
            if e[0] == 'orth':
                Q = e[1]; hf = hf - (hf @ Q) @ Q.T
            else:
                mu, A = e[1], e[2]; hf = hf - (hf - mu) @ A.T
        else:  # swap at last token
            x = hf[:, -1]; s = st['src'].to(h.device).float()
            if e[0] == 'orth':
                Q = e[1]; hf[:, -1] = x - (x @ Q) @ Q.T + (s @ Q) @ Q.T
            else:
                A = e[2]; hf[:, -1] = x + (s - x) @ A.T
        h2 = hf.to(h.dtype)
        return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
    hd = layer_mod.register_forward_hook(hook)

    @torch.no_grad()
    def run(prompts, ids, e=None, mode='abl', src=None, full=False, want_h=False):
        st.update(e=e, mode=mode); P, LP, Hs = [], [], []
        for i in range(0, len(prompts), a.bs):
            if mode == 'swap': st['src'] = src[i:i + a.bs]
            enc = tok(prompts[i:i + a.bs], return_tensors='pt', padding=True, truncation=True, max_length=1024).to('cuda')
            out = model(**enc, output_hidden_states=want_h, logits_to_keep=1)
            lg = out.logits[:, -1].float()
            lp = lg.log_softmax(-1) if full else lg[:, ids].log_softmax(-1)
            LP.append(lp.cpu()); P.append(lg[:, ids].argmax(-1).cpu())
            if want_h: Hs.append(out.hidden_states[hi][:, -1].float())
        st['e'] = None
        return torch.cat(P), torch.cat(LP), (torch.cat(Hs) if Hs else None)

    p0, lp0, H0 = run(mm_prompts, letter_ids, want_h=True)
    s0, _, _ = run(sst_prompts, pn)
    _, la0, _ = run(alp_prompts, letter_ids, full=True)
    rng = random.Random(a.fold); src_idx = []
    for i in range(len(mm_prompts)):
        c = [j for j in range(len(mm_prompts)) if p0[j] != p0[i]]; src_idx.append(rng.choice(c) if c else i)
    src = H0[src_idx]; src_pred = p0[src_idx]
    res = dict(layer=L, fold=a.fold, n=len(mm_prompts), clean_acc=(p0 == gold).float().mean().item(), sae=sae['name'], methods={})
    for key, e in erasers.items():
        p, lp, _ = run(mm_prompts, letter_ids, e)
        ps, _, _ = run(mm_prompts, letter_ids, e, mode='swap', src=src)
        sp, _, _ = run(sst_prompts, pn, e)
        _, la, _ = run(alp_prompts, letter_ids, e, full=True)
        dim = e[1].shape[1] if e[0] == 'orth' else int(torch.linalg.matrix_rank(e[2].double()).item())
        r = dict(dim=dim, flip=(p != p0).float().mean().item(), acc=(p == gold).float().mean().item(),
                 kl=(lp0.exp() * (lp0 - lp)).sum(-1).mean().item(),
                 swap_follow=(ps == src_pred).float().mean().item(), swap_flip=(ps != p0).float().mean().item(),
                 sst_flip=(sp != s0).float().mean().item(),
                 alp_kl=(la0.exp() * (la0 - la)).sum(-1).clamp_min(0).mean().item())
        res['methods'][key] = r
        print(key, json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()}), flush=True)
    hd.remove()
    res['secs'] = time.time() - t0
    json.dump(res, open(outf, 'w'), indent=1)
    print('DONE', outf, f'{res["secs"]:.0f}s', flush=True)

if __name__ == '__main__':
    main()
