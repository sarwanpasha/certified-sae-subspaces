"""Extract last-token residual-stream activations at ALL layers for several
targets, plus the model's own predictions.

Datasets / targets (all stored per-example):
  mmlu   : random 2000 MMLU test items (seed 0). targets: pred letter (model
           argmax over A-D), gold letter, supercategory (4 classes)
  sst2   : GLUE SST-2 validation (872). target: label; also model pred.
  alpaca : 1000 Alpaca instructions (baseline distribution for the
           task-vs-baseline mean-difference selection used in prior work)
Output: data/<tag>/<ds>.pt with dict(h=[N, L+1, d] fp16, meta...)
"""
import os, argparse, json, random
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

SUPER = {
 'stem': ['abstract_algebra','anatomy','astronomy','college_biology','college_chemistry','college_computer_science','college_mathematics','college_physics','computer_security','conceptual_physics','electrical_engineering','elementary_mathematics','high_school_biology','high_school_chemistry','high_school_computer_science','high_school_mathematics','high_school_physics','high_school_statistics','machine_learning'],
 'humanities': ['formal_logic','high_school_european_history','high_school_us_history','high_school_world_history','international_law','jurisprudence','logical_fallacies','moral_disputes','moral_scenarios','philosophy','prehistory','professional_law','world_religions'],
 'social': ['econometrics','high_school_geography','high_school_government_and_politics','high_school_macroeconomics','high_school_microeconomics','high_school_psychology','human_sexuality','professional_psychology','public_relations','security_studies','sociology','us_foreign_policy'],
}
def supercat(s):
    for i, k in enumerate(['stem', 'humanities', 'social']):
        if s in SUPER[k]:
            return i
    return 3  # other

def mmlu_prompt(ex):
    L = 'ABCD'
    q = ex['question'].strip() + '\n' + '\n'.join(f'{L[i]}. {c}' for i, c in enumerate(ex['choices']))
    return q + '\nAnswer:'

@torch.no_grad()
def run(model, tok, prompts, layers_all, bs, letter_ids=None):
    H, P = [], []
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i:i+bs], return_tensors='pt', padding=True, truncation=True, max_length=1024).to(model.device)
        out = model(**enc, output_hidden_states=True)
        hs = torch.stack([h[:, -1, :] for h in out.hidden_states], 1)  # [b, L+1, d]
        H.append(hs.float().cpu().half())
        if letter_ids is not None:
            lg = out.logits[:, -1, :][:, letter_ids].float()
            P.append(lg.argmax(-1).cpu())
        if i % (bs*20) == 0:
            print(f'  {i}/{len(prompts)}', flush=True)
    H = torch.cat(H)
    return H, (torch.cat(P) if P else None)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--root', default=os.environ.get('CERTSAE_ROOT', './outputs'))
    ap.add_argument('--n_mmlu', type=int, default=2000)
    ap.add_argument('--n_alpaca', type=int, default=1000)
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--datasets', default='mmlu,sst2,alpaca')
    a = ap.parse_args()
    out = f'{a.root}/data/{a.tag}'; os.makedirs(out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = 'left'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = dict(torch_dtype=torch.bfloat16, device_map='cuda')
    if 'gemma' in a.model.lower():
        kw['attn_implementation'] = 'eager'
    model = AutoModelForCausalLM.from_pretrained(a.model, **kw).eval()
    letter_ids = [tok.encode(' ' + c, add_special_tokens=False)[-1] for c in 'ABCD']
    print('letter ids', letter_ids, [tok.decode([t]) for t in letter_ids])
    dss = a.datasets.split(',')

    if 'mmlu' in dss:
        ds = load_dataset('cais/mmlu', 'all', split='test')
        idx = list(range(len(ds))); random.Random(0).shuffle(idx); idx = idx[:a.n_mmlu]
        exs = [ds[i] for i in idx]
        prompts = [mmlu_prompt(e) for e in exs]
        H, P = run(model, tok, prompts, None, a.bs, letter_ids)
        gold = torch.tensor([e['answer'] for e in exs]); sc = torch.tensor([supercat(e['subject']) for e in exs])
        torch.save(dict(h=H, pred=P, gold=gold, supercat=sc, idx=idx,
                        subject=[e['subject'] for e in exs]), f'{out}/mmlu.pt')
        print('mmlu pred dist', torch.bincount(P, minlength=4).tolist(), 'acc', (P == gold).float().mean().item(), flush=True)

    if 'sst2' in dss:
        ds = load_dataset('nyu-mll/glue', 'sst2', split='validation')
        prompts = [f"Review: {e['sentence'].strip()}\nSentiment (positive or negative):" for e in ds]
        pn = [tok.encode(' ' + w, add_special_tokens=False)[0] for w in ['negative', 'positive']]
        H, P = run(model, tok, prompts, None, a.bs, pn)
        lab = torch.tensor(ds['label'])
        torch.save(dict(h=H, pred=P, label=lab), f'{out}/sst2.pt')
        print('sst2 acc', (P == lab).float().mean().item(), flush=True)

    if 'alpaca' in dss:
        ds = load_dataset('tatsu-lab/alpaca', split='train')
        idx = list(range(len(ds))); random.Random(0).shuffle(idx); idx = idx[:a.n_alpaca]
        prompts = [(ds[i]['instruction'] + ('\n' + ds[i]['input'] if ds[i]['input'] else '')) for i in idx]
        H, _ = run(model, tok, prompts, None, a.bs)
        torch.save(dict(h=H, idx=idx), f'{out}/alpaca.pt')
    print('DONE', flush=True)

if __name__ == '__main__':
    main()
