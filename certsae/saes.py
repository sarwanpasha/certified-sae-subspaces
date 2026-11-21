"""Load public SAE decoders/encoders.
Returns dict(W_dec=[m,d] unit-norm rows, W_enc=[d,m], b_enc=[m], thr=[m]/scalar/None,
             pre_bias=[d] or None, in_scale=float, name=str).
encode(h) = act((h*in_scale - pre_bias) @ W_enc + b_enc)."""
import re, json, numpy as np, torch
from huggingface_hub import list_repo_files, hf_hub_download

_cache = {}

def _pack(Wd, We, be, thr=None, pre=None, scale=1.0, name='', topk=None):
    Wd = Wd.float()
    return dict(W_dec=Wd / Wd.norm(dim=1, keepdim=True).clamp_min(1e-8), W_enc=We.float(), b_enc=be.float(),
                thr=(None if thr is None else torch.as_tensor(thr).float()), pre_bias=(None if pre is None else pre.float()),
                in_scale=float(scale), name=name, topk=topk)

def gemmascope(layer, repo='google/gemma-scope-9b-pt-res', width='16k', target_l0=100):
    files = _cache.get(repo) or list_repo_files(repo); _cache[repo] = files
    cands = []
    for f in files:
        m = re.match(rf'layer_{layer}/width_{width}/average_l0_(\d+)/params.npz', f)
        if m:
            cands.append((abs(int(m.group(1)) - target_l0), f))
    if not cands:
        raise ValueError(f'no SAE for layer {layer} in {repo}')
    f = sorted(cands)[0][1]
    p = np.load(hf_hub_download(repo, f))
    return _pack(torch.tensor(p['W_dec']), torch.tensor(p['W_enc']), torch.tensor(p['b_enc']), torch.tensor(p['threshold']), name=f'{repo}/{f}')

def llamascope(layer, repo='OpenMOSS-Team/Llama3_1-8B-Base-LXR-8x'):
    from safetensors.torch import load_file
    sub = f'Llama3_1-8B-Base-L{layer}R-8x'
    hp = json.load(open(hf_hub_download(repo, f'{sub}/hyperparams.json')))
    sd = load_file(hf_hub_download(repo, f'{sub}/checkpoints/final.safetensors'))
    print('llamascope keys', {k: tuple(v.shape) for k, v in sd.items()}, {k: hp[k] for k in hp if 'norm' in k or 'act' in k or 'thresh' in k})
    We = sd['encoder.weight']; Wd = sd['decoder.weight']
    if We.shape[0] != Wd.shape[0] and We.shape[0] == Wd.shape[1]:
        pass
    We = We.T if We.shape[0] > We.shape[1] else We          # -> [d, m]
    Wd = Wd.T if Wd.shape[0] < Wd.shape[1] else Wd          # -> [m, d]
    d = We.shape[0]
    scale = 1.0
    an = hp.get('dataset_average_activation_norm')
    if isinstance(an, dict) and an.get('in'):
        scale = (d ** 0.5) / an['in']
    thr = hp.get('jump_relu_threshold')
    return _pack(Wd, We, sd['encoder.bias'], thr, pre=sd.get('decoder.bias') if hp.get('apply_decoder_bias_to_pre_encoder') else None,
                 scale=scale, name=f'{repo}/{sub}')

def andyrdt_qwen(layer, trainer=1, repo='andyrdt/saes-qwen2.5-7b-instruct'):
    sd = torch.load(hf_hub_download(repo, f'resid_post_layer_{layer}/trainer_{trainer}/ae.pt'), map_location='cpu')
    print('qwen sae keys', {k: tuple(v.shape) if torch.is_tensor(v) else v for k, v in sd.items()})
    We = sd['encoder.weight'].T                 # [d, m]
    Wd = sd['decoder.weight'].T                 # [m, d]
    thr = sd.get('threshold'); thr = None if thr is None or float(thr) < 0 else thr
    return _pack(Wd, We, sd['encoder.bias'], thr, pre=sd.get('b_dec'), name=f'{repo}/L{layer}/t{trainer}',
                 topk=int(sd['k']) if 'k' in sd else None)

def goodfire_llama(layer=19, repo='Goodfire/Llama-3.1-8B-Instruct-SAE-l19'):
    sd = torch.load(hf_hub_download(repo, 'Llama-3.1-8B-Instruct-SAE-l19.pth'), map_location='cpu')
    print('goodfire keys', {k: tuple(v.shape) for k, v in sd.items()})
    return _pack(sd['decoder_linear.weight'].T, sd['encoder_linear.weight'].T, sd['encoder_linear.bias'], None,
                 pre=None, name=repo)

def encode(sae, H):
    x = H * sae.get('in_scale', 1.0)
    if sae.get('pre_bias') is not None:
        x = x - sae['pre_bias'].to(H)
    pre = x @ sae['W_enc'].to(H) + sae['b_enc'].to(H)
    if sae.get('thr') is not None:
        return pre * (pre > sae['thr'].to(H))
    z = torch.relu(pre)
    if sae.get('topk'):
        v, i = torch.topk(z, sae['topk'], dim=-1)
        z = torch.zeros_like(z).scatter_(-1, i, v)
    return z

def load(family, layer):
    if family == 'gemma9b':
        return gemmascope(layer)
    if family == 'llamascope':
        return llamascope(layer)
    if family == 'goodfire':
        return goodfire_llama(layer)
    if family == 'qwen':
        return andyrdt_qwen(layer)
    raise ValueError(family)

# hidden_states index for an SAE reading the residual stream *after* block `layer`
def hs_index(family, layer):
    # GemmaScope layer L, LlamaScope LxR, andyrdt resid_post_layer_L, Goodfire l19 all read the output of block L
    return layer + 1
