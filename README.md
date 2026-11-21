# certsae: certified sufficient and necessary subspaces from SAE dictionaries

Tools for selecting subspaces spanned by sparse-autoencoder (SAE) dictionary atoms that are
**sufficient** (a linear probe on the subspace recovers a target as well as the full
representation) and/or **necessary** (removing the subspace makes the target linearly
undecodable), and for testing them causally inside a language model.

Key facts implemented and tested here:

* sufficiency requires the subspace to contain the least-squares readout `Sigma^{-1} Sigma_hy`;
* necessity (orthogonal erasure) requires it to contain the cross-covariance `Sigma_hy`;
* greedy dictionary-constrained selection (orthogonal least squares / simultaneous OMP) of atoms
  that approximate these targets, compared against univariate top-k rankings
  (mean difference, t-statistic, ANOVA), random atoms and random subspaces.

## Contents

| file | purpose |
|---|---|
| `certsae/extract.py` | last-token residual-stream activations (all layers) + model predictions for MMLU, SST-2, Alpaca |
| `certsae/saes.py` | loaders for public SAEs (GemmaScope, Llama Scope, Goodfire Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct BatchTopK) |
| `certsae/analyze.py` | selection methods + cross-validated sufficiency/necessity probes (rho_in / rho_out) |
| `certsae/causal2.py` | causal tests: subspace ablation (orthogonal and LEACE-style), interchange (swap), specificity controls |
| `certsae/synth_theory.py` | synthetic planted-support study with controllable dictionary coherence |
| `certsae/check_theorems.py` | numerical checks of the sufficiency/necessity bounds and the greedy guarantee |
| `certsae/summarize.py`, `certsae/agg_causal2.py` | aggregate results across folds |

## Usage

```bash
pip install -r requirements.txt
export CERTSAE_ROOT=./outputs          # where data/ and results/ are written
cd certsae
python extract.py --model google/gemma-2-9b-it --tag gemma9b
python analyze.py --tag gemma9b --family gemma9b --layers 21 25
python causal2.py --model google/gemma-2-9b-it --tag gemma9b --family gemma9b --layer 25 --fold 0
python synth_theory.py --jobs 8
python check_theorems.py
```

Gated models/SAEs require a Hugging Face token (`HF_TOKEN`). All feature selection is done on
training folds only; held-out folds are used for every reported number.

## License

MIT
