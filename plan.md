# AdamW vs Muon vs Aurora: Measuring Optimizer Behavior in Small Transformer Pretraining

## 1. Project Overview
- **Goal:** Evaluate whether Muon/Aurora outperform AdamW on small transformers
- **Scope:** 10M–125M parameter GPT-style models on TinyStories/FineWeb-Edu
- **Philosophy:** Learning-first; document math and mechanism, not just results
- **Hardware target:** Google Colab (T4 GPU, 16GB VRAM)
- **VC:** https://github.com/Atharva2099/Optimizer-Ablation-Study.git

## 2. Research Questions & Hypotheses

### Main Question
Do Muon-style optimizers outperform AdamW on small transformer pretraining, 
and does Aurora reduce tall-matrix neuron/update anisotropy better than Muon?

### Subquestions
1. Loss-per-token and wall-clock speed vs AdamW?
2. Aurora improvement specifically on tall MLP matrices?
3. Row norm distribution changes in MLP projections?
4. Gains visible at small scale or only speedrun-tuned setups?

### Hypotheses
- **H1 (AdamW):** Most stable, easiest to tune, decent loss, slower convergence
- **H2 (Muon):** Faster convergence, possible instability, row update anisotropy in tall MLPs
- **H3 (Aurora):** Similar/better eval loss than Muon, fewer dead neurons, strongest gains at large MLP expansion ratios

## 3. Experimental Design

### Model Sizes
| Model  | Params       | Purpose        |
|--------|-------------|----------------|
| Tiny   | 10M–20M     | Debugging      |
| Small  | 60M–80M     | Main Colab run |
| Medium | 120M–150M   | Stretch goal   |

### Dataset Priority
1. TinyStories (fast, clean, cheap)
2. FineWeb-Edu sample (more realistic)

### Training Budget (First Run)
- Sequence length: 256
- Batch size: max that fits GPU
- Tokens: 50M–200M
- Eval interval: every 500 steps
- Seeds: 1 first, 3 later

## 4. Optimizer Configurations

### AdamW (Baseline)
- LRs: 3e-4, 6e-4, 1e-3
- Weight decay: 0.1
- Betas: (0.9, 0.95) or (0.9, 0.999)

### Muon
- Applied to: hidden 2D weight matrices only
- Fallback: AdamW for embeddings, output head, biases, norms
- Newton-Schulz iterations: TBD (default 5–10)
- Momentum: 0.95

### Aurora
- Applied to: tall MLP up/gate projections (primary), optionally all eligible
- Fallback: Muon for non-tall hidden matrices, AdamW for non-2D params
- Damping/iterations: TBD from Tilde repo reference

## 5. Fairness Rules

**Identical across optimizers:**
- Architecture, init seed, tokenizer, dataset split
- Batch size, sequence length, warmup schedule
- Total tokens, eval prompts, checkpoint intervals

**Optimizer-specific:**
- LR, momentum/betas, weight decay
- Newton-Schulz iterations
- Aurora damping/iterations

**Reporting:**
- Best tuned result AND same-hyperparameter result

## 6. Metrics & Diagnostics

### Core Metrics
| Metric            | Purpose              |
|-------------------|---------------------|
| train loss        | basic convergence   |
| validation loss   | generalization      |
| tokens/sec        | speed               |
| wall-clock time   | practical cost      |
| peak VRAM         | usability           |
| instability count | failed runs matter  |

### Optimizer-Specific Diagnostics (MLP projections)
1. Row norm distribution of weights
2. Row norm distribution of gradients
3. Row norm distribution of optimizer updates
4. Leverage-like score approximation
5. Dead neuron percentage

### Dead Neuron Definition
Mark dead if ALL low:
- activation mean near zero
- gradient row norm in bottom X%
- update row norm in bottom X%

Thresholds: bottom 5%, 10%, 25%

## 7. Required Plots
1. train loss vs tokens
2. validation loss vs tokens
3. validation loss vs wall-clock
4. tokens/sec by optimizer
5. peak VRAM by optimizer
6. MLP update row norm histogram
7. dead neuron percentage over time
8. row norm Gini coefficient over time
9. final benchmark table

**Best plot:** Muon vs Aurora row-update distribution for tall MLP matrices

## 8. Repository Structure
```
optimizer-comparison/
├── README.md
├── plan.md                 # this file
├── agent.md                # coding rules and constraints
├── requirements.txt        # python dependencies
└── notebooks/
    └── Ablation_Study.ipynb # SINGLE file containing everything (model, data, optimizers, training)
```

## 9. Experimental Phases

### Phase 1: Smoke Test
- Model: 10M–20M
- Tokens: 1M–5M
- Seed: 1
- Pass: all optimizers run, logs save, eval works

### Phase 2: Main Run
- Model: 60M–80M
- Tokens: 50M–200M
- Seeds: 1–3
- Pass: comparable curves, no hidden mismatches, diagnostics collected

### Phase 3: Scaling Probe
- Models: 20M, 80M, 125M
- Question: Does optimizer ranking change with scale?

### Phase 4: MLP Expansion Probe
- Expansion ratios: 2x, 4x, 8x
- Question: Is Aurora more interesting when tall matrices dominate?

## 10. Documentation Plan

### README
- project goal, research question
- how to run notebooks (local + Colab)
- hardware used, dataset, model configs
- optimizer configs, headline results
- limitations

### Experiment Report Structure
- Abstract (5–8 sentences)
- Motivation, Background (AdamW/Muon/Aurora)
- Experimental Setup, Methods, Results
- Discussion, Limitations, Conclusion
- Reproducibility (commits, configs, commands, seeds)

### Blog Title Options
1. Do Muon and Aurora Beat AdamW on Small Transformers?
2. Testing Whether Aurora Fixes Muon's Tall-Matrix Problem
3. Optimizer Shootout: AdamW, Muon, and Aurora on Tiny GPTs

## 11. Success Criteria
Even if AdamW wins, project is useful.

| Outcome | Conclusion |
|---------|-----------|
| AdamW wins | Muon/Aurora need scale or tuning |
| Muon wins but has anisotropy | Effective but mechanically weird |
| Aurora matches Muon, better dead-neuron metrics | Mechanism shows at small scale |
| Aurora beats both | Actual result, irritatingly exciting |

## 12. Tooling Constraints
- **Python:** UV only (`uv venv`, `uv pip install`, `uv run`)
- **No local downloads:** Datasets streamed via HuggingFace, models built from scratch
- **VC:** GitHub (https://github.com/Atharva2099/Optimizer-Ablation-Study.git)
- **Runtime:** Colab .ipynb notebooks for GPU training
- **Pace:** Document math/logic before moving to next file

## 13. First Exact Version
| Component | Choice |
|-----------|--------|
| Dataset   | TinyStories |
| Model     | 60M GPT |
| Seq len   | 256 |
| Optimizers| AdamW, Muon, Aurora |
| Tokens    | 50M |
| Seeds     | 1 |
| Metrics   | loss, tokens/sec, VRAM, update row norms |
| Output    | Colab + README + report |

## 14. References
- [Keller Jordan Muon](https://kellerjordan.github.io/posts/muon/)
- [Muon GitHub](https://github.com/KellerJordan/Muon)
- [Tilde Aurora](https://github.com/tilde-research)
