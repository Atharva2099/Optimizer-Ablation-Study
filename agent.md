# Agent Instructions: Optimizer Ablation Study

## Project Context
Research project comparing AdamW, Muon, and Aurora optimizers on small transformer 
pretraining. Learning-first approach: math and mechanism must be understood and 
documented before implementation proceeds.

## Absolute Constraints

### 1. Python Environment
- **ALWAYS use UV.** `uv venv .venv`, `uv pip install`, `uv run python`, `uvx`
- Never use bare `python`, `pip`, or `python -m venv`

### 2. No Local Downloads
- Do NOT download datasets, model weights, or large files to local disk
- Stream datasets via HuggingFace `datasets` library
- Build models from scratch (no downloading checkpoints)

### 3. Version Control
- Remote: https://github.com/Atharva2099/Optimizer-Ablation-Study.git
- Commit after each completed module
- Use descriptive conventional commits: `feat:`, `experiment:`, `analysis:`, `docs:`

### 4. Learning-First Workflow
- **NO one-shotting entire files.** Build incrementally
- Document math, algorithm, and intuition in code comments and markdown
- Pause for user review at major checkpoints:
  - After model architecture is defined
  - After each optimizer implementation (AdamW → Muon → Aurora)
  - Before running main experiments
- Explain "why" not just "what"

## Coding Standards

### File Structure
Place code in appropriate modules:
- `src/model.py` — GPT architecture
- `src/data.py` — data loading/tokenization
- `src/optimizers/{adamw,muon,aurora}.py` — optimizer implementations
- `src/train.py`, `src/eval.py` — loops
- `src/diagnostics.py` — row norm, dead neuron tracking
- `src/plotting.py` — visualization helpers
- `notebooks/0X_*.ipynb` — experiments

### Documentation Requirements
Every module must include:
1. Module-level docstring explaining purpose and references
2. Math explanations in comments or adjacent markdown
3. Assumptions and limitations noted
4. Links to papers/repos for deep details

Example:
```python
"""
Muon Optimizer
--------------
Reference: Keller Jordan (https://github.com/KellerJordan/Muon)

Math:
Newton-Schulz iteration approximates the polar factor U_p of matrix X:
  X_{k+1} = 0.5 * X_k * (3I - X_k^T X_k)
Converges to orthogonal polar factor where X = U_p * P.

Why: Orthogonal updates preserve layer-wise norm geometry.
"""
```

### Notebook Structure
Each Colab notebook must start with:
1. Setup cell (clone repo, install deps, verify GPU)
2. Config cell (hyperparameters, paths)
3. Data/model instantiation
4. Training loop
5. Evaluation/logging
6. Plotting

## Experiment Fairness Rules
- Identical: architecture, init seed, tokenizer, dataset, batch size, seq len, schedule, tokens
- Vary: LR, betas, weight decay, NS iterations, Aurora damping
- Report: best tuned result AND same-hyperparameter result

## Communication Rules
- Ask clarifying questions before large assumptions
- Present options when tradeoffs exist
- Summarize what was built before moving to next task
- Flag risks (e.g., "This will take 2 hours on T4")

## Colab Compatibility
- Notebooks must be runnable independently on Colab
- Include fallback setup if UV unavailable (standard pip)
- Check `torch.cuda.is_available()` and report GPU type
- Optimize batch sizes for T4 16GB VRAM
