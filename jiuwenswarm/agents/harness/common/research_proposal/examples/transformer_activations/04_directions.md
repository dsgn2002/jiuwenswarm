# Potential research directions

| ID | Direction | Gap | Novelty | Feasibility | Impact | Score |
| --- | --- | --- | --- | --- | --- | --- |
| D1 | Natively sparse gated feed-forward units (recommended) | G2 | 4 | 4 | 4 | 4.0 |
| D2 | Ablating what makes GLU variants work | G1 | 3 | 5 | 3 | 3.67 |
| D3 | Layer-specific learnable activations in decoder language models | G3 | 3 | 3 | 3 | 3.0 |
| D4 | Co-designing per-layer activation and sparsity budget | G4 | 4 | 2 | 3 | 3.0 |

**Selection.** D1 ties the property that drives quality (gating) to the property that drives efficiency (exact zeros), which the evidence shows are currently traded off against each other; it scores 4 on novelty, feasibility and impact, and its first experiment also yields D2's ablation data as a by-product. D2 is the safest but mostly explanatory; D3 and D4 are costlier or less certain.

## D1. Natively sparse gated feed-forward units

**Hypothesis.** A GLU whose gate is a thresholded (exactly-zero) function, trained from scratch, keeps SwiGLU-level perplexity while making at least half of feed-forward activations exactly zero, removing the need for post-training sparsification.

**Rationale.** Gating drives current quality but loses ReLU's natural sparsity; efficiency work then imposes sparsity after training with thresholds or TopK. Building sparsity into the gate could give both at once.

**Builds on:** arxiv:2002.05202, arxiv:2210.06313, arxiv:2412.12178, arxiv:2603.12272

**First experiment.** Train small decoder language models with ReLU, GELU, SwiGLU and ReLU-gated GLU feed-forward layers at equal parameter count; compare validation perplexity and the fraction of exactly-zero feed-forward activations.

**Risks.**
- Hard zeros in the gate may hurt optimisation and erase the gain over ReLU.
- Sparsity may only pay off with custom kernels, so measured speed-ups could be small.

## D2. Ablating what makes GLU variants work

**Hypothesis.** Most of the GLU gain over ReLU and GELU comes from the multiplicative product of two projections, not from the gate nonlinearity's shape, once parameter count is matched.

**Rationale.** GLU variant gains are reported without explanation; a controlled ablation would separate multiplicative interaction, gate shape and parameter count.

**Builds on:** arxiv:2002.05202, arxiv:2506.23225

**First experiment.** Compare bilinear (no gate nonlinearity), sigmoid, Swish and GELU gates against ReLU and GELU feed-forward layers at matched parameters in small Transformer language models.

**Risks.**
- Results may depend on scale and not transfer to large models.
- A purely explanatory result may be seen as incremental.

## D3. Layer-specific learnable activations in decoder language models

**Hypothesis.** Learnable rational activations in decoder-only language models learn distinct shapes per layer and reduce perplexity relative to a fixed GELU or SwiGLU feed-forward layer.

**Rationale.** Learned activations vary by layer and task in BERT-style models and help small CNNs, but decoder language models are untested in the evidence.

**Builds on:** arxiv:2208.14111, arxiv:2409.08283, arxiv:1710.05941

**First experiment.** Replace GELU with rational activations in a small decoder language model and track per-layer learned shapes and validation perplexity.

**Risks.**
- Rational functions can be numerically unstable in long training runs.
- Extra parameters and kernels may cancel any quality gain.

## D4. Co-designing per-layer activation and sparsity budget

**Hypothesis.** Choosing each layer's activation by its weight-spectrum heavy-tail exponent improves the perplexity-sparsity trade-off over a single activation used everywhere.

**Rationale.** Per-projection sparsity budgets beat uniform ones and learned activations vary per layer, but the two have not been linked.

**Builds on:** arxiv:2603.12272, arxiv:2208.14111

**First experiment.** On a pretrained model, compute per-layer heavy-tail exponents and test whether layers with heavier tails tolerate higher activation sparsity under different activations.

**Risks.**
- The search space over per-layer choices is large and expensive.
- The link between spectra and activation choice may be weak.
