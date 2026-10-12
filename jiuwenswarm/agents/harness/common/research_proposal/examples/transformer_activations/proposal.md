# Natively Sparse Gated Feed-Forward Units for Transformer Language Models

## Summary

Gated feed-forward layers such as SwiGLU give Transformers better perplexity than ReLU or GELU [1], but they discard the natural activation sparsity that ReLU feed-forward layers show [2], so efficiency work must impose sparsity after training [3] [4]. We propose to build exact zeros into the gate itself and test whether a thresholded gate keeps SwiGLU-level quality while making at least half of feed-forward activations exactly zero, from training onwards.

## Background

The Transformer feed-forward layer originally applied ReLU between two projections. Gated linear unit variants replace this with the product of two projections, one passed through a gate nonlinearity, and some variants improve pre-training perplexity and downstream scores over ReLU and GELU [1]. Gating is now standard in large language models, and work on its cost shows that the separate gate and value matrices double memory reads, which masked GLUs reduce while matching SwiGLU accuracy [5]. Why gating helps remains unexplained in the original work [1].

A second line studies activation sparsity. In trained Transformers with ReLU feed-forward layers only about 3% of activations are non-zero for T5-Base, larger models are sparser, and enforcing even more sparsity with Top-k improves robustness and calibration [2]. For current large language models this sparsity is not natural: around 50% FFN memory and compute savings are possible with negligible accuracy loss, but only after injecting zero-enforcing thresholds [3], and allocating TopK sparsity per projection by weight-spectrum statistics beats uniform allocation [4].

The two lines meet at a trade-off: the property associated with quality (smooth multiplicative gating) is not the property that provides efficiency (exact zeros). Activation-function search [6] and learnable activations [7] have explored the shape of the nonlinearity, but not a gate designed to be exactly zero for most inputs.

## Research questions

1. Can a gated feed-forward unit with an exactly-zero gate region match SwiGLU perplexity at equal parameter count?
2. How much natural activation sparsity does such a unit reach during training, and does it grow with model width as reported for ReLU [2]?
3. Does native sparsity make post-training sparsification such as threshold injection [3] or TopK allocation [4] unnecessary or more effective?

## Hypotheses

1. H1: A ReLU-gated or shifted-threshold GLU reaches validation perplexity within 1% of SwiGLU at equal parameters.
2. H2: At least 50% of its feed-forward activations are exactly zero at the end of training, without any post-hoc threshold.
3. H3: At a fixed sparsity target, the natively sparse unit loses less perplexity than SwiGLU sparsified after training.

## Method

Train decoder-only Transformer language models at two small scales with five feed-forward variants at matched parameter count: ReLU, GELU, SwiGLU [1], a ReLU-gated GLU, and a GLU whose gate is a shifted ReLU with a learnable per-layer threshold. Log, per layer and over training, the fraction of exactly-zero feed-forward activations, following the measurement of [2]. After training, apply threshold-based sparsification [3] and per-projection TopK allocation [4] to SwiGLU and compare with the natively sparse units at equal sparsity. Use the same data, tokenizer, optimiser and token budget for every variant and three seeds per configuration.

## Evaluation

Primary metrics: validation perplexity on a held-out split of the pre-training corpus and the fraction of exactly-zero feed-forward activations per layer. Secondary: downstream zero-shot accuracy on standard language-understanding benchmarks, and measured feed-forward latency with a sparse kernel. Baselines: ReLU, GELU and SwiGLU, plus SwiGLU with post-training threshold and TopK sparsification. H1 is refuted if the best natively sparse unit is more than 1% worse in perplexity than SwiGLU across seeds; H2 if fewer than 50% of activations are exactly zero; H3 if post-hoc sparsified SwiGLU matches or beats it at equal sparsity.

## Risks and mitigations

- Hard zeros may slow optimisation and remove the gating gain; mitigate with a learnable threshold initialised near zero and a warm-up from a smooth gate.
- Small-scale results may not transfer; mitigate by running two scales and reporting the trend.
- Sparsity may not give wall-clock savings without kernels; mitigate by reporting FLOP and memory-traffic reductions separately from latency.

## Milestones

1. Reproduce ReLU, GELU and SwiGLU baselines at the smaller scale and confirm the reported SwiGLU advantage [1].
2. Implement the two natively sparse gated units and the per-layer sparsity logging.
3. Run the five-variant comparison at both scales with three seeds.
4. Apply post-training sparsification to SwiGLU and compare at matched sparsity.
5. Write up results, including negative findings for any hypothesis that is refuted.

## References

1. Noam Shazeer (2020). *GLU Variants Improve Transformer*. arXiv. https://doi.org/10.48550/arXiv.2002.05202 <https://arxiv.org/abs/2002.05202>
2. Zonglin Li, Chong You, Srinadh Bhojanapalli et al. (2022). *The Lazy Neuron Phenomenon: On Emergence of Activation Sparsity in Transformers*. arXiv. https://doi.org/10.48550/arXiv.2210.06313 <https://arxiv.org/abs/2210.06313>
3. Nobel Dhar, Bobin Deng, Md Romyull Islam et al. (2024). *Activation Sparsity Opportunities for Compressing General Large Language Models*. 2024 IEEE International Performance, Computing, and Communications Conference (IPCCC), Orlando, FL, USA, 2024. https://doi.org/10.1109/IPCCC59868.2024.10850382 <https://arxiv.org/abs/2412.12178>
4. Wenwen Hou, Xinyuan Song, Shiwei Liu (2026). *ActTail: Global Activation Sparsity in Large Language Models*. arXiv. https://doi.org/10.48550/arXiv.2603.12272 <https://arxiv.org/abs/2603.12272>
5. Yukito Tajima, Nakamasa Inoue, Yusuke Sekikawa et al. (2025). *Masked Gated Linear Unit*. arXiv. https://doi.org/10.48550/arXiv.2506.23225 <https://arxiv.org/abs/2506.23225>
6. Prajit Ramachandran, Barret Zoph, Quoc V. Le (2017). *Searching for Activation Functions*. arXiv. https://doi.org/10.48550/arXiv.1710.05941 <https://arxiv.org/abs/1710.05941>
7. Haishuo Fang, Ji-Ung Lee, Nafise Sadat Moosavi et al. (2022). *Transformers with Learnable Activation Functions*. arXiv. https://doi.org/10.48550/arXiv.2208.14111 <https://arxiv.org/abs/2208.14111>
