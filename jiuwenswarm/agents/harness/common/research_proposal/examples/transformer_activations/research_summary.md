# Literature review: Which properties of the feed-forward activation in Transformers (gating, smoothness, sparsity, learnability) account for quality differences between ReLU, GELU and GLU-style units, and which under-explored property offers the best next research target?

## Synthesis

The retrieved work splits into three lines that rarely meet. The gating line shows that replacing the Transformer feed-forward activation with GLU-style products improves pre-training perplexity and downstream scores over ReLU and GELU, yet its originating paper offers no explanation for the gain; later work extends gating to vision MLPs, attention values and adapters, and reduces its memory cost. The sparsity line shows that ReLU feed-forward activations in trained Transformers are naturally very sparse (about 3% non-zero in T5-Base) and that this sparsity can cut FLOPs and memory traffic, but for current LLMs the extra sparsity has to be imposed with thresholds or TopK rules after training. The learnability line replaces fixed functions with learned ones (rational functions in BERT-style models, series-based units in CNNs) or searches for them (Swish), reporting gains whose learned shapes vary by layer and task.

Taken together, the sources show that the property most responsible for current quality (multiplicative gating with smooth gates) is not the property that gives efficiency (exact zeros from ReLU), and no retrieved source designs an activation for both. Evidence on learnable activations comes mostly from small models or image classification, so whether layer-specific shapes matter in large decoder language models is not stated in these sources.

## Themes

### Multiplicative gating

GLU-style products improve Transformer feed-forward quality and spread to attention, vision MLPs and adapters; their memory cost is being reduced.

Sources: arxiv:2002.05202, arxiv:2506.23225, arxiv:2405.15953, arxiv:2507.00022, arxiv:2508.07836

### Activation sparsity

ReLU Transformers are naturally sparse, and sparsity is exploited for efficiency, but modern LLMs need sparsity imposed after training.

Sources: arxiv:2210.06313, arxiv:2412.12178, arxiv:2603.12272

### Learned and searched activations

Learning or searching the activation shape gives gains, with shapes varying by layer and task, mostly shown on small models.

Sources: arxiv:2208.14111, arxiv:2409.08283, arxiv:1710.05941, arxiv:1911.03925

## Gaps

- **G1.** The quality gain of GLU variants in Transformer feed-forward layers is reported but unexplained, so it is unknown whether it comes from multiplicative interaction, the gate's shape, or extra parameters. (evidence: arxiv:2002.05202, arxiv:2506.23225)

- **G2.** Natural activation sparsity is documented for ReLU feed-forward layers, while current gated and smooth activations need sparsity imposed after training; no retrieved work designs a gated activation that is natively sparse without losing quality. (evidence: arxiv:2210.06313, arxiv:2412.12178, arxiv:2603.12272, arxiv:2002.05202)

- **G3.** Learnable activations show layer- and task-specific shapes and gains, but evidence comes from BERT-scale encoders or image classifiers; their value in large decoder language models is not established. (evidence: arxiv:2208.14111, arxiv:2409.08283, arxiv:1911.03925)

- **G4.** Sparsity budgets matter per projection (uniform allocation hurts), suggesting activation design could be layer-specific, but no source links per-layer activation choice to per-layer sparsity. (evidence: arxiv:2603.12272, arxiv:2208.14111)

## Sources

### [arxiv:2210.06313] The Lazy Neuron Phenomenon: On Emergence of Activation Sparsity in Transformers

Zonglin Li, Chong You, Srinadh Bhojanapalli et al. (2022). *The Lazy Neuron Phenomenon: On Emergence of Activation Sparsity in Transformers*. arXiv. https://doi.org/10.48550/arXiv.2210.06313 <https://arxiv.org/abs/2210.06313>

Local copy: `sources/arxiv_2210.06313.pdf` (downloaded_pdf)

**Summary.** Measures activation sparsity after ReLU in Transformer MLPs and finds it emerges across tasks, sizes and even random-label data.

**Key findings.**
- On average about 3.0% of entries are non-zero for T5-Base and 6.3% for ViT-B16.
- Larger Transformers with more layers and wider MLPs are sparser.
- Enforcing more sparsity with Top-k thresholding improves robustness to noisy labels and input corruption and improves calibration.

**Limitations.**
- Studied on ReLU activation maps; behaviour of GELU or GLU feed-forward layers is not stated.
- Why sparsity emerges is not fully explained by the tested hypotheses.

**Relevance:** 5/5

### [arxiv:2208.14111] Transformers with Learnable Activation Functions

Haishuo Fang, Ji-Ung Lee, Nafise Sadat Moosavi et al. (2022). *Transformers with Learnable Activation Functions*. arXiv. https://doi.org/10.48550/arXiv.2208.14111 <https://arxiv.org/abs/2208.14111>

Local copy: `sources/arxiv_2208.14111.pdf` (downloaded_pdf)

**Summary.** Replaces GELU with a learnable rational activation function (RAF) in a BERT-style Transformer (RAFT).

**Key findings.**
- RAFT reaches lower validation perplexity than a vanilla BERT with GELU.
- RAFT beats the fixed-activation model on GLUE by 5.71 points with 100 training examples and by 2.05 points on SQuAD with all data.
- Learned activation shapes differ across layers and tasks.

**Limitations.**
- Evaluated on BERT-style encoders; large decoder LLMs are not tested in the evidence.
- Compute or memory cost of RAFs is not stated.

**Relevance:** 5/5

### [arxiv:2002.05202] GLU Variants Improve Transformer

Noam Shazeer (2020). *GLU Variants Improve Transformer*. arXiv. https://doi.org/10.48550/arXiv.2002.05202 <https://arxiv.org/abs/2002.05202>

Local copy: `sources/arxiv_2002.05202.pdf` (downloaded_pdf)

**Summary.** Tests GLU variants (different gate nonlinearities, including linear) in Transformer feed-forward sublayers.

**Key findings.**
- Some GLU variants improve pre-training perplexity and downstream language-understanding results over ReLU and GELU.
- The variants are simple to implement with no apparent computational drawbacks.

**Limitations.**
- The author offers no explanation of why the variants work.

**Relevance:** 5/5

### [arxiv:2506.23225] Masked Gated Linear Unit

Yukito Tajima, Nakamasa Inoue, Yusuke Sekikawa et al. (2025). *Masked Gated Linear Unit*. arXiv. https://doi.org/10.48550/arXiv.2506.23225 <https://arxiv.org/abs/2506.23225>

Local copy: `sources/arxiv_2506.23225.pdf` (downloaded_pdf)

**Summary.** Proposes Masked GLUs that share one weight matrix between gate and value via learned binary masks, with a fused kernel.

**Key findings.**
- Standard GLUs need twice the memory reads of ungated feed-forward layers.
- FlashMGLU is 47% more memory-efficient and 34% faster than standard GLUs on an RTX5090 GPU.
- SwiMGLU matches or surpasses SwiGLU downstream accuracy in LLM experiments.

**Limitations.**
- Speed figures are for one GPU type.
- Targets memory cost, not why gating improves quality.

**Relevance:** 4/5

### [arxiv:2412.12178] Activation Sparsity Opportunities for Compressing General Large Language Models

Nobel Dhar, Bobin Deng, Md Romyull Islam et al. (2024). *Activation Sparsity Opportunities for Compressing General Large Language Models*. 2024 IEEE International Performance, Computing, and Communications Conference (IPCCC), Orlando, FL, USA, 2024. https://doi.org/10.1109/IPCCC59868.2024.10850382 <https://arxiv.org/abs/2412.12178>

Local copy: `sources/arxiv_2412.12178.pdf` (downloaded_pdf)

**Summary.** Studies the trade-off between enforced activation sparsity and perplexity in LLM feed-forward networks for edge deployment.

**Key findings.**
- Feed-forward networks hold around two thirds of LLM parameters.
- About 50% memory and compute reduction in FFN components with negligible accuracy loss.
- This extra 50% sparsity does not occur naturally in current LLMs and requires zero-enforcing thresholds.
- Activation patterns are predictable enough to prefetch weights.

**Limitations.**
- Sparsity is imposed after training rather than designed into the activation.
- Model list and exact accuracy deltas are not stated in the excerpt.

**Relevance:** 4/5

### [arxiv:2409.08283] Activation function optimization method: Learnable series linear units (LSLUs)

Chuan Feng, Xi Lin, Shiping Zhu et al. (2024). *Activation function optimization method: Learnable series linear units (LSLUs)*. arXiv. https://doi.org/10.48550/arXiv.2409.08283 <https://arxiv.org/abs/2409.08283>

Local copy: `sources/arxiv_2409.08283.pdf` (downloaded_pdf)

**Summary.** Proposes LSLU, a learnable series-based activation with parameters controlling its non-linearity per layer.

**Key findings.**
- LSLU improves CIFAR100 accuracy by 3.17% for VanillaNet.
- Shallow wide networks benefit from higher per-layer non-linearity; deep narrow ResNet-18 loses accuracy when non-linearity is too high.

**Limitations.**
- Evaluated on image classification, not Transformers or language models.

**Relevance:** 2/5

### [arxiv:1911.03925] Symmetrical Gaussian Error Linear Units (SGELUs)

Chao Yu, Zhiguo Su (2019). *Symmetrical Gaussian Error Linear Units (SGELUs)*. arXiv. https://doi.org/10.48550/arXiv.1911.03925 <https://arxiv.org/abs/1911.03925>

Local copy: `sources/arxiv_1911.03925.pdf` (downloaded_pdf)

**Summary.** Proposes SGELU, a symmetric variant of GELU.

**Key findings.**
- SGELU is compared against GELU and LiSHT on MNIST classification and an MNIST auto-encoder.

**Limitations.**
- Evaluated only on MNIST; Transformer results are not stated.

**Relevance:** 2/5

### [arxiv:1710.05941] Searching for Activation Functions

Prajit Ramachandran, Barret Zoph, Quoc V. Le (2017). *Searching for Activation Functions*. arXiv. https://doi.org/10.48550/arXiv.1710.05941 <https://arxiv.org/abs/1710.05941>

Local copy: `sources/arxiv_1710.05941.pdf` (downloaded_pdf)

**Summary.** Uses exhaustive and reinforcement-learning search to discover activation functions; the best is Swish, x times sigmoid(beta x).

**Key findings.**
- Hand-designed ReLU alternatives had not replaced ReLU because of inconsistent gains.
- Swish improves ImageNet top-1 by 0.9% for Mobile NASNet-A and 0.6% for Inception-ResNet-v2.

**Limitations.**
- Evaluated on image models; Transformer feed-forward results are not stated in the evidence.

**Relevance:** 4/5

### [arxiv:2405.15953] Activator: GLU Activation Function as the Core Component of a Vision Transformer

Abdullah Nazhat Abdullah, Tarkan Aydin (2024). *Activator: GLU Activation Function as the Core Component of a Vision Transformer*. arXiv. https://doi.org/10.48550/arXiv.2405.15953 <https://arxiv.org/abs/2405.15953>

Local copy: `sources/arxiv_2405.15953.pdf` (downloaded_pdf)

**Summary.** Replaces both the MLP and attention of a vision Transformer with GLU-based MLPs to cut compute.

**Key findings.**
- GLU-based design gives competitive performance against selected baselines at reduced computational complexity.

**Limitations.**
- Vision only; specific accuracy numbers are not stated in the evidence.

**Relevance:** 3/5

### [arxiv:2507.00022] GLU Attention Improve Transformer

Zehao Wang (2025). *GLU Attention Improve Transformer*. arXiv. https://doi.org/10.48550/arXiv.2507.00022 <https://arxiv.org/abs/2507.00022>

Local copy: `sources/arxiv_2507.00022.pdf` (downloaded_pdf)

**Summary.** Adds GLU nonlinearity to the value projections of multi-head attention.

**Key findings.**
- GLU Attention improves performance and convergence speed across text and vision with zero extra parameters.
- GLU FFNs are reported as adopted in Llama 3.

**Limitations.**
- Single-author study; model scales are not stated in the evidence.

**Relevance:** 3/5

### [arxiv:2508.07836] G-IFT: A Gated Linear Unit adapter with Iterative Fine-Tuning for Low-Resource Children's Speaker Verification

Vishwas M. Shetty, Jiusi Zheng, Abeer Alwan (2025). *G-IFT: A Gated Linear Unit adapter with Iterative Fine-Tuning for Low-Resource Children's Speaker Verification*. arXiv. https://doi.org/10.48550/arXiv.2508.07836 <https://arxiv.org/abs/2508.07836>

Local copy: `sources/arxiv_2508.07836.pdf` (downloaded_pdf)

**Summary.** Uses a GLU adapter with iterative fine-tuning to adapt adult speaker-verification models to children's speech.

**Key findings.**
- Consistent equal-error-rate reductions on OGI and MyST across three architectures.

**Limitations.**
- Speaker verification, not Transformer feed-forward activations; only tangentially relevant.

**Relevance:** 1/5

### [arxiv:2603.12272] ActTail: Global Activation Sparsity in Large Language Models

Wenwen Hou, Xinyuan Song, Shiwei Liu (2026). *ActTail: Global Activation Sparsity in Large Language Models*. arXiv. https://doi.org/10.48550/arXiv.2603.12272 <https://arxiv.org/abs/2603.12272>

Local copy: `sources/arxiv_2603.12272.pdf` (downloaded_pdf)

**Summary.** ActTail allocates TopK activation sparsity per projection using heavy-tail exponents of weight spectra.

**Key findings.**
- Uniform sparsity across projections amplifies degradation.
- At 80% sparsity perplexity is reduced by 21.8% on LLaMA-2-7B, 40.1% on LLaMA-2-13B and 9.4% on Mistral-7B relative to uniform allocation.

**Limitations.**
- Post-training sparsification; the activation function itself is unchanged.

**Relevance:** 4/5

---
Queries run: 7. Sources kept: 12. Search results rejected as off-topic or below relevance threshold: 61.
