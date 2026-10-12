# Potential research directions

| ID | Direction | Gap | Novelty | Feasibility | Impact | Score |
| --- | --- | --- | --- | --- | --- | --- |
| D1 | Verifying tool outputs against visual evidence (recommended) | G1 | 4 | 4 | 4 | 4.0 |
| D2 | End-to-end choose-call-use benchmark for multimodal agents | G2 | 3 | 4 | 3 | 3.33 |
| D3 | Temporal context for GUI tool use | G3 | 3 | 3 | 3 | 3.0 |

**Selection.** D1 targets the failure that the evidence says better perception does not fix, and it can be tested on an existing multimodal tool benchmark with a small intervention; D2 is mostly measurement and D3 is narrower to GUIs.

## D1. Verifying tool outputs against visual evidence

**Hypothesis.** Adding an explicit check of each tool result against the image (consistent, contradicted, or undecidable) before it is used reduces false acceptance in multimodal agents more than stronger perception models do.

**Rationale.** Visual ignorance and false acceptance survive stronger embeddings, and current agent training rewards only final answers.

**Builds on:** arxiv:2609.31718, arxiv:2505.14246

**First experiment.** On MAT-style tasks, compare an agent with and without a tool-result verification step, measuring false acceptance and final accuracy.

**Risks.**
- The verifier may share the agent's perception errors.
- Undecidable cases may dominate.

## D2. End-to-end choose-call-use benchmark for multimodal agents

**Hypothesis.** Errors compound across tool selection, argument formation and output use, so end-to-end success is well below the product of step-level scores measured in isolation.

**Rationale.** Selection and argument correctness are benchmarked separately with simplified setups.

**Builds on:** arxiv:2503.01763, arxiv:2509.18420

**First experiment.** Chain ToolRet-style retrieval with IFEval-FC-style argument checks on the same tasks and compare end-to-end with per-step success.

**Risks.**
- Benchmark construction cost.

## D3. Temporal context for GUI tool use

**Hypothesis.** Giving GUI agents an explicit operation history improves tool and action choice on dynamic content more than higher-resolution frames do.

**Rationale.** Models fail on dynamic GUI content without keyframes or operation history.

**Builds on:** arxiv:2406.10819, arxiv:2607.09822

**First experiment.** Ablate operation history versus frame resolution on GUI-World dynamic tasks.

**Risks.**
- History may overflow context on long tasks.
