# Faithful Chain-of-Thought Reasoning

Source: arxiv:2301.13379. Authors: Qing Lyu, Shreya Havaldar, Adam Stein, Li Zhang, Delip Rao, Eric Wong. Year: 2023. DOI: 10.48550/arXiv.2301.13379.

Topics: When LLMs interleave long reasoning with tool calls, how far do their reasoning steps actually check and depend on intermediate tool results, and can process-level rewards make reasoning verify tool outputs rather than only reach correct final answers?

Summary: Faithful CoT translates a natural-language query into a symbolic reasoning chain and executes it with a deterministic solver so the chain faithfully explains the answer.

Abstract: While Chain-of-Thought (CoT) prompting boosts Language Models' (LM) performance on a gamut of complex reasoning tasks, the generated reasoning chain does not necessarily reflect how the model arrives at the answer (aka. faithfulness). We propose Faithful CoT, a reasoning framework involving two stages: Translation (Natural Language query $\rightarrow$ symbolic reasoning chain) and Problem Solving (reasoning chain $\rightarrow$ answer), using an LM and a deterministic solver respectively. This guarantees that the reasoning chain provides a faithful explanation of the final answer. Aside from interpretability, Faithful CoT also improves empirical performance: it outperforms standard CoT on 9 of 10 benchmarks from 4 diverse domains, with a relative accuracy gain of 6.3% on Math Word Problems (MWP), 3.4% on Planning, 5.5% on Multi-hop Question Answering (QA), and 21.4% on Relational Inference. Furthermore, with GPT-4 and Codex, it sets the new state-of-the-art few-shot performance on 7 datasets (with 95.0+ accuracy on 6 of them), showing a strong synergy between faithfulness and accuracy.

Key findings:
- Standard CoT chains do not necessarily reflect how the model reaches its answer.
- Faithful CoT beats standard CoT on 9 of 10 benchmarks, e.g. +6.3% relative on math word problems and +21.4% on relational inference.
- With GPT-4 and Codex it sets few-shot state of the art on 7 datasets.

Limitations:
- The translation step remains opaque, so the pipeline is not fully interpretable.
