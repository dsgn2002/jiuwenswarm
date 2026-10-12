# Literature review: When LLMs interleave long reasoning with tool calls, how far do their reasoning steps actually check and depend on intermediate tool results, and can process-level rewards make reasoning verify tool outputs rather than only reach correct final answers?

## Synthesis

The retrieved work treats reasoning and acting as increasingly coupled. ReAct interleaves reasoning traces with actions so that retrieved information updates the plan, and newer reinforcement-learning methods feed tool or execution results back into each reasoning step (multi-turn tool-integrated RL for text-to-SQL; visual agentic RL with verifiable rewards, carried over from yesterday's session). RL with verifiable rewards also drives pure reasoning gains, with work on which training problems to synthesise from model weaknesses.

A second line asks whether reasoning traces can be trusted. Standard chain of thought does not necessarily reflect how the answer is reached; Faithful CoT guarantees faithfulness by translating the query into a symbolic chain run by a deterministic solver, and representation-level analysis localises reasoning errors. Across both lines, rewards and checks attach to final answers or executable outcomes; no source tests whether intermediate reasoning steps actually depend on the tool results they cite, which connects to yesterday's gap G1 on unverified tool outputs.

## Themes

### Interleaving reasoning with tool feedback

Reasoning traces and tool or execution feedback are alternated, from prompting (ReAct) to RL training.

Sources: arxiv:2210.03629, arxiv:2510.25510, arxiv:2505.14246

### RL with verifiable rewards for reasoning

Outcome-verifiable rewards and targeted problem synthesis drive reasoning gains.

Sources: arxiv:2506.08989, arxiv:2505.14246, arxiv:2510.25510, arxiv:2306.11885

### Faithfulness and inspection of reasoning

Reasoning chains may not reflect how answers are produced; symbolic execution and representation analysis address this.

Sources: arxiv:2301.13379, arxiv:2410.03595

### Tool selection as a bottleneck

Choosing the right tool limits agent success before reasoning starts.

Sources: arxiv:2503.01763

## Gaps

- **G1.** Whether reasoning steps actually depend on the tool results they cite is untested: rewards check final answers or executable outcomes, and faithfulness work covers solver-executed chains, not interleaved tool use. (evidence: arxiv:2301.13379, arxiv:2210.03629, arxiv:2505.14246, arxiv:2510.25510)

- **G2.** Step-level feedback is shown to help only where tool output is directly executable (SQL); how to give step-level reward when tool results are open-ended (search, vision) is not stated. (evidence: arxiv:2510.25510, arxiv:2505.14246)

- **G3.** Weakness-driven training targets final-answer failures; no source synthesises training problems from reasoning or tool-use process failures. (evidence: arxiv:2506.08989, arxiv:2410.03595)

## Sources

### [arxiv:2503.01763] Retrieval Models Aren't Tool-Savvy: Benchmarking Tool Retrieval for Large Language Models

Zhengliang Shi, Yuhan Wang, Lingyong Yan et al. (2025). *Retrieval Models Aren't Tool-Savvy: Benchmarking Tool Retrieval for Large Language Models*. arXiv. https://doi.org/10.48550/arXiv.2503.01763 <https://arxiv.org/abs/2503.01763>

Local copy: `sources/arxiv_2503.01763.pdf` (downloaded_pdf)

**Summary.** Carried over from day1-agents: ToolRet shows retrieval models select tools poorly from large toolsets, lowering end-to-end pass rates.

**Key findings.**
- Strong conventional IR models perform poorly on tool retrieval.
- Poor retrieval lowers end-to-end task pass rate.

**Limitations.**
- Does not study reasoning; text tasks only.

**Relevance:** 2/5

### [arxiv:2505.14246] Visual Agentic Reinforcement Fine-Tuning

Ziyu Liu, Yuhang Zang, Yushan Zou et al. (2025). *Visual Agentic Reinforcement Fine-Tuning*. arXiv. https://doi.org/10.48550/arXiv.2505.14246 <https://arxiv.org/abs/2505.14246>

Local copy: `sources/arxiv_2505.14246.pdf` (downloaded_pdf)

**Summary.** Carried over from day1-agents: Visual-ARFT uses RL with verifiable rewards to teach vision-language models agentic search and coding.

**Key findings.**
- Large F1/EM gains on MAT-Search and MAT-Coding.
- Rewards are verifiable outcomes rather than step checks.

**Limitations.**
- Step-level use of tool results is not evaluated.

**Relevance:** 4/5

### [arxiv:2410.03595] Understanding Reasoning in Chain-of-Thought from the Hopfieldian View

Lijie Hu, Liang Liu, Shu Yang et al. (2024). *Understanding Reasoning in Chain-of-Thought from the Hopfieldian View*. arXiv. https://doi.org/10.48550/arXiv.2410.03595 <https://arxiv.org/abs/2410.03595>

Local copy: `sources/arxiv_2410.03595.pdf` (downloaded_pdf)

**Summary.** Interprets chain-of-thought reasoning through a Hopfieldian view of representation spaces and proposes Representation-of-Thought to localise errors and improve robustness.

**Key findings.**
- Reasoning errors in CoT responses can be localised in representation space.
- RoT improves the robustness of CoT reasoning.

**Limitations.**
- Exact gains are not stated in the evidence.
- Does not consider tool use.

**Relevance:** 3/5

### [arxiv:2306.11885] Reward Shaping via Diffusion Process in Reinforcement Learning

Peeyush Kumar (2023). *Reward Shaping via Diffusion Process in Reinforcement Learning*. arXiv. https://doi.org/10.48550/arXiv.2306.11885 <https://arxiv.org/abs/2306.11885>

Local copy: `sources/arxiv_2306.11885.pdf` (downloaded_pdf)

**Summary.** Theoretical framework for reward shaping in Markov decision processes using diffusion processes and stochastic thermodynamics.

**Key findings.**
- Frames the exploration-exploitation trade-off as a maximum-entropy or information-cost optimisation.

**Limitations.**
- Not about language models or reasoning; matched on reinforcement-learning keywords only.

**Relevance:** 1/5

### [arxiv:2301.13379] Faithful Chain-of-Thought Reasoning

Qing Lyu, Shreya Havaldar, Adam Stein et al. (2023). *Faithful Chain-of-Thought Reasoning*. arXiv. https://doi.org/10.48550/arXiv.2301.13379 <https://arxiv.org/abs/2301.13379>

Local copy: `sources/arxiv_2301.13379.pdf` (downloaded_pdf)

**Summary.** Faithful CoT translates a natural-language query into a symbolic reasoning chain and executes it with a deterministic solver so the chain faithfully explains the answer.

**Key findings.**
- Standard CoT chains do not necessarily reflect how the model reaches its answer.
- Faithful CoT beats standard CoT on 9 of 10 benchmarks, e.g. +6.3% relative on math word problems and +21.4% on relational inference.
- With GPT-4 and Codex it sets few-shot state of the art on 7 datasets.

**Limitations.**
- The translation step remains opaque, so the pipeline is not fully interpretable.

**Relevance:** 5/5

### [arxiv:2506.08989] SwS: Self-aware Weakness-driven Problem Synthesis in Reinforcement Learning for LLM Reasoning

Xiao Liang, Zhong-Zhi Li, Yeyun Gong et al. (2025). *SwS: Self-aware Weakness-driven Problem Synthesis in Reinforcement Learning for LLM Reasoning*. arXiv. https://doi.org/10.48550/arXiv.2506.08989 <https://arxiv.org/abs/2506.08989>

Local copy: `sources/arxiv_2506.08989.pdf` (downloaded_pdf)

**Summary.** SwS synthesises new RL training problems from questions the model consistently fails during RL with verifiable rewards.

**Key findings.**
- Problem synthesis targeted at model weaknesses improves reasoning generalisation.
- Most synthesis strategies expand problem sets without regard to model capability.

**Limitations.**
- Focused on math with verifiable final answers.

**Relevance:** 3/5

### [arxiv:2210.03629] ReAct: Synergizing Reasoning and Acting in Language Models

Shunyu Yao, Jeffrey Zhao, Dian Yu et al. (2022). *ReAct: Synergizing Reasoning and Acting in Language Models*. arXiv. https://doi.org/10.48550/arXiv.2210.03629 <https://arxiv.org/abs/2210.03629>

Local copy: `sources/arxiv_2210.03629.pdf` (downloaded_pdf)

**Summary.** ReAct prompts LLMs to interleave reasoning traces with task actions that query external sources.

**Key findings.**
- Reasoning traces help induce, track and update action plans and handle exceptions.
- On HotpotQA and Fever, ReAct reduces hallucination and error propagation compared with reasoning alone.
- Interleaved trajectories are more interpretable and diagnosable.

**Limitations.**
- Main experiments use PaLM, which is not openly accessible.

**Relevance:** 5/5

### [arxiv:2510.25510] MTIR-SQL: Multi-turn Tool-Integrated Reasoning Reinforcement Learning for Text-to-SQL

Zekun Xu, Siyu Xia, Chuhuai Yue et al. (2025). *MTIR-SQL: Multi-turn Tool-Integrated Reasoning Reinforcement Learning for Text-to-SQL*. arXiv. https://doi.org/10.48550/arXiv.2510.25510 <https://arxiv.org/abs/2510.25510>

Local copy: `sources/arxiv_2510.25510.pdf` (downloaded_pdf)

**Summary.** MTIR-SQL trains text-to-SQL models with multi-turn tool-integrated reasoning RL that uses database execution feedback at each step.

**Key findings.**
- Execution results inform each reasoning step, creating a self-correcting loop.
- Extends GRPO with trajectory filtering and removes KL constraints.
- Reaches 64.4% accuracy on BIRD-SQL.

**Limitations.**
- Single domain (SQL) where tool feedback is directly executable.

**Relevance:** 4/5

---
Queries run: 7. Sources kept: 8. Search results rejected as off-topic or below relevance threshold: 68.
