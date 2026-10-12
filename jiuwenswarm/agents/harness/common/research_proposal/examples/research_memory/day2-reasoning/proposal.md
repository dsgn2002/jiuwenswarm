# Does the Reasoning Read the Tool? Measuring Evidence Dependence in Tool-Integrated Reasoning

## Summary

Reasoning models increasingly interleave thought with tool calls [1] and are trained with rewards on final or executable outcomes [2] [3]. Chain-of-thought text is known not to necessarily reflect how answers are produced [4], but nobody has measured whether reasoning steps in tool-using agents depend on the tool results they cite. We propose counterfactual tool-result perturbation as a measurement, and test whether outcome-trained models ignore tool evidence more often than prompted agents.

## Background

ReAct showed that interleaving reasoning traces with actions lets retrieved information update an agent's plan and reduces hallucination on HotpotQA and Fever [1]. Reinforcement learning has since been applied to the same loop: execution feedback at each step improves text-to-SQL [3], and verifiable rewards teach vision-language agents to search and run image tools [2]. In both cases the reward checks an outcome, not whether intermediate reasoning used the tool result.

Separately, chain-of-thought faithfulness work shows reasoning text may not explain the answer, and fixes this only by executing a symbolic chain with a deterministic solver [4]. Our earlier session on multimodal agents found that tool outputs go unverified against the evidence they should agree with; this proposal asks the reasoning-side question.

## Research questions

1. How often does a reasoning step remain unchanged when the tool result it follows is replaced by a contradicting one?
2. Do outcome-trained RL agents ignore tool results more than prompted ReAct agents?
3. Does evidence dependence predict accuracy across tasks?

## Hypotheses

1. H1: At least 20% of reasoning steps are unchanged after a contradicting tool result is substituted.
2. H2: The insensitivity rate is higher for outcome-only RL agents than for prompted ReAct agents.
3. H3: Tasks with higher evidence dependence show higher final accuracy.

## Method

Collect interleaved trajectories from a prompted ReAct agent [1] and an outcome-trained tool-integrated agent [3] on multi-hop QA and text-to-SQL. At each tool call, replay the trajectory with the tool result replaced by a contradicting or irrelevant one, and compare the next reasoning step and the final answer with the original. Score dependence as the share of perturbations that change the next step in the direction the new evidence implies, judged by string match where possible and by a separate model otherwise.

## Evaluation

Metrics: perturbation-sensitivity rate of the next step and final answer; accuracy on unperturbed tasks. Baselines: random-perturbation controls and a no-tool reasoning baseline. H1 is refuted if fewer than 20% of steps are insensitive; H2 if RL and prompted agents do not differ; H3 if dependence and accuracy are uncorrelated.

## Risks and mitigations

- Contradicting results may make questions unanswerable: also test neutral, irrelevant replacements.
- Judging step changes with a model may be noisy: validate on a hand-labelled sample.

## Milestones

1. Build the perturbation harness for ReAct-style trajectories.
2. Collect trajectories from prompted and RL-trained agents.
3. Run perturbations and validate the step-change judge.
4. Analyse sensitivity against accuracy and write up.

## References

1. Shunyu Yao, Jeffrey Zhao, Dian Yu et al. (2022). *ReAct: Synergizing Reasoning and Acting in Language Models*. arXiv. https://doi.org/10.48550/arXiv.2210.03629 <https://arxiv.org/abs/2210.03629>
2. Ziyu Liu, Yuhang Zang, Yushan Zou et al. (2025). *Visual Agentic Reinforcement Fine-Tuning*. arXiv. https://doi.org/10.48550/arXiv.2505.14246 <https://arxiv.org/abs/2505.14246>
3. Zekun Xu, Siyu Xia, Chuhuai Yue et al. (2025). *MTIR-SQL: Multi-turn Tool-Integrated Reasoning Reinforcement Learning for Text-to-SQL*. arXiv. https://doi.org/10.48550/arXiv.2510.25510 <https://arxiv.org/abs/2510.25510>
4. Qing Lyu, Shreya Havaldar, Adam Stein et al. (2023). *Faithful Chain-of-Thought Reasoning*. arXiv. https://doi.org/10.48550/arXiv.2301.13379 <https://arxiv.org/abs/2301.13379>
