# Potential research directions

| ID | Direction | Gap | Novelty | Feasibility | Impact | Score |
| --- | --- | --- | --- | --- | --- | --- |
| D1 | Measuring whether reasoning uses tool results (recommended) | G1 | 4 | 4 | 4 | 4.0 |
| D2 | Step-level rewards for open-ended tool results | G2 | 4 | 2 | 4 | 3.33 |
| D3 | Process-failure-driven problem synthesis | G3 | 3 | 3 | 3 | 3.0 |

**Selection.** D1 is the measurement both other directions need: it shows whether reasoning ignores tool results before anyone designs rewards (D2) or training data (D3) to fix it. It is feasible with existing agents and benchmarks, and it connects today's reasoning question to yesterday's unverified-tool-output gap.

## D1. Measuring whether reasoning uses tool results

**Hypothesis.** In interleaved reasoning-and-tool trajectories, perturbing a tool result often leaves the following reasoning step and final answer unchanged, showing the reasoning does not depend on the evidence it cites; this rate is higher for outcome-only RL models than for prompted ReAct agents.

**Rationale.** Faithfulness has been studied for solver-executed chains but not for interleaved tool use, and rewards check outcomes, not dependence on tool results. It also tests yesterday's gap G1 from the reasoning side.

**Builds on:** arxiv:2210.03629, arxiv:2301.13379, arxiv:2505.14246, arxiv:2510.25510

**First experiment.** Run a ReAct agent on HotpotQA-style tasks, swap one retrieved passage for a contradicting one, and measure how often the next reasoning step and the answer change.

**Risks.**
- Perturbations may make tasks unanswerable rather than test dependence.
- Behaviour may differ across model families.

## D2. Step-level rewards for open-ended tool results

**Hypothesis.** Rewarding each reasoning step for consistency with the tool result it follows improves accuracy over outcome-only rewards in search and vision tools, as execution feedback does for SQL.

**Rationale.** Step feedback helps where tool output is executable; open-ended tools lack an equivalent signal.

**Builds on:** arxiv:2510.25510, arxiv:2505.14246

**First experiment.** Add an LLM-judged step-consistency reward to a small tool-integrated RL run and compare with outcome-only reward.

**Risks.**
- Step-consistency judges may be unreliable and reward-hackable.

## D3. Process-failure-driven problem synthesis

**Hypothesis.** Synthesising RL training problems from process failures (ignored tool results, wrong tool) improves tool-integrated reasoning more than synthesis from final-answer failures.

**Rationale.** Weakness-driven synthesis currently uses only final-answer failures.

**Builds on:** arxiv:2506.08989, arxiv:2410.03595

**First experiment.** Label process failures in tool-integrated trajectories and compare synthesis from those against SwS-style final-answer failures.

**Risks.**
- Detecting process failures automatically is itself unsolved.
