# Research session day2-reasoning

Question: When LLMs interleave long reasoning with tool calls, how far do their reasoning steps actually check and depend on intermediate tool results, and can process-level rewards make reasoning verify tool outputs rather than only reach correct final answers?

Key concepts: chain-of-thought reasoning, reinforcement learning for reasoning, process reward models, reasoning faithfulness, tool use, interleaved reasoning and acting, verification

Synthesis: The retrieved work treats reasoning and acting as increasingly coupled. ReAct interleaves reasoning traces with actions so that retrieved information updates the plan, and newer reinforcement-learning methods feed tool or execution results back into each reasoning step (multi-turn tool-integrated RL for text-to-SQL; visual agentic RL with verifiable rewards, carried over from yesterday's session). RL with verifiable rewards also drives pure reasoning gains, with work on which training problems to synthesise from model weaknesses.

A second line asks whether reasoning traces can be trusted. Standard chain of thought does not necessarily reflect how the answer is reached; Faithful CoT guarantees faithfulness by translating the query into a symbolic chain run by a deterministic solver, and representation-level analysis localises reasoning errors. Across both lines, rewards and checks attach to final answers or executable outcomes; no source tests whether intermediate reasoning steps actually depend on the tool results they cite, which connects to yesterday's gap G1 on unverified tool outputs.

Recommended direction: Measuring whether reasoning uses tool results. In interleaved reasoning-and-tool trajectories, perturbing a tool result often leaves the following reasoning step and final answer unchanged, showing the reasoning does not depend on the evidence it cites; this rate is higher for outcome-only RL models than for prompted ReAct agents.

Gaps:
- G1: Whether reasoning steps actually depend on the tool results they cite is untested: rewards check final answers or executable outcomes, and faithfulness work covers solver-executed chains, not interleaved tool use.
- G2: Step-level feedback is shown to help only where tool output is directly executable (SQL); how to give step-level reward when tool results are open-ended (search, vision) is not stated.
- G3: Weakness-driven training targets final-answer failures; no source synthesises training problems from reasoning or tool-use process failures.
