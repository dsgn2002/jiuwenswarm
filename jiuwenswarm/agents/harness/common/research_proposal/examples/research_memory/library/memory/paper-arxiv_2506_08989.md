# SwS: Self-aware Weakness-driven Problem Synthesis in Reinforcement Learning for LLM Reasoning

Source: arxiv:2506.08989. Authors: Xiao Liang, Zhong-Zhi Li, Yeyun Gong, Yang Wang, Hengyuan Zhang, Yelong Shen. Year: 2025. DOI: 10.48550/arXiv.2506.08989.

Topics: When LLMs interleave long reasoning with tool calls, how far do their reasoning steps actually check and depend on intermediate tool results, and can process-level rewards make reasoning verify tool outputs rather than only reach correct final answers?

Summary: SwS synthesises new RL training problems from questions the model consistently fails during RL with verifiable rewards.

Abstract: Reinforcement Learning with Verifiable Rewards (RLVR) has proven effective for training large language models (LLMs) on complex reasoning tasks, such as mathematical problem solving. A prerequisite for the scalability of RLVR is a high-quality problem set with precise and verifiable answers. However, the scarcity of well-crafted human-labeled math problems and limited-verification answers in existing distillation-oriented synthetic datasets limit their effectiveness in RL. Additionally, most problem synthesis strategies indiscriminately expand the problem set without considering the model's capabilities, leading to low efficiency in generating useful questions. To mitigate this issue, we introduce a Self-aware Weakness-driven problem Synthesis framework (SwS) that systematically identifies model deficiencies and leverages them for problem augmentation. Specifically, we define weaknesses as questions that the model consistently fails to learn through its iterative sampling during RL training. We then extract the core concepts from these failure cases and synthesize new problems to strengthen the model's weak areas in subsequent augmented training, enabling it to focus on and gradually overcome its weaknesses. Without relying on external knowledge distillation, our framework enables robust generalization byempowering the model to self-identify and address its weaknesses in RL, yielding average performance gains of 10.0% and 7.7% on 7B and 32B models across eight mainstream reasoning benchmarks.

Key findings:
- Problem synthesis targeted at model weaknesses improves reasoning generalisation.
- Most synthesis strategies expand problem sets without regard to model capability.

Limitations:
- Focused on math with verifiable final answers.
