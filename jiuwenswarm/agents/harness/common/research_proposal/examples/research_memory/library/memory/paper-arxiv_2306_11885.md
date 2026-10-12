# Reward Shaping via Diffusion Process in Reinforcement Learning

Source: arxiv:2306.11885. Authors: Peeyush Kumar. Year: 2023. DOI: 10.48550/arXiv.2306.11885.

Topics: When LLMs interleave long reasoning with tool calls, how far do their reasoning steps actually check and depend on intermediate tool results, and can process-level rewards make reasoning verify tool outputs rather than only reach correct final answers?

Summary: Theoretical framework for reward shaping in Markov decision processes using diffusion processes and stochastic thermodynamics.

Abstract: Reinforcement Learning (RL) models have continually evolved to navigate the exploration - exploitation trade-off in uncertain Markov Decision Processes (MDPs). In this study, I leverage the principles of stochastic thermodynamics and system dynamics to explore reward shaping via diffusion processes. This provides an elegant framework as a way to think about exploration-exploitation trade-off. This article sheds light on relationships between information entropy, stochastic system dynamics, and their influences on entropy production. This exploration allows us to construct a dual-pronged framework that can be interpreted as either a maximum entropy program for deriving efficient policies or a modified cost optimization program accounting for informational costs and benefits. This work presents a novel perspective on the physical nature of information and its implications for online learning in MDPs, consequently providing a better understanding of information-oriented formulations in RL.

Key findings:
- Frames the exploration-exploitation trade-off as a maximum-entropy or information-cost optimisation.

Limitations:
- Not about language models or reasoning; matched on reinforcement-learning keywords only.
