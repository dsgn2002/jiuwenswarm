# MTIR-SQL: Multi-turn Tool-Integrated Reasoning Reinforcement Learning for Text-to-SQL

Source: arxiv:2510.25510. Authors: Zekun Xu, Siyu Xia, Chuhuai Yue, Jiajun Chai, Mingxue Tian, Xiaohan Wang. Year: 2025. DOI: 10.48550/arXiv.2510.25510.

Topics: When LLMs interleave long reasoning with tool calls, how far do their reasoning steps actually check and depend on intermediate tool results, and can process-level rewards make reasoning verify tool outputs rather than only reach correct final answers?

Summary: MTIR-SQL trains text-to-SQL models with multi-turn tool-integrated reasoning RL that uses database execution feedback at each step.

Abstract: As large language models (LLMs) are increasingly used in Text-to-SQL tasks, Reinforcement Learning (RL) has become a common method for improving performance. Existing methods primarily rely on static execution feedback, which restricts real-time error correction. However, integrating multi-turn tool invocation along with dynamic feedback could significantly improve adaptability and robustness, ultimately enhancing model performance. To address these issues, we propose MTIR-SQL, an innovative Multi-turn Tool-Integrated Reasoning reinforcement learning framework for Text-to-SQL. Our approach introduces an execution-aware multi-turn reasoning paradigm that seamlessly incorporates database execution feedback at each reasoning step, enabling context-sensitive query generation and progressive refinement throughout the reasoning process. The framework extends the GRPO algorithm to accommodate complex multi-turn interaction scenarios. Considering the training instability characteristics of MTIR and the potential for significant Deviation of model distribution from the initial model, we enhance the GRPO algorithm by adding a trajectory filtering mechanism and removing KL loss constraints. Experimental results demonstrate that MTIR-SQL, with 4B parameters, achieves \textbf{64.4}\% accuracy in the BIRD Dev and 84.6% execution accuracy in the SPIDER Dev, significantly outperforming existing approaches.

Key findings:
- Execution results inform each reasoning step, creating a self-correcting loop.
- Extends GRPO with trajectory filtering and removes KL constraints.
- Reaches 64.4% accuracy on BIRD-SQL.

Limitations:
- Single domain (SQL) where tool feedback is directly executable.
