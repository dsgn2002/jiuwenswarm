# Instruction-Following Evaluation in Function Calling for Large Language Models

Source: arxiv:2509.18420. Authors: Nikolai Skripko. Year: 2025. DOI: 10.48550/arXiv.2509.18420.

Topics: Where do multimodal LLM agents fail when deciding which tool to call, forming the call, and using tool outputs, and which failure mode is the most tractable research target?

Summary: IFEval-FC tests whether models follow format instructions embedded in function-parameter descriptions.

Abstract: Function calling is a core capability of large language models, essential for AI agents. Existing benchmarks such as the Berkeley Function Calling Leaderboard (BFCL), tau^2-Bench (arXiv:2506.07982), and ACEBench (arXiv:2501.12851) evaluate argument correctness but do not test adherence to format instructions embedded in parameter descriptions, such as enclosing values in double quotes or using ISO date formats. We introduce IFEval-FC, a benchmark inspired by IFEval (arXiv:2311.07911) that assesses precise instruction following in function calling. IFEval-FC encodes verifiable formats directly within JSON schema descriptions, for example specifying that a value must not contain punctuation. It includes 750 test cases, each consisting of a function with an embedded format for one of its input parameters and a corresponding user query. Evaluation is fully algorithmic, ensuring objectivity, reproducibility, and scalability. Our results show that even state-of-the-art proprietary models, including GPT-5 and Claude 4.1 Opus, frequently fail to follow basic formatting rules, highlighting a practical limitation for real-world agent systems. The complete codebase and data are publicly available at https://github.com/Skripkon/IFEval-FC.

Key findings:
- No evaluated model surpassed 80% accuracy, including GPT-5 and Claude 4.1 Opus.
- Existing function-calling benchmarks check argument correctness but not format adherence.

Limitations:
- Only one function is available per test; tool selection is not tested.
- English only.
