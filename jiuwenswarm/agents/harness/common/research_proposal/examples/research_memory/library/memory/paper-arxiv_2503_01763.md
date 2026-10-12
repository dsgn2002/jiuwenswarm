# Retrieval Models Aren't Tool-Savvy: Benchmarking Tool Retrieval for Large Language Models

Source: arxiv:2503.01763. Authors: Zhengliang Shi, Yuhan Wang, Lingyong Yan, Pengjie Ren, Shuaiqiang Wang, Dawei Yin. Year: 2025. DOI: 10.48550/arXiv.2503.01763.

Topics: Where do multimodal LLM agents fail when deciding which tool to call, forming the call, and using tool outputs, and which failure mode is the most tractable research target?; When LLMs interleave long reasoning with tool calls, how far do their reasoning steps actually check and depend on intermediate tool results, and can process-level rewards make reasoning verify tool outputs rather than only reach correct final answers?

Summary: Carried over from day1-agents: ToolRet shows retrieval models select tools poorly from large toolsets, lowering end-to-end pass rates.

Abstract: Tool learning aims to augment large language models (LLMs) with diverse tools, enabling them to act as agents for solving practical tasks. Due to the limited context length of tool-using LLMs, adopting information retrieval (IR) models to select useful tools from large toolsets is a critical initial step. However, the performance of IR models in tool retrieval tasks remains underexplored and unclear. Most tool-use benchmarks simplify this step by manually pre-annotating a small set of relevant tools for each task, which is far from the real-world scenarios. In this paper, we propose ToolRet, a heterogeneous tool retrieval benchmark comprising 7.6k diverse retrieval tasks, and a corpus of 43k tools, collected from existing datasets. We benchmark six types of models on ToolRet. Surprisingly, even the models with strong performance in conventional IR benchmarks, exhibit poor performance on ToolRet. This low retrieval quality degrades the task pass rate of tool-use LLMs. As a further step, we contribute a large-scale training dataset with over 200k instances, which substantially optimizes the tool retrieval ability of IR models.

Key findings:
- Models strong on conventional IR benchmarks perform poorly on tool retrieval.
- Low retrieval quality degrades the end-to-end task pass rate of tool-using LLMs.
- A 200k-instance training set substantially improves tool retrieval.
- Strong conventional IR models perform poorly on tool retrieval.
- Poor retrieval lowers end-to-end task pass rate.

Limitations:
- Text tasks only.
- Does not study reasoning; text tasks only.
