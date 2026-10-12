# OR-Toolformer: Modeling and Solving Operations Research Problems with Tool Augmented Large Language Models

Source: arxiv:2510.01253. Authors: Jianzhang Zhang, Jialong Zhou, Chuang Liu. Year: 2025. DOI: 10.48550/arXiv.2510.01253.

Topics: Where do multimodal LLM agents fail when deciding which tool to call, forming the call, and using tool outputs, and which failure mode is the most tractable research target?

Summary: OR-Toolformer fine-tunes Llama-3.1-8B to call external operations-research solvers.

Abstract: Large language models (LLMs) demonstrate strong mathematical reasoning, but reliance on closed-source APIs for OR tasks raises privacy concerns, and training open-source models from scratch incurs high compute costs. We introduce OR-Toolformer, which fine-tunes Llama-3.1-8B-Instruct with a semi-automatic data synthesis pipeline that generates diverse OR problem-answer pairs and augments the model with external solvers to produce API calls. On three of four standard benchmarks, OR-Toolformer achieves up to 80.1% execution accuracy, exceeding size-matched baselines by over 4.3%. In zero-shot evaluation on two unseen OR problem types, it attains 54% average accuracy, a 21 percentage-point improvement over the strongest baseline. These findings validate the efficacy of tool-augmented fine-tuning LLMs for accurate and generalizable OR problem modeling and solving.

Key findings:
- Up to 80.1% execution accuracy on three of four benchmarks, over 4.3% above size-matched baselines.
- 54% zero-shot accuracy on unseen problem types, 21 points over the strongest baseline.

Limitations:
- Lower accuracy on complex industry-scale tasks.
- Only one open-source model evaluated.
