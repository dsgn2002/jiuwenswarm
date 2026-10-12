# Understanding Reasoning in Chain-of-Thought from the Hopfieldian View

Source: arxiv:2410.03595. Authors: Lijie Hu, Liang Liu, Shu Yang, Xin Chen, Zhen Tan, Muhammad Asif Ali. Year: 2024. DOI: 10.48550/arXiv.2410.03595.

Topics: When LLMs interleave long reasoning with tool calls, how far do their reasoning steps actually check and depend on intermediate tool results, and can process-level rewards make reasoning verify tool outputs rather than only reach correct final answers?

Summary: Interprets chain-of-thought reasoning through a Hopfieldian view of representation spaces and proposes Representation-of-Thought to localise errors and improve robustness.

Abstract: Large Language Models have demonstrated remarkable abilities across various tasks, with Chain-of-Thought (CoT) prompting emerging as a key technique to enhance reasoning capabilities. However, existing research primarily focuses on improving performance, lacking a comprehensive framework to explain and understand the fundamental factors behind CoT's success. To bridge this gap, we introduce a novel perspective grounded in the Hopfieldian view of cognition in cognitive neuroscience. We establish a connection between CoT reasoning and key cognitive elements such as stimuli, actions, neural populations, and representation spaces. From our view, we can understand the reasoning process as the movement between these representation spaces. Building on this insight, we develop a method for localizing reasoning errors in the response of CoTs. Moreover, we propose the Representation-of-Thought (RoT) framework, which leverages the robustness of low-dimensional representation spaces to enhance the robustness of the reasoning process in CoTs. Experimental results demonstrate that RoT improves the robustness and interpretability of CoT reasoning while offering fine-grained control over the reasoning process.

Key findings:
- Reasoning errors in CoT responses can be localised in representation space.
- RoT improves the robustness of CoT reasoning.

Limitations:
- Exact gains are not stated in the evidence.
- Does not consider tool use.
