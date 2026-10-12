# Memory-Conditioned Tool Calling for Camera-First Visual Agents

Source: arxiv:2607.09822. Authors: Xiaofan Wu, Xi Zeng, Miaoxia Chen, Peishan Chen, Shuyan Li, Jiyun Yao. Year: 2026. DOI: 10.48550/arXiv.2607.09822.

Topics: Where do multimodal LLM agents fail when deciding which tool to call, forming the call, and using tool outputs, and which failure mode is the most tractable research target?

Summary: Studies whether personal visual memory improves tool choice and arguments for camera-first agents that receive only an image.

Abstract: Recognition tells an agent what is in an image; personal memory affects what is worth looking up next. In a camera-first setting the user can send only an image, so the agent must form the lookups. We study whether personal visual memory improves agent-side tool choice and tool arguments, and thereby more user-aligned multi-tool lookups. The design uses a three-layer personal visual memory (profile, short-term focus, observations) that is loaded on each turn to condition an LLM tool-calling loop under camera-first intake, and includes conflict-aware write-back intended to refresh the user model for later captures. On 800 images paired with synthetic memory blocks constructed for controlled ablation, removing the full three-layer memory block reduces tool-query relevance by 0.47 points absolute (4.21 -> 3.74 on a 5-point scale; 11.2% relative) and end-to-end utility by 0.082 absolute (0.842 -> 0.760; 9.7% relative). These results measure memory conditioning of tool policy under image-only intake with fixed synthetic blocks, not multi-session write-back from live user histories.

Key findings:
- Removing the three-layer memory block reduces tool-query relevance from 4.21 to 3.74 (11.2% relative) and end-to-end utility from 0.842 to 0.760.

Limitations:
- Synthetic memory blocks; multi-session write-back from live histories is not measured.
