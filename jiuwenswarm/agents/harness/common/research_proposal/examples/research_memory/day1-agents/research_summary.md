# Literature review: Where do multimodal LLM agents fail when deciding which tool to call, forming the call, and using tool outputs, and which failure mode is the most tractable research target?

## Synthesis

Tool use by language and multimodal agents breaks into three steps that the sources study separately: choosing a tool, forming a correct call, and using what comes back. Toolformer shows a model can learn when and how to call APIs from self-supervision, and later work fine-tunes for domain tools (OR solvers) or trains vision-language agents with verifiable rewards to search and run image-manipulation code. Benchmarks expose weaknesses at each step: retrieval models are poor at selecting tools from large toolsets, even frontier models break simple format rules in function arguments, and multimodal models struggle with dynamic GUI content.

The multimodal sources add a fourth problem: deciding when visual evidence is decisive and checking tool outputs against it. Failure-guided fusion shows that stronger embeddings do not remove visual ignorance or false acceptance, and memory-conditioned tool calling shows that what an agent looks up depends on context beyond the image. No source evaluates the full choose-call-use chain for multimodal agents end to end, so where errors compound is not stated.

## Themes

### Learning when and how to call tools

Self-supervised and fine-tuning approaches teach models to call APIs and domain tools.

Sources: arxiv:2302.04761, arxiv:2510.01253, arxiv:2505.14246

### Step-level tool-use failures

Benchmarks isolate weak steps: tool retrieval and argument format adherence.

Sources: arxiv:2503.01763, arxiv:2509.18420

### Multimodal grounding of tool use

Multimodal agents must perceive dynamic visual content, decide when vision is decisive, and condition lookups on context.

Sources: arxiv:2406.10819, arxiv:2609.31718, arxiv:2607.09822, arxiv:2505.14246

## Gaps

- **G1.** Multimodal agents do not verify tool outputs against visual evidence: visual ignorance and false acceptance persist even with stronger embeddings, and training rewards check final answers rather than intermediate tool results. (evidence: arxiv:2609.31718, arxiv:2505.14246)

- **G2.** Tool selection is evaluated separately from argument formation: retrieval benchmarks pre-annotate or isolate tools, and function-calling benchmarks offer a single function, so compounding errors across the choose-call-use chain are unmeasured. (evidence: arxiv:2503.01763, arxiv:2509.18420)

- **G3.** Multimodal agents lack temporal and contextual grounding for tool use: they struggle with dynamic GUI content and their lookups depend on context such as memory that benchmarks do not model. (evidence: arxiv:2406.10819, arxiv:2607.09822)

## Sources

### [arxiv:2505.14246] Visual Agentic Reinforcement Fine-Tuning

Ziyu Liu, Yuhang Zang, Yushan Zou et al. (2025). *Visual Agentic Reinforcement Fine-Tuning*. arXiv. https://doi.org/10.48550/arXiv.2505.14246 <https://arxiv.org/abs/2505.14246>

Local copy: `sources/arxiv_2505.14246.pdf` (downloaded_pdf)

**Summary.** Visual-ARFT trains open-source vision-language models with reinforcement fine-tuning and verifiable rewards to search the web and write image-manipulation code, and introduces the MAT benchmark.

**Key findings.**
- Visual-ARFT beats its baseline by +18.6% F1 / +13.0% EM on MAT-Coding and +10.3% F1 / +8.7% EM on MAT-Search.
- It outperforms GPT-4o on MAT and generalises to multi-hop QA (2Wiki, HotpotQA).
- Multimodal agentic capabilities and their benchmarks are less explored than language-only tool use.

**Limitations.**
- Tool set limited to search and image-processing code.

**Relevance:** 5/5

### [arxiv:2302.04761] Toolformer: Language Models Can Teach Themselves to Use Tools

Timo Schick, Jane Dwivedi-Yu, Roberto Dessì et al. (2023). *Toolformer: Language Models Can Teach Themselves to Use Tools*. arXiv. https://doi.org/10.48550/arXiv.2302.04761 <https://arxiv.org/abs/2302.04761>

Local copy: `sources/arxiv_2302.04761.pdf` (downloaded_pdf)

**Summary.** Toolformer teaches a language model to decide which APIs to call, when, with what arguments, and how to use results, via self-supervised filtering of sampled calls.

**Key findings.**
- API calls are kept if they reduce perplexity on future tokens.
- A 6.7B GPT-J with tools outperforms much larger GPT-3 on several downstream tasks zero-shot.

**Limitations.**
- Text-only tools; multimodal inputs are not covered.

**Relevance:** 4/5

### [arxiv:2503.01763] Retrieval Models Aren't Tool-Savvy: Benchmarking Tool Retrieval for Large Language Models

Zhengliang Shi, Yuhan Wang, Lingyong Yan et al. (2025). *Retrieval Models Aren't Tool-Savvy: Benchmarking Tool Retrieval for Large Language Models*. arXiv. https://doi.org/10.48550/arXiv.2503.01763 <https://arxiv.org/abs/2503.01763>

Local copy: `sources/arxiv_2503.01763.pdf` (downloaded_pdf)

**Summary.** ToolRet benchmarks tool retrieval: selecting useful tools from a 43k-tool corpus for 7.6k tasks.

**Key findings.**
- Models strong on conventional IR benchmarks perform poorly on tool retrieval.
- Low retrieval quality degrades the end-to-end task pass rate of tool-using LLMs.
- A 200k-instance training set substantially improves tool retrieval.

**Limitations.**
- Text tasks only.

**Relevance:** 4/5

### [arxiv:2406.10819] GUI-World: A Video Benchmark and Dataset for Multimodal GUI-oriented Understanding

Dongping Chen, Yue Huang, Siyuan Wu et al. (2024). *GUI-World: A Video Benchmark and Dataset for Multimodal GUI-oriented Understanding*. arXiv. https://doi.org/10.48550/arXiv.2406.10819 <https://arxiv.org/abs/2406.10819>

Local copy: `sources/arxiv_2406.10819.pdf` (downloaded_pdf)

**Summary.** GUI-World is a video benchmark covering six GUI scenarios to test multimodal models' understanding of dynamic, multi-step GUI content.

**Key findings.**
- Current models struggle with dynamic GUI content without annotated keyframes or operation history.
- Video LLMs fall short on all GUI-oriented tasks evaluated.

**Limitations.**
- Measures understanding rather than tool or action execution.

**Relevance:** 4/5

### [arxiv:2509.18420] Instruction-Following Evaluation in Function Calling for Large Language Models

Nikolai Skripko (2025). *Instruction-Following Evaluation in Function Calling for Large Language Models*. arXiv. https://doi.org/10.48550/arXiv.2509.18420 <https://arxiv.org/abs/2509.18420>

Local copy: `sources/arxiv_2509.18420.pdf` (downloaded_pdf)

**Summary.** IFEval-FC tests whether models follow format instructions embedded in function-parameter descriptions.

**Key findings.**
- No evaluated model surpassed 80% accuracy, including GPT-5 and Claude 4.1 Opus.
- Existing function-calling benchmarks check argument correctness but not format adherence.

**Limitations.**
- Only one function is available per test; tool selection is not tested.
- English only.

**Relevance:** 4/5

### [arxiv:2609.31718] MM-VeriRec: Failure-Guided Fusion for Verifiable Agentic Multimodal Recommendation

Yufeng Wang (2026). *MM-VeriRec: Failure-Guided Fusion for Verifiable Agentic Multimodal Recommendation*. The 1st International Workshop on Agentic Multimodal Intelligence, co-located with ACM Multimedia 2026. https://doi.org/10.1145/3841452.3841492 <https://arxiv.org/abs/2609.31718>

Local copy: `sources/arxiv_2609.31718.pdf` (downloaded_pdf)

**Summary.** MM-VeriRec verifies agentic multimodal recommendations against deterministic visual attributes and uses a failure taxonomy to route repairs.

**Key findings.**
- Failures are labelled text-trap following, visual ignorance and false acceptance.
- Stronger text and vision embeddings improve retrieval but do not remove these failure modes; failure-guided fusion does.
- Cross-modal checks should be applied where the observed failure distribution says they matter.

**Limitations.**
- Recommendation domain; relies on an aligned visual analyser.

**Relevance:** 4/5

### [arxiv:2607.09822] Memory-Conditioned Tool Calling for Camera-First Visual Agents

Xiaofan Wu, Xi Zeng, Miaoxia Chen et al. (2026). *Memory-Conditioned Tool Calling for Camera-First Visual Agents*. arXiv. https://doi.org/10.48550/arXiv.2607.09822 <https://arxiv.org/abs/2607.09822>

Local copy: `sources/arxiv_2607.09822.pdf` (downloaded_pdf)

**Summary.** Studies whether personal visual memory improves tool choice and arguments for camera-first agents that receive only an image.

**Key findings.**
- Removing the three-layer memory block reduces tool-query relevance from 4.21 to 3.74 (11.2% relative) and end-to-end utility from 0.842 to 0.760.

**Limitations.**
- Synthetic memory blocks; multi-session write-back from live histories is not measured.

**Relevance:** 3/5

### [arxiv:2510.01253] OR-Toolformer: Modeling and Solving Operations Research Problems with Tool Augmented Large Language Models

Jianzhang Zhang, Jialong Zhou, Chuang Liu (2025). *OR-Toolformer: Modeling and Solving Operations Research Problems with Tool Augmented Large Language Models*. arXiv. https://doi.org/10.48550/arXiv.2510.01253 <https://arxiv.org/abs/2510.01253>

Local copy: `sources/arxiv_2510.01253.pdf` (downloaded_pdf)

**Summary.** OR-Toolformer fine-tunes Llama-3.1-8B to call external operations-research solvers.

**Key findings.**
- Up to 80.1% execution accuracy on three of four benchmarks, over 4.3% above size-matched baselines.
- 54% zero-shot accuracy on unseen problem types, 21 points over the strongest baseline.

**Limitations.**
- Lower accuracy on complex industry-scale tasks.
- Only one open-source model evaluated.

**Relevance:** 2/5

---
Queries run: 7. Sources kept: 8. Search results rejected as off-topic or below relevance threshold: 53.
