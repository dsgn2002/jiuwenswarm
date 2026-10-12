# Verify Before You Trust: Checking Tool Outputs Against Visual Evidence in Multimodal Agents

## Summary

Multimodal agents increasingly act by calling search and image-processing tools [1], but they accept tool results that contradict what the image shows: visual ignorance and false acceptance persist even with stronger embeddings [2]. We propose an explicit verification step that labels each tool result as consistent with, contradicted by, or undecidable from the visual evidence before the agent uses it, and test whether it reduces false acceptance more than a stronger perception model does.

## Background

Language models can learn when to call tools and how to use their results [3], and vision-language models trained with verifiable rewards now search the web and run image-manipulation code, beating GPT-4o on multimodal tool benchmarks [1]. Those rewards score the final answer, not whether intermediate tool results were checked.

Evaluations show each step of tool use is fragile: tool retrieval is weak [4] and even frontier models break simple argument formats [5]. For multimodal agents, failure-guided fusion work finds that text-trap following, visual ignorance and false acceptance are not removed by stronger embeddings, and argues for conditional cross-modal checks where failures occur [2].

## Research questions

1. How often do multimodal agents use tool results that the image contradicts?
2. Does a verification step reduce false acceptance more than upgrading the perception model?
3. When is the image undecidable, and how should the agent act then?

## Hypotheses

1. H1: A verification step cuts false acceptance by at least a third on multimodal tool tasks.
2. H2: The same perception upgrade without verification gives a smaller reduction.
3. H3: Abstaining on undecidable cases improves precision at a modest cost in coverage.

## Method

Instrument a multimodal agent on MAT-Search and MAT-Coding style tasks [1] to log every tool result. Add a verifier that compares the result with the image and returns consistent, contradicted or undecidable, using the failure taxonomy of [2] to label errors. Compare four variants: baseline, stronger vision encoder, verifier, and both.

## Evaluation

Metrics: false-acceptance rate of contradicted tool results, final F1/EM, and abstention rate. Baselines: the unmodified agent and the stronger-encoder agent. H1 is refuted if false acceptance drops by less than a third; H2 if the encoder upgrade matches the verifier.

## Risks and mitigations

- The verifier may share perception errors: use a different model family for verification.
- Labelling contradictions is costly: start with tasks whose visual attributes can be checked deterministically.

## Milestones

1. Log tool results and label contradictions on a benchmark sample.
2. Build the verifier and the four variants.
3. Run the comparison and analyse undecidable cases.

## References

1. Ziyu Liu, Yuhang Zang, Yushan Zou et al. (2025). *Visual Agentic Reinforcement Fine-Tuning*. arXiv. https://doi.org/10.48550/arXiv.2505.14246 <https://arxiv.org/abs/2505.14246>
2. Yufeng Wang (2026). *MM-VeriRec: Failure-Guided Fusion for Verifiable Agentic Multimodal Recommendation*. The 1st International Workshop on Agentic Multimodal Intelligence, co-located with ACM Multimedia 2026. https://doi.org/10.1145/3841452.3841492 <https://arxiv.org/abs/2609.31718>
3. Timo Schick, Jane Dwivedi-Yu, Roberto Dessì et al. (2023). *Toolformer: Language Models Can Teach Themselves to Use Tools*. arXiv. https://doi.org/10.48550/arXiv.2302.04761 <https://arxiv.org/abs/2302.04761>
4. Zhengliang Shi, Yuhan Wang, Lingyong Yan et al. (2025). *Retrieval Models Aren't Tool-Savvy: Benchmarking Tool Retrieval for Large Language Models*. arXiv. https://doi.org/10.48550/arXiv.2503.01763 <https://arxiv.org/abs/2503.01763>
5. Nikolai Skripko (2025). *Instruction-Following Evaluation in Function Calling for Large Language Models*. arXiv. https://doi.org/10.48550/arXiv.2509.18420 <https://arxiv.org/abs/2509.18420>
