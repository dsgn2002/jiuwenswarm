# MM-VeriRec: Failure-Guided Fusion for Verifiable Agentic Multimodal Recommendation

Source: arxiv:2609.31718. Authors: Yufeng Wang. Year: 2026. DOI: 10.1145/3841452.3841492.

Topics: Where do multimodal LLM agents fail when deciding which tool to call, forming the call, and using tool outputs, and which failure mode is the most tractable research target?

Summary: MM-VeriRec verifies agentic multimodal recommendations against deterministic visual attributes and uses a failure taxonomy to route repairs.

Abstract: Images often carry recommendation constraints that text metadata only hints at. A movie may need to look dark, a product may need a minimal style, and a visually impossible request should be rejected. Agentic multimodal recommenders must reason over text-image evidence, decide when visual evidence is decisive, and abstain when no valid action exists. We introduce MM-VeriRec, a verifiable multimodal recommendation protocol and failure-guided fusion method for hidden visual constraints, image-text mismatch, and impossible-task abstention. MM-VeriRec builds tasks from real movie-poster and product-image datasets, verifies each recommendation with deterministic visual attributes, and converts failures into actionable labels: text-trap following, visual ignorance, and false acceptance. Fusion should not merely concatenate modalities, but should diagnose which modality failed and route to the appropriate repair. Across MM-ML 1M and Amazon Reviews datasets, stronger text and vision embeddings improve retrieval but do not remove these failure modes, whereas failure-guided fusion does. The adaptive attribute gate reads the same tags the verifier checks and its scores are verifier-aligned upper bounds testing whether the taxonomy routes to the correct repair. More informative is transfer under a non-aligned gate: an independently derived leave-one-out CLIP detector still reaches 0.7028 and 0.6111 visual-grounded success, above both a VBPR baseline and plain fusion. The text-versus-visual gap reproduces across two LLM families, and the repair that helps differs by domain. MM-VeriRec is both a benchmark and a practical diagnostic loop for trustworthy agentic multimodal recommendation.

Key findings:
- Failures are labelled text-trap following, visual ignorance and false acceptance.
- Stronger text and vision embeddings improve retrieval but do not remove these failure modes; failure-guided fusion does.
- Cross-modal checks should be applied where the observed failure distribution says they matter.

Limitations:
- Recommendation domain; relies on an aligned visual analyser.
