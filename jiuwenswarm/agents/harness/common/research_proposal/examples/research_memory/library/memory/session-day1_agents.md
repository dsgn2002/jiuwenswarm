# Research session day1-agents

Question: Where do multimodal LLM agents fail when deciding which tool to call, forming the call, and using tool outputs, and which failure mode is the most tractable research target?

Key concepts: multimodal agents, tool calling, tool use, vision-language models, visual tools, GUI agents, function calling, benchmarks

Synthesis: Tool use by language and multimodal agents breaks into three steps that the sources study separately: choosing a tool, forming a correct call, and using what comes back. Toolformer shows a model can learn when and how to call APIs from self-supervision, and later work fine-tunes for domain tools (OR solvers) or trains vision-language agents with verifiable rewards to search and run image-manipulation code. Benchmarks expose weaknesses at each step: retrieval models are poor at selecting tools from large toolsets, even frontier models break simple format rules in function arguments, and multimodal models struggle with dynamic GUI content.

The multimodal sources add a fourth problem: deciding when visual evidence is decisive and checking tool outputs against it. Failure-guided fusion shows that stronger embeddings do not remove visual ignorance or false acceptance, and memory-conditioned tool calling shows that what an agent looks up depends on context beyond the image. No source evaluates the full choose-call-use chain for multimodal agents end to end, so where errors compound is not stated.

Recommended direction: Verifying tool outputs against visual evidence. Adding an explicit check of each tool result against the image (consistent, contradicted, or undecidable) before it is used reduces false acceptance in multimodal agents more than stronger perception models do.

Gaps:
- G1: Multimodal agents do not verify tool outputs against visual evidence: visual ignorance and false acceptance persist even with stronger embeddings, and training rewards check final answers rather than intermediate tool results.
- G2: Tool selection is evaluated separately from argument formation: retrieval benchmarks pre-annotate or isolate tools, and function-calling benchmarks offer a single function, so compounding errors across the choose-call-use chain are unmeasured.
- G3: Multimodal agents lack temporal and contextual grounding for tool use: they struggle with dynamic GUI content and their lookups depend on context such as memory that benchmarks do not model.
