# Toolformer: Language Models Can Teach Themselves to Use Tools

Source: arxiv:2302.04761. Authors: Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer. Year: 2023. DOI: 10.48550/arXiv.2302.04761.

Topics: Where do multimodal LLM agents fail when deciding which tool to call, forming the call, and using tool outputs, and which failure mode is the most tractable research target?

Summary: Toolformer teaches a language model to decide which APIs to call, when, with what arguments, and how to use results, via self-supervised filtering of sampled calls.

Abstract: Language models (LMs) exhibit remarkable abilities to solve new tasks from just a few examples or textual instructions, especially at scale. They also, paradoxically, struggle with basic functionality, such as arithmetic or factual lookup, where much simpler and smaller models excel. In this paper, we show that LMs can teach themselves to use external tools via simple APIs and achieve the best of both worlds. We introduce Toolformer, a model trained to decide which APIs to call, when to call them, what arguments to pass, and how to best incorporate the results into future token prediction. This is done in a self-supervised way, requiring nothing more than a handful of demonstrations for each API. We incorporate a range of tools, including a calculator, a Q\&A system, two different search engines, a translation system, and a calendar. Toolformer achieves substantially improved zero-shot performance across a variety of downstream tasks, often competitive with much larger models, without sacrificing its core language modeling abilities.

Key findings:
- API calls are kept if they reduce perplexity on future tokens.
- A 6.7B GPT-J with tools outperforms much larger GPT-3 on several downstream tasks zero-shot.

Limitations:
- Text-only tools; multimodal inputs are not covered.
