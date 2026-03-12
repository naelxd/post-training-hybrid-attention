# post-training-hybrid-attention

Code & notes on the Post-Training Hybrid Attention (Full + Linear) project

## Project description

**Topic**: Post-Training Hybrid Full + Linear Attention (Частичная линеаризация слоев внимания без дообучения)

**Description**: This project explores post-training conversion of transformer attention into a hybrid scheme that mixes exact full (softmax) attention with linear/approximate attention in selected layers/heads. The goal is to reduce inference-time memory -- primarily KV-cache size/pressure during long-context decoding -- while preserving model quality, without any additional training or fine-tuning.

## Research objective

### Abstract

Inference with large language models is often bottlenecked by KV-cache memory: the cache grows linearly with sequence length and can dominate GPU memory in long-context decoding, limiting batch size and throughput. We study a post-training "hybrid attention" approach that keeps full (softmax) attention in a subset of layers/heads and replaces the remainder with a linear-time attention approximation that reduces or avoids storing large KV states. The central hypothesis is that not all attention blocks are equally critical for quality, so a careful selection of which blocks remain full attention can preserve perplexity and downstream task performance. We evaluate memory savings (KV footprint), throughput/latency, and quality across context lengths, aiming to produce a practical recipe for retrofitting existing checkpoints for long-context inference under tight memory budgets.

### Key research question

- How much inference KV-cache memory can we save by replacing a subset of full-attention layers/heads with linear attention post hoc (no training) while preserving the model capabilities, especially at long context lengths?

- Which layers/heads are "safe" to linearize, and does the optimal replacement pattern depend on context length?

### Why this deserves studying

- KV-cache memory is a practical deployment limiter for long-context inference; reducing it directly increases feasible context, batch size, and throughput on fixed hardware.

- Most methods require re-training/fine-tuning or architecture changes; a post-training recipe can be applied immediately to existing checkpoints.

- This work can also reveal which attention blocks are functionally essential for quality, providing actionable guidance for model compression and inference optimization.
