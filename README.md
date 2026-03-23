# frame-symbolic-model

> Training pipeline for building a compact AI model that outputs deterministic symbolic programs instead of natural language by distilling high-entropy model behavior into a canonical low-entropy representation, using a custom tokenizer and constrained grammar to maximize efficiency and enforce correctness through AST parsing and execution equivalence, enabling fast, verifiable, and low-cost inference aligned with FRAME’s intent-driven execution system

---

> [!IMPORTANT]  
> * **deterministic program generation** instead of natural language
> * extreme token compression and **lower inference cost**
> * **faster inference** due to shorter sequences
> * **elimination of linguistic ambiguity and hallucination** surface
> * **exact AST-level correctness and validation**
> * execution-aligned outputs (**directly runnable logic**)
> * smaller models achieving **higher effective capacity**
> * **stable training** distributions with **low entropy**
> * **efficient multi-agent communication** via symbolic protocol
> * **verifiable, replayable reasoning and state transitions**

## pre run estimate table (WIP)

| Metric                              | Tiny Symbolic Model (Browser / Edge 0.5B–2B)         | Mid Symbolic Model (Local 3B–7B)                  | Large Symbolic Model (Server 8B–70B+)                | Notes / How Repo Enables This                                      |
|-------------------------------------|-----------------------------------------------------|--------------------------------------------------|------------------------------------------------------|--------------------------------------------------------------------|
| Primary deployment                  | Browser (WebGPU / WASM)                             | Local machine (llama.cpp / Ollama)               | Remote node / hosted inference                        | Same dataset + tokenizer scales across all sizes                   |
| Memory footprint (inference)        | ~200MB – 1.5GB                                      | ~3GB – 10GB                                      | 16GB – 80GB+                                          | Depends on quantization + model size                               |
| Latency                            | ~50–200 ms                                          | ~100–500 ms                                      | ~200–1500 ms                                          | Larger models slower but more capable                              |
| Capability scope                   | Basic intents, routing, UI planning                 | Multi-step reasoning, agent orchestration         | Complex reasoning, long-chain planning, analysis      | Bigger models = more generalization                                |
| Token efficiency (interlang)        | High                                                | Very high                                        | Extremely high                                        | All benefit from compressed symbolic sequences                     |
| Determinism                        | High                                                | Very high                                        | Very high                                             | Grammar + AST enforcement scales across sizes                      |
| Training cost                      | Low–medium                                          | Medium                                           | High                                                  | Same pipeline, just different base model + compute                 |
| Dataset requirement                | Small–medium                                        | Medium–large                                     | Large                                                 | All use canonical symbolic dataset from this repo                  |
| Use case in FRAME                  | Default built-in AI                                 | Power-user local AI dApp                          | Specialized AI dApp (analysis, coding, research)      | Developers choose model tier per dApp                              |
| dApp integration model             | Runs inside FRAME directly                          | Runs via local model bridge                       | Runs via remote/edge node with capability gating      | All accessed via intents (ai.run, etc.)                            |
| Multi-agent coordination           | Limited                                             | Strong                                           | Very strong                                           | Larger models better at planning/decomposition                     |
| Why use this repo                  | Train ultra-efficient small model                   | Train balanced local model                        | Distill large high-capability model into symbolic form| Same distillation pipeline scales vertically across model sizes    |


```frame-symbolic-model is a deterministic symbolic distillation system that replaces natural language generation with canonical executable program synthesis by transforming high entropy teacher model outputs into a constrained, low entropy intermediate representation (interlang) and training compact student models to directly emit these programs under a co-designed tokenizer and grammar, eliminating linguistic variance and ambiguity while preserving full execution semantics; instead of optimizing for textual similarity, the system enforces correctness through deterministic AST reconstruction and execution equivalence, ensuring that model outputs are not approximations but exact, replayable programs that map directly into FRAME’s intent routing and capability execution layer, dramatically increasing information density per token, reducing sequence length, improving convergence stability, and enabling smaller models to match or exceed the effective capability of larger general-purpose models within the domain, as all reasoning is encoded as compositional operations and arguments rather than prose, allowing the model to function as a compiler from intent to execution where compression mechanisms such as predicate mapping, argument minimization, reference reuse, and pattern factoring further collapse redundant structure over time, producing a stable training distribution with minimal entropy and highly predictable decoding behavior, ultimately shifting the paradigm from probabilistic language modeling to deterministic program generation where intelligence is expressed as executable structure rather than interpreted text, resulting in faster inference, lower compute cost, stronger guarantees of correctness, and a fully verifiable, replayable reasoning pipeline aligned with sovereign, local first AI systems.```
