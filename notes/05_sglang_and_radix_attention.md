# Technical Guide: SGLang, RadixAttention & Jump-Forward Decoding

---

## 1. What is SGLang and Why Does it Exist?

**SGLang (Structured Generation Language)** was created by LMSYS Org and UC Berkeley Sky Computing Lab (Lianmin Zheng, Hao Zhang, et al.). It is the engine powering LMSYS Chatbot Arena and the official production serving backend for **DeepSeek-R1 and DeepSeek-V3**.

Traditional LLM serving engines treat inference as isolated, stateless, single-turn string-in / string-out queries. In reality, modern AI engineering is dominated by **structured, stateful, branching LLM programs**:
1. Multi-turn autonomous agent loops (ReAct, tool calling, state accumulation).
2. Shared invariant context (e.g., 10k+ token OpenAPI tool declarations shared across thousands of requests).
3. Reasoning trees (Tree-of-Thought, Monte Carlo Tree Search, DeepSeek-R1 reasoning traces).
4. Strictly constrained outputs (JSON schemas, regex, typed grammars).

SGLang co-designs a high-level frontend programming language (**SGLang DSL**) with a high-performance serving runtime (**SRuntime**).

---

## 2. RadixAttention: Dynamic Radix Tree vs vLLM Hash Chains

### The Fundamental Problem
In autoregressive decoding, computing attention over past tokens requires storing Key and Value activations in GPU memory (the KV Cache). Recomputing KV activations for common prompt prefixes wastes compute and drives up **Time-To-First-Token (TTFT)**.

### vLLM Hash Chains (Block-Quantized Matching)
vLLM divides VRAM into fixed physical pages (e.g., 16 tokens) and computes cryptographic hash chains:
- $h_0 = \text{Hash}(\text{Tokens}[0:16])$
- $h_1 = \text{Hash}(h_0 \parallel \text{Tokens}[16:32])$

**Flaws of Hash Chains:**
* **Boundary Quantization Penalty:** Prefix sharing only matches at strict 16-token boundaries. If a system prompt or tool result deviates by even one token at position 15, cache matching fails.
* **Flat LRU Eviction:** Evicting a single intermediate physical block breaks all downstream hash chains.
* **No Native Tree Topology:** Branching requires duplicating metadata across flat block lists.

### SGLang RadixAttention: Dynamic Radix Tree (Compressed Trie)
SGLang maintains the full hierarchical relationship of all resident KV caches in a **Dynamic Radix Tree**:

```
                  [ Root Node (Empty) ]
                            │
               "You are an AI assistant..." (Tokens 0-511)
                            │
              ┌─────────────┴─────────────┐
        Turn 1 User A               Turn 1 User B
     "Search for AAPL..."        "Explain Quantum..."
       (Tokens 512-575)            (Tokens 512-590)
              │                           │
       Tool Exec Output                   ▼
      {"price": 182.50}              [Generate...]
       (Tokens 576-640)
        ┌─────┴─────┐
     Fork 1       Fork 2
   "Summarize"  "Plot chart"
```

1. **Arbitrary Token Granularity:** Uses **Longest Prefix Match (LPM)** to match cached tokens down to single-token granularity without block-boundary alignment constraints.
2. **Tree-Structured Eviction (Recursive LRU):** Eviction prunes only stale leaf nodes. Invariant ancestor trunks (system prompts, tool definitions) remain hot in GPU memory.
3. **In-Place Mutation:** When a generation diverges from an existing path, SRuntime splits the edge at the divergence index, creates an interior node, and attaches the new continuation as a child branch with zero memory copy.

---

## 3. Jump-Forward Decoding: Compressed Finite-State Machines

### The Flaw of Traditional Constrained Decoding
When forcing an LLM to generate valid JSON, traditional engines (Outlines, standard guidance) run token-by-token masked sampling:
- In JSON, large syntax sequences are **100% deterministic** (e.g., `{"status": "success", "data": {"items": [ ... }`).
- Running full autoregressive forward passes (which are memory-bandwidth bound at 1 token per pass) to generate predictable structural syntax wastes significant time and GPU cycles.

### SGLang's Path Compression
SGLang compiles regexes and JSON schemas into **Compressed FSMs**:

```
Traditional FSM (Token-by-Token Autoregressive Decoding):
State 0 ──['{']──> State 1 ──['\n']──> State 2 ──['  "']──> State 3 ──['id']──> State 4 ──['":']──> State 5
  (Pass 1)           (Pass 2)            (Pass 3)             (Pass 4)            (Pass 5)
  [5 Forward Passes for Deterministic Boilerplate]

SGLang Jump-Forward Decoding (Path Compression):
State 0 ──────────────────────['{\n  "id": ']──────────────────────> State 5
  (Single Jump-Forward Prefill Step via RadixAttention)
  [1 Forward Pass: Batch prefill entire deterministic slice into KV cache!]
```

* **Speedup:** Reduces decoding latency for complex structured JSON by **2x to 5x**, making schema-constrained generation faster than unconstrained generation.

---

## 4. Architectural Comparison: SGLang vs vLLM vs TensorRT-LLM

| Dimension | SGLang (SRuntime) | vLLM | TensorRT-LLM |
| :--- | :--- | :--- | :--- |
| **KV Cache Structure** | **Dynamic Radix Tree** (Arbitrary granularity) | **PagedAttention + Hash Chains** (16-token pages) | **Paged KV Cache** / Static buffers |
| **Prefix Caching** | **Optimal for trees, forks, and agents** | Flat hash-matching | Fixed prompt cache / KV retention |
| **JSON Constrained Decoding**| **Jump-Forward FSM** (Bypasses deterministic tokens) | Token-level logit masking (XGrammar) | Token-level logit masking |
| **Frontend Model** | **SGLang DSL** (`@sgl.function`, `fork`, `join`) | Standard OpenAI HTTP / Python Engine | C++ Executor API / Python Builder |
| **DeepSeek R1/V3 MLA** | **Native Day-0 optimized FlashInfer kernels** | Supported | Supported via custom plugins |
| **Ideal Workload** | **Agentic loops, reasoning trees, structured JSON** | General-purpose serving & high concurrency | Fixed massive-scale models |
