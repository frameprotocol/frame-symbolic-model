# frame-symbolic-model

> Training pipeline for a compact model that maps natural language intents to deterministic interlang programs. The system enforces a constrained grammar, canonical serialization, and parse time validation to produce low entropy, execution ready outputs aligned with FRAME’s intent execution layer.

---

> [!IMPORTANT]  
> * deterministic program generation instead of natural language  
> * reduced token count and lower inference cost  
> * faster inference from short, structured outputs  
> * minimized ambiguity and hallucination surface  
> * strict AST-level validation and canonicalization  
> * outputs map directly to executable logic  
> * higher effective capacity from smaller models  
> * stable training due to constrained low-entropy format  
> * efficient agent-to-agent communication via symbolic protocol  
> * verifiable and replayable execution through deterministic outputs  
## Core concept

Training uses paired inputs of natural language and interlang programs. Each program is a linear chain of operations. Accept outputs only after parsing, validation, and canonicalization.

`frame-symbolic-model is a deterministic symbolic distillation system that replaces natural language generation with canonical program synthesis. It transforms high entropy teacher outputs into a constrained, low entropy intermediate representation (interlang) and trains compact models to emit these programs directly using a co-designed tokenizer and grammar. This removes linguistic variance and ambiguity while preserving execution semantics. Correctness is enforced through deterministic parsing, AST reconstruction, and canonicalization. Outputs are validated programs, not approximations, and can be replayed exactly. Each program maps directly into FRAME’s intent routing and capability execution layer. The system increases information density per token, reduces sequence length, and stabilizes training. Smaller models achieve high effective capability within the domain because reasoning is encoded as structured operations and arguments rather than prose. The model functions as a compiler from intent to execution. Compression mechanisms such as predicate mapping, argument minimization, reference reuse, and pattern factoring reduce redundancy and maintain a low entropy training distribution with predictable decoding behavior. This shifts the system from probabilistic language modeling to deterministic program generation. The result is faster inference, lower compute cost, strict correctness guarantees, and a fully verifiable, replayable execution pipeline aligned with local first, sovereign AI systems.`

## Architecture

| Component | Role |
|-----------|------|
| `data/canonical.jsonl`, `data/generated.jsonl` | Training data |
| `tokenizer/corpus.py` | JSONL ingest; canonical programs only |
| `interlang/parser.py` | Parse to `{op, args}` lists |
| `pipeline/canonicalize.py` | Canonicalization |
| `pipeline/validate.py` | Parse plus character and key/op checks |
| `training/dataset.py` | Records; optional primary-op rebalance |
| `tokenizer/train_tokenizer.py` | BPE on programs → `tokenizer/tokenizer.json` |
| `tokenizer/symbolic_pre.py` | BPE corpus prep (`TOK_JOIN`) |
| `training/train_lora.py` | LoRA; base `AutoTokenizer` + `<INPUT>` / `<OUTPUT>` |
| `training/infer.py` | LoRA inference; greedy decode by default |

Pipeline: load JSONL → optional primary-op rebalance → LoRA training → generate after `<OUTPUT>` → clean → validate.



## Interlang specification

- Programs start with `.` and contain one or more segments.
- Separate segments with `;` or `->` outside double-quoted strings.
- Each segment: `op` then zero or more `:key=value` arguments.
- `op` matches `[a-zA-Z0-9_.]+`.
- `key` matches `[a-zA-Z_][a-zA-Z0-9_]*`.
- Values: double-quoted strings (supports escapes) or unquoted tokens.
- `canonicalize` emits quoted values, sorts keys per op, drops consecutive duplicate `(op, args)`.
- It joins segments with `" ; "`; output prefixes `.` with trailing space.

## Data and training

**Schema (JSONL):** `input` (string). `program` or `output` (string); empty program fields skip the row.

Invalid or non-canonical rows are discarded. `DEDUPE_BY_PROGRAM_HASH` `False` collapses duplicate `(input, output)` pairs. `True` keeps at most one row per program hash (`training/dataset.py`).

`REBALANCE_DATASET` `True` runs `rebalance_by_primary_op`. It round-robins by first parsed op. Greedy retention caps each primary op at 40% of kept rows (`_MAX_PRIMARY_OP_SHARE`).

Supervision: `<INPUT>\n{input}\n<OUTPUT>\n`, program, EOS; `training/train_lora.py` masks labels on the prompt.

```bash
python training/train_lora.py --max-steps 1000
```

`python tokenizer/train_tokenizer.py --vocab-size 1024` builds `tokenizer/tokenizer.json`. `train_lora.py` uses the base `AutoTokenizer` only.

`training/config.py` sets defaults.

## Inference behavior

`training/infer.py` uses greedy generation (`--temperature` 0) by default. Non-zero `--temperature` enables sampling. Cleans generated span (`sanitize_text`), prints `RAW`, `PARSED`, and sets `STATUS` via `canonicalize` and `validate`.

Supply `adapter_model.safetensors` or `adapter_model.bin` under `models/symbolic-lora/`.

## Constraints and guarantees

| Guarantee | Scope |
|-----------|--------|
| Parser and `canonicalize` are deterministic for a fixed input string. | `interlang/parser.py`, `pipeline/canonicalize.py`. |
| `validate` accepts only strings that parse and match allowed character and key/op patterns. | `pipeline/validate.py`. |
| Training retains only rows that pass canonicalization and validation. `ensure_dot_prefix` forces a leading `.` on targets. | `training/dataset.py`. |
| Greedy generation at `temperature` 0 reproduces identical output for fixed weights and tokenizer. | `training/infer.py`. |

Model outputs may be invalid. Validate every output.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

```bash
python scripts/generate_dataset.py -n 20000
python training/train_lora.py --max-steps 1000
```

```bash
python tokenizer/train_tokenizer.py --vocab-size 1024
```

```bash
python training/infer.py "get current time"
```
