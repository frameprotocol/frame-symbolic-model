---
license: mit
language:
- en
tags:
- intent-classification
- structured-output
- gguf
- llama.cpp
pipeline_tag: text-generation
---

FRAME NL → Intent Compiler (English)

This model converts natural language into structured intent JSON:

{
  "intent": "string",
  "params": { "key": "value" }
}

Important:
- The model is NOT trusted for correctness
- Runtime MUST enforce substring validation
- Runtime MUST compute missing params

Example:

Input:
send bob hello

Output:
{
  "intent": "message.send",
  "params": { "to": "bob", "text": "hello" }
}

*Note: this is not perfect for a reason, frames runtime will fix it entirely over time*

---

## Run locally (llama.cpp)

**Requirements:** llama.cpp built (`llama-cli` binary available)

**Example:**

```bash
cd ~/frame-symbolic-model

./llama.cpp/build/bin/llama-cli \
  -m models/english/model.gguf \
  -p "send bob 5 dollars" \
  -n 100 \
  --temp 0.0
```

Expected output (approx):

```json
{"intent":"payment.send","params":{"to":"bob","text":"5"}}
```

**Run with validation (this repo):**

```bash
cd models/english
python infer.py "send bob 5 dollars"
```

Expected output:

```json
{"intent":"payment.send","params":{"to":"bob"}}
```

**Notes:**
- Output is strict JSON only
- Params not present in input are removed by validation

---

## How it works

```mermaid
flowchart LR
    A[Natural language input] --> B[GGUF model\nllama-cli]
    B --> C[Raw JSON output]
    C --> D[Validation\ninfer.py]
    D --> E[Cleaned intent JSON]
```
