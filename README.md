*This project has been created as part of the 42 curriculum by berrabia*

# call_me_maybe

## Description
This project implements a constrained-decoding function-calling prototype on top of a local Hugging Face causal LLM. It takes natural-language prompts and produces a strict JSON function call with a selected function name and typed arguments. The goal is to reliably generate a machine-readable function invocation without post-hoc parsing hacks.

## Instructions

### Requirements
- Python >= 3.10
- Internet access for the first model download (unless already cached)

Dependencies (from pyproject.toml):
- torch
- transformers
- huggingface-hub

### Installation

Option 1: pip

```bash
pip install torch transformers huggingface-hub
```

Option 2: make target

```bash
make install
```

### Execution

Run with default input/output paths:

```bash
python3 -m src.main
```

Run with explicit files:

```bash
python3 -m src.main \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

Alternative entrypoint:

```bash
python3 -m src
```

## Algorithm Explanation (Constrained Decoding)
The model does not get to write anything it wants. Instead, it follows a step-by-step map (a finite-state machine) that describes the exact JSON shape of a function call. At each step, the FSM says which tokens are allowed next. The decoder then hides every other token and picks from the allowed ones only. The FSM moves forward based on what was emitted, walking through steps like opening `{`, writing a key, adding `:`, writing a value, and finally closing `}`.

Function names can only come from the known list, and argument keys must match the selected function schema. Argument values are also guided by type: numbers are built from digits (and optional `-` or `.` in the right place), while strings are limited to safe characters plus the closing quote. This keeps the output on track so it stays valid JSON and matches the function definition.

## Design Decisions
- Use a lightweight local model (Qwen/Qwen3-0.6B) to keep inference fast and offline-friendly.
- Encode the output format as a FSM to enforce a strict JSON grammar at token level.
- Separate function definitions and test prompts into JSON files for repeatable batch evaluation.
- Keep the schema minimal (number, string) to simplify reliable constrained decoding.

## Performance Analysis
- Accuracy: High when prompts align with available functions; constrained decoding prevents malformed JSON and invalid keys.
- Speed: Token masking adds overhead per step but remains acceptable for small models and short outputs.
- Reliability: The FSM prevents format drift and truncation errors, but relies on tokenizer-specific token boundaries and can be sensitive to model vocabulary.

## Challenges Faced
- Tokenizer alignment: Some tokens do not map cleanly to JSON fragments, requiring careful masking choices.
- Numeric and string handling: Avoiding infinite loops and malformed numbers required explicit state transitions.
- Schema enforcement: Ensuring keys and values match the selected function schema demanded strict state gating.

## Testing Strategy
- Run the batch test file in data/input/function_calling_tests.json and verify outputs match the expected schema.
- Inspect JSON validity and check that argument types conform to the function definitions.
- Re-run after any tokenizer or model change to validate FSM transitions.

## Example Usage

Input (data/input/function_calling_tests.json):

```json
[
  { "prompt": "What is the sum of 2 and 3?" },
  { "prompt": "Greet shrek" }
]
```

Function definitions (data/input/functions_definition.json):

```json
[
  {
    "name": "fn_add_numbers",
    "parameters": {
      "a": { "type": "number" },
      "b": { "type": "number" }
    }
  }
]
```

Run:

```bash
python3 -m src.main \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

Output (data/output/function_calling_results.json):

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "fn_name": "fn_add_numbers",
    "args": {
      "a": 2,
      "b": 3
    }
  }
]
```

## Resources
- Hugging Face Transformers documentation: https://huggingface.co/docs/transformers
- Hugging Face Hub documentation: https://huggingface.co/docs/huggingface_hub
- Constrained decoding overview (conceptual): https://arxiv.org/abs/2304.09118
- JSON specification: https://www.json.org/json-en.html

### AI Usage
AI was used to refine README wording and structure only. No AI-generated code was introduced into the implementation(lkdob 7ram ana ktbt chi 60% o howa 40% hh).
