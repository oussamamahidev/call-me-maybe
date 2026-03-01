# call_me_maybe

Function-calling prototype built on top of a local Hugging Face causal LLM.

The project takes natural-language prompts and generates a strict JSON function call:

```json
{
	"fn_name": "<function_name>",
	"arguments": {
		"<arg_name>": <value>
	}
}
```

It uses a finite-state generation loop with logit masking to force output structure.

## Features

- Local inference via `Qwen/Qwen3-0.6B`
- Token-by-token constrained decoding (FSM)
- Typed arguments (`number`, `string`)
- Input/Output JSON files for batch evaluation

## Project Layout

```text
.
├── src/
│   ├── __main__.py
│   └── main.py                      # function-calling FSM + batch runner
├── llm_sdk/
│   └── __init__.py                  # Small_LLM_Model wrapper
├── data/
│   ├── input/
│   │   ├── function_calling_tests.json
│   │   └── functions_definition.json
│   └── output/
│       └── function_calling_results.json
├── pyproject.toml
├── Makefile
└── mypy.ini
```

## Requirements

- Python `>= 3.10`
- Internet access for first model download (unless already cached)

Dependencies (from `pyproject.toml`):

- `torch`
- `transformers`
- `huggingface-hub`

## Installation

### Option 1: pip

```bash
pip install torch transformers huggingface-hub
```

### Option 2: make target

```bash
make install
```

## Usage

Run with default input/output paths:

```bash
python3 -m src.main
```

Or with explicit files:

```bash
python3 -m src.main \
	--input data/input/function_calling_tests.json \
	--output data/output/function_calling_results.json
```

Alternative entrypoint:

```bash
python3 -m src
```

## Input Format

`data/input/function_calling_tests.json`

```json
[
	{ "prompt": "What is the sum of 2 and 3?" },
	{ "prompt": "Greet shrek" }
]
```

`data/input/functions_definition.json`

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

## Output Format

`data/output/function_calling_results.json`

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

## Development Helpers

- Run: `make run`
- Lint/type-check: `make lint`
- Debug: `make debug`
- Clean cache files: `make clean`

## Notes

- The generation logic is heuristic and strongly tied to tokenizer behavior.
- Regex/string and float handling are constrained to avoid infinite generation loops.
- For production-grade function calling, add robust post-processing/validation.
