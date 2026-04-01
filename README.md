*This project has been created as part of the 42 curriculum by berrabia*

# call_me_maybe

## Description
This project implements a constrained-decoding function-calling prototype on top of an LLM. It takes natural-language prompts and produces a strict JSON function call with a selected function name and typed arguments.

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


## Algorithm Explanation (Constrained Decoding)
The model does not get to write anything it wants. Instead, it follows a step-by-step map (a finite-state machine) that describes the exact JSON shape of a function call. At each step, the FSM says which tokens are allowed next. The decoder then hides every other token and picks from the allowed ones only. The FSM moves forward based on what was emitted, walking through steps like opening `{`, writing a key, adding `:`, writing a value, and finally closing `}`.

Function names can only come from the known list, and argument keys must match the selected function schema. Argument values are also guided by type: numbers are built from digits (and optional `-` or `.` in the right place), while strings are limited to safe characters plus the closing quote. This keeps the output on track so it stays valid JSON and matches the function definition.

## Core Concepts (Short Definitions)
- Tokenizer: turns raw text into tokens (ids) that the model can process, and converts ids back to text.
- Encoder: transforms input tokens into internal representations (contextual embeddings).
- Decoder: generates output tokens step-by-step, conditioned on previous tokens (and encoder output in encoder-decoder models).
- Transformer: the neural architecture behind the model, using self-attention and feed-forward layers to model sequences efficiently.

## FSM Detailed 
The FSM lives in src/finite_state_machine.py and controls output token-by-token. It follows these states in order and loops across parameters as needed.

### State: START
Writes the JSON prefix for a function call.
Output: {
"fn_name": "
Next: FN_VALUE

### State: FN_VALUE
Chooses the function name from the allowed list. Tokens are restricted to substrings of known names. The state completes when a closing quote is emitted, and the selected function is detected by decoding the output.
Next: ARG_STRUCT_START

### State: ARG_STRUCT_START
Writes the arguments object prefix.
Output: ,
"arguments": {
Next: ARG_KEY

### State: ARG_KEY
Writes the next argument key for the selected function.
Output: "<param_name>": 
Next: ARG_VALUE

### State: ARG_VALUE (number)
Generates a JSON number with tight constraints.
- Allowed tokens: digits and a single '.' (decimal point), plus ',' or '}' depending on whether more params remain.
- Keeps counters for total digits and digits after the decimal.
- Prevents invalid formatting (for example, multiple leading zeros for the final param).
- Trims long runs of trailing zeros after a decimal (e.g., 16.0000 -> 16).
- Stops if max steps or max digits are reached, then advances to the next param or closes the JSON.
Next: ARG_KEY (more params) or END_STRUCT (last param)

### State: ARG_VALUE (string)
Generates a JSON string with restricted characters.
- Always emits an opening quote.
- Allows a filtered set of readable tokens and always allows the closing quote.
- For regex params, uses stricter character rules and shorter limits to avoid runaway patterns.
- Forces a closing quote if max steps or character limits are reached.
Next: ARG_AFTER_VALUE

### State: ARG_AFTER_VALUE
Moves to the next parameter or ends the argument object.
- If more params remain, emits a comma and goes back to ARG_KEY.
- If no params remain, goes to END_STRUCT.

### State: END_STRUCT
Closes the JSON object (and the nested arguments object if needed).
Output: \n}\n or \n}\n}\n

## Design Decisions
- Picked a small local model (Qwen/Qwen3-0.6B) so it runs fast and works offline.
- Used a finite-state machine so the output always follows valid JSON structure.
- Kept function definitions and test prompts in separate JSON files to make testing easy to repeat.
- Limited the schema to just number and string to keep decoding stable and predictable.

## Performance Analysis
- Accuracy: Strong when the prompt matches a known function; the FSM blocks bad JSON or wrong keys.
- Speed: Masking tokens adds work each step, but it is still fine for small models and short outputs.
- Reliability: The FSM keeps the structure correct, though it depends on tokenizer details and the model's vocab.

## Challenges Faced
- Numbers and strings: Preventing infinite loops and bad number formats required extra rules in the FSM.
- Schema enforcement: Keys and values had to be tightly restricted to match the selected function.

## Testing Strategy
- Run the batch tests in data/input/function_calling_tests.json and confirm the output shape is correct.
- Check that the JSON is valid and that argument types match the function definitions.
- Re-run tests after changing the model or tokenizer to make sure the FSM still behaves.

## Example Usage

Input (data/input/function_calling_tests.json):

```json
[
  { "prompt": "What is the sum of 2 and 3?" }
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
python3 -m src.main
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

https://jalammar.github.io/illustrated-bert/

https://bbycroft.net/llm

https://www.aidancooper.co.uk/constrained-decoding/

https://arxiv.org/pdf/1706.03762v7

https://arxiv.org/pdf/2505.09388

https://www.youtube.com/watch?v=xpvFinvqRCA&t=536s

https://www.youtube.com/watch?v=6JGzwI2pNfo

https://www.youtube.com/watch?v=KJtZARuO3JY

https://www.youtube.com/watch?v=zduSFxRajkE

https://jalammar.github.io/illustrated-transformer/

https://blog.dottxt.ai/coalescence.html?ref=aidancooper.co.uk#orgefede7b

https://www.youtube.com/playlist?list=PLUfbC589u-FSwnqsvTHXVcgmLg8UnbIy3


### AI Usage
AI was used to refine README wording and structure only. No AI-generated code was introduced into the implementation(lkdob 7ram ana ktbt chi 60% o howa 40% hh).
