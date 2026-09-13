*This project has been created as part of the 42 curriculum by oumahi.*

# call_me_maybe

## Description

`call_me_maybe` is a constrained-decoding function-calling prototype built on top of a local Large Language Model.

The goal of the project is to convert a natural-language request into a structured function call while ensuring that the generated output follows a strict JSON format.

For example, given:

```text
What is the sum of 2 and 3?
```

the program can generate:

```json
{
  "fn_name": "fn_add_numbers",
  "arguments": {
    "a": 2,
    "b": 3
  }
}
```

Instead of allowing the language model to freely generate any token, the project uses a **Finite-State Machine (FSM)** to control the decoding process.

At every generation step:

1. The language model produces logits for the complete vocabulary.
2. The FSM determines which tokens are valid in the current state.
3. Invalid tokens are masked.
4. The best valid token is selected.
5. The FSM moves to the next state.
6. The process continues until the complete JSON structure is generated.

The system therefore combines two components:

* The **LLM**, which provides semantic predictions.
* The **FSM**, which guarantees structural constraints.

The current implementation supports function arguments of type:

* `number`
* `string`

The project uses the local model:

```text
Qwen/Qwen3-0.6B
```

This model was chosen because it is small enough to run locally while still being capable of handling basic function-selection and argument-generation tasks.

---

## Instructions

### Requirements

The project requires:

* Python 3.10 or newer
* Internet access for the first model download, unless the model is already cached
* Enough memory to run the local Qwen model

The main dependencies are:

```text
torch
transformers
huggingface-hub
```

They are defined in `pyproject.toml`.

### Installation

Install the required dependencies using `pip`:

```bash
pip install torch transformers huggingface-hub
```

Alternatively, use the provided Makefile:

```bash
make install
```

### Execution

Run the program with the default input and output files:

```bash
python3 -m src.main
```

The default input file is:

```text
data/input/function_calling_tests.json
```

The generated results are written to:

```text
data/output/function_calling_results.json
```

Custom input and output files can also be provided:

```bash
python3 -m src.main \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

---

## Algorithm Explanation

The project implements **constrained decoding**.

A normal autoregressive language model predicts the next token from its complete vocabulary.

Conceptually:

```text
Input prompt
    ↓
Tokenizer
    ↓
Language model
    ↓
Logits for every vocabulary token
    ↓
Token selection
    ↓
Next token
```

Without constraints, the model could generate arbitrary content such as:

```text
Sure! You should call fn_add_numbers with 2 and 3.
```

Even if the answer is understandable, it does not respect the required JSON structure.

In this project, a constraint layer is inserted before token selection:

```text
Input prompt
    ↓
Tokenizer
    ↓
Language model
    ↓
Logits
    ↓
Finite-State Machine
    ↓
Determine valid tokens
    ↓
Mask invalid tokens
    ↓
Select the best valid token
    ↓
Update FSM state
    ↓
Repeat
```

Invalid tokens receive an effectively impossible score, conceptually equivalent to:

```text
-infinity
```

The decoder therefore chooses only between tokens that are valid for the current FSM state.

### Finite-State Machine

The FSM is implemented in:

```text
src/finite_state_machine.py
```

It controls the JSON generation step by step.

The main states are:

```text
START
  ↓
FN_VALUE
  ↓
ARG_STRUCT_START
  ↓
ARG_KEY
  ↓
ARG_VALUE
  ↓
ARG_AFTER_VALUE
  ↓
ARG_KEY or END_STRUCT
  ↓
END_STRUCT
```

### START

The `START` state begins the JSON object.

It writes the prefix:

```json
{
  "fn_name": "
```

The next state is:

```text
FN_VALUE
```

### FN_VALUE

This state generates the function name.

The decoder does not allow arbitrary function names.

Only names defined in the function-definition file are valid.

For example:

```json
[
  {
    "name": "fn_add_numbers",
    "parameters": {
      "a": {
        "type": "number"
      },
      "b": {
        "type": "number"
      }
    }
  }
]
```

If `fn_add_numbers` is one of the available functions, the FSM only allows tokens that can continue toward that valid function name.

Because tokenizers may divide a function name into several tokens, the decoder verifies valid prefixes during generation.

Once the complete function name is generated, the FSM moves to:

```text
ARG_STRUCT_START
```

### ARG_STRUCT_START

This state opens the arguments object.

It generates:

```json
",
"arguments": {
```

At this point, the selected function is already known.

The FSM can therefore retrieve its expected parameters and their types.

The next state is:

```text
ARG_KEY
```

### ARG_KEY

This state generates the next parameter name.

The parameter key must come directly from the selected function schema.

For example, for:

```json
{
  "name": "fn_add_numbers",
  "parameters": {
    "a": {
      "type": "number"
    },
    "b": {
      "type": "number"
    }
  }
}
```

only:

```text
a
b
```

are valid parameter names.

The model cannot invent keys such as:

```text
x
first_number
value
```

unless they are explicitly declared in the schema.

After generating the expected argument key, the FSM moves to:

```text
ARG_VALUE
```

### ARG_VALUE for Numbers

When the current parameter has type:

```text
number
```

the decoder enters numeric-generation mode.

The allowed characters are restricted mainly to:

```text
0 1 2 3 4 5 6 7 8 9
-
.
,
}
```

depending on the current position.

The number-generation logic tracks information such as:

* Number of generated digits
* Decimal-point usage
* Digits after the decimal point
* Maximum allowed digits
* Whether the value can terminate
* Whether more parameters remain

This prevents malformed values such as:

```text
12..4
4.2.8
000000003
```

Generation is also limited to avoid an infinite sequence of digits.

Trailing decimal zeros can be removed when appropriate.

For example:

```text
16.0000
```

can be normalized to:

```text
16
```

When the number is complete, the FSM either moves to the next argument or starts closing the JSON structure.

### ARG_VALUE for Strings

When the current parameter type is:

```text
string
```

the FSM first generates an opening quotation mark:

```text
"
```

It then restricts generation to a filtered set of safe and readable tokens.

The closing quotation mark is always allowed when the string may legally terminate.

Several limits are used to prevent uncontrolled generation:

* Maximum number of decoding steps
* Maximum string length
* Restricted character set
* Additional restrictions for regular-expression parameters

If the model reaches the configured limit without closing the string, the FSM forces the string to terminate.

Example:

```json
"hello world"
```

### ARG_AFTER_VALUE

After generating a value, the FSM checks whether additional parameters remain.

If another parameter exists:

```text
emit comma
   ↓
ARG_KEY
```

Otherwise:

```text
END_STRUCT
```

### END_STRUCT

The final state closes the arguments object and the complete function-call object.

The generated result has the form:

```json
{
  "fn_name": "fn_add_numbers",
  "arguments": {
    "a": 2,
    "b": 3
  }
}
```

Generation stops after the final JSON object has been completed.

---

## Design Decisions

### Local Language Model

The project uses:

```text
Qwen/Qwen3-0.6B
```

A relatively small model was selected because the goal of the project is to study constrained decoding rather than maximize raw language-model performance.

Using a small local model provides several advantages:

* Lower hardware requirements
* Faster testing
* No API dependency
* Offline execution after the model has been downloaded
* Reproducible experiments

### Finite-State Machine

An explicit FSM was chosen instead of relying only on prompting.

Prompting the model with instructions such as:

```text
Return only valid JSON.
```

does not guarantee that the output will actually respect the expected structure.

With an FSM, invalid structural choices are removed directly during generation.

### Token-Level Constraints

Constraints are applied during decoding instead of generating free-form text and repairing it afterward.

This means invalid outputs are prevented before they are produced.

The approach is conceptually:

```text
prevent invalid generation
```

rather than:

```text
generate → detect error → repair
```

### External Function Definitions

Function definitions are stored separately in JSON files.

This makes the system easier to extend and test because the decoding logic is not tightly coupled to specific functions.

### Limited Type Support

The current implementation supports:

```text
number
string
```

The scope was intentionally limited to make the decoding rules stable and understandable before adding more complex JSON types such as:

* Arrays
* Booleans
* Nested objects
* Null values
* Enums

### Greedy Selection

Among the currently valid tokens, the decoder chooses the token with the best model score.

This keeps the implementation relatively simple and deterministic.

---

## Performance Analysis

### Accuracy

The FSM strongly improves structural accuracy.

It prevents many invalid outputs such as:

* Unknown argument keys
* Arbitrary function names
* Missing JSON delimiters
* Unexpected natural-language text
* Invalid number structures

However, constrained decoding does not guarantee complete semantic correctness.

For example, the generated JSON may be structurally valid while the model selects the wrong function for an ambiguous prompt.

Similarly, a generated number may respect JSON syntax while still being semantically incorrect.

Therefore, two different forms of accuracy must be considered:

```text
Structural accuracy
Semantic accuracy
```

Structural accuracy is mainly controlled by the FSM.

Semantic accuracy still depends on the language model.

### Speed

Constrained decoding introduces additional work for every generated token.

At each decoding step, the system must:

1. Run the model.
2. Obtain the logits.
3. Determine valid token IDs.
4. Mask all invalid tokens.
5. Select the best valid token.
6. Update the FSM.
7. Repeat.

This is slower than completely unconstrained greedy generation.

However, because the selected model is small and the generated outputs are short, the performance remains acceptable for this project.

### Reliability

The solution is reliable for enforcing the supported output schema.

The FSM provides predictable control over:

* JSON syntax
* Function names
* Parameter names
* Number formatting
* String termination

Reliability still depends on several implementation details, especially tokenizer behavior.

Different models or tokenizers may split JSON syntax, function names, numbers, or strings differently.

For this reason, changing the model requires re-testing the decoding logic.

---

## Challenges Faced

### Tokenizer Behavior

One of the main difficulties was that an LLM tokenizer does not necessarily operate character by character.

For example:

```text
fn_add_numbers
```

may be represented by several tokens instead of one.

Some tokens may also contain:

* Leading spaces
* Multiple characters
* Punctuation
* Complete substrings

This makes token-level filtering more difficult than simple character-level validation.

The solution was to inspect decoded token content and only keep tokens that are compatible with the current FSM state.

### Function Name Constraints

Function names can be composed of multiple tokens.

The decoder therefore cannot simply allow one complete token for each function name.

It has to ensure that the generated sequence remains a valid prefix of at least one known function name until the full name is completed.

### Number Generation

Number generation required additional state tracking.

Without restrictions, the model could generate invalid sequences such as:

```text
12..8
0000005
4.5.6
```

Additional rules were introduced to track:

* Decimal points
* Digits
* Number length
* Termination
* Trailing zeros

Maximum-step limits were also added to prevent infinite numeric generation.

### String Generation

Strings are more difficult to constrain because they can contain a large variety of characters.

A safe token set was introduced to avoid invalid or uncontrolled sequences.

Maximum length limits were also added.

For regex-related values, stricter rules were necessary because regular expressions can contain many special characters and can easily result in long generation sequences.

### Preventing Infinite Generation

The model may repeatedly select tokens without naturally terminating a value.

Maximum token, digit, and character limits were therefore introduced.

When required, the FSM forces the value to terminate and moves to the next state.

### Schema Enforcement

The implementation had to ensure that argument names correspond exactly to the selected function.

The solution was to load the function schema after function selection and use it directly to determine the next expected argument.

---

## Testing Strategy

The implementation is tested using predefined prompts stored in:

```text
data/input/function_calling_tests.json
```

Function schemas are stored separately in:

```text
data/input/functions_definition.json
```

The output is generated in:

```text
data/output/function_calling_results.json
```

### Batch Testing

The complete test suite can be executed with:

```bash
python3 -m src.main
```

The generated results are then checked for:

* Valid JSON structure
* Valid function names
* Correct parameter names
* Correct parameter types
* Correct argument count
* Correct nesting
* Correct termination

### Number Tests

Numeric tests include values such as:

```text
0
5
42
-10
3.14
16.0000
```

They are used to verify:

* Negative values
* Decimal values
* Leading-zero behavior
* Decimal-point restrictions
* Maximum digit limits
* Correct termination

### String Tests

String tests verify:

* Opening quotation marks
* Closing quotation marks
* Safe-character filtering
* Maximum string lengths
* Forced termination
* Regex-specific restrictions

### Function Selection Tests

Prompts corresponding to different available functions are used to check whether the model selects the expected function.

### Regression Testing

Tests are re-run after modifying:

* The FSM
* Token filtering
* Number rules
* String rules
* The tokenizer
* The language model
* Function schemas

This helps verify that a change in one generation state does not break another part of the decoder.

---

## Example Usage

### Function Definition

Example content of:

```text
data/input/functions_definition.json
```

```json
[
  {
    "name": "fn_add_numbers",
    "parameters": {
      "a": {
        "type": "number"
      },
      "b": {
        "type": "number"
      }
    }
  }
]
```

### Input

Example content of:

```text
data/input/function_calling_tests.json
```

```json
[
  {
    "prompt": "What is the sum of 2 and 3?"
  }
]
```

### Running the Program

```bash
python3 -m src.main
```

Or explicitly:

```bash
python3 -m src.main \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

### Generated Result

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

Another example could be a function taking a string:

```json
{
  "name": "fn_search_text",
  "parameters": {
    "query": {
      "type": "string"
    }
  }
}
```

Input:

```json
[
  {
    "prompt": "Search for hello world"
  }
]
```

Possible generated function call:

```json
{
  "fn_name": "fn_search_text",
  "arguments": {
    "query": "hello world"
  }
}
```

---

## Resources

The following resources were used to understand transformers, tokenization, language-model generation, and constrained decoding.

### Transformers and Language Models

* The Illustrated Transformer
  https://jalammar.github.io/illustrated-transformer/

* The Illustrated BERT
  https://jalammar.github.io/illustrated-bert/

* The Illustrated GPT-2
  https://jalammar.github.io/illustrated-gpt2/

* Interactive LLM visualization
  https://bbycroft.net/llm

* Attention Is All You Need
  https://arxiv.org/pdf/1706.03762v7

### Constrained Decoding

* Constrained Decoding for Language Models
  https://www.aidancooper.co.uk/constrained-decoding/

* Coalescence and structured generation
  https://blog.dottxt.ai/coalescence.html?ref=aidancooper.co.uk#orgefede7b

* Research paper related to constrained generation
  https://arxiv.org/pdf/2505.09388

### Video Resources

* https://www.youtube.com/watch?v=xpvFinvqRCA&t=536s

* https://www.youtube.com/watch?v=6JGzwI2pNfo

* https://www.youtube.com/watch?v=KJtZARuO3JY

* https://www.youtube.com/watch?v=zduSFxRajkE

* https://www.youtube.com/playlist?list=PLUfbC589u-FSwnqsvTHXVcgmLg8UnbIy3

### AI Usage

AI tools were used as support during the project.

They were mainly used for:

* Improving the wording and organization of the README
* Clarifying documentation
* Explaining theoretical concepts such as transformers, tokenization, logits, and constrained decoding
* Reviewing explanations to make them easier to understand
* Discussing possible edge cases and testing ideas

AI was not used as a replacement for understanding the implementation.

The constrained-decoding logic, finite-state-machine behavior, integration decisions, debugging, testing, and final implementation were reviewed, understood, and adapted by the project author.
