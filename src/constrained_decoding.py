import json
from typing import Any, Dict, List, Optional, Set

from llm_sdk import Small_LLM_Model
from src.models.functions_definition import (
    FunctionDefinition,
    Parameter,
)


def extract_complete_json(text: str) -> Optional[str]:
    """
    Extract the first complete JSON object from generated text.

    Braces inside JSON strings are ignored.
    """
    start = text.find("{")

    if start == -1:
        return None

    brace_count = 0
    in_string = False
    escaped = False

    for index in range(start, len(text)):
        char = text[index]

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            brace_count += 1
        elif char == "}":
            brace_count -= 1

            if brace_count == 0:
                return text[start:index + 1]

    return None


def get_best_valid_token(
    logits: List[float],
    valid_ids: Set[int],
) -> int:
    """
    Return the allowed token ID with the highest logit.
    """
    if not valid_ids:
        raise RuntimeError("No valid token IDs available.")

    return max(
        valid_ids,
        key=lambda token_id: logits[token_id],
    )


def build_json_valid_ids(
    vocab: Dict[str, int],
) -> Set[int]:
    """
    Build a set of token IDs containing JSON-safe characters.
    """
    json_safe = set(
        'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        '0123456789*_.,:-+/\'!?()[]{}"\\ĠĊ '
    )

    valid_ids: Set[int] = set()

    for token_string, token_id in vocab.items():
        if token_string and all(
            char in json_safe
            for char in token_string
        ):
            valid_ids.add(token_id)

    return valid_ids


def get_allowed_values(
    parameter: Parameter,
) -> Optional[List[Any]]:
    """
    Return optional allowed values stored in a parameter definition.
    """
    for field_name in (
        "enum",
        "allowed_values",
        "choices",
    ):
        value = getattr(
            parameter,
            field_name,
            None,
        )

        if isinstance(value, list):
            return value

    return None


def validate_parameter_type(
    value: Any,
    parameter: Parameter,
) -> bool:
    """
    Validate a value against the parameter type and allowed values.
    """
    param_type = parameter.type.lower()

    if param_type == "number":
        valid_type = (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
        )

    elif param_type == "integer":
        valid_type = (
            isinstance(value, int)
            and not isinstance(value, bool)
        )

    elif param_type == "string":
        valid_type = isinstance(value, str)

    elif param_type == "boolean":
        valid_type = isinstance(value, bool)

    elif param_type == "array":
        valid_type = isinstance(value, list)

    elif param_type == "object":
        valid_type = isinstance(value, dict)

    elif param_type == "null":
        valid_type = value is None

    else:
        return False

    if not valid_type:
        return False

    allowed = get_allowed_values(parameter)

    if allowed is not None and value not in allowed:
        return False

    return True


def validate_function_call(
    data: Any,
    functions: List[FunctionDefinition],
) -> bool:
    """
    Validate generated JSON against the selected function definition.
    """
    if not isinstance(data, dict):
        return False

    if set(data.keys()) != {
        "name",
        "parameters",
    }:
        return False

    name = data.get("name")
    parameters = data.get("parameters")

    if not isinstance(name, str):
        return False

    if not isinstance(parameters, dict):
        return False

    if name == "none":
        return parameters == {}

    selected_function: Optional[FunctionDefinition] = None

    for function in functions:
        if function.name == name:
            selected_function = function
            break

    if selected_function is None:
        return False

    expected_names = set(
        selected_function.parameters.keys()
    )
    received_names = set(parameters.keys())

    if expected_names != received_names:
        return False

    for parameter_name, definition in (
        selected_function.parameters.items()
    ):
        value = parameters[parameter_name]

        if not validate_parameter_type(
            value,
            definition,
        ):
            return False

    return True


def load_vocabulary(
    model: Small_LLM_Model,
) -> Dict[str, int]:
    """
    Load the tokenizer vocabulary using the public SDK method.
    """
    vocab_path = model.get_path_to_tokenizer_file()

    with open(
        vocab_path,
        "r",
        encoding="utf-8",
    ) as file:
        tokenizer_data = json.load(file)

    raw_vocab = tokenizer_data.get(
        "model",
        {},
    ).get(
        "vocab",
        {},
    )

    if not isinstance(raw_vocab, dict):
        raise RuntimeError(
            "Invalid tokenizer vocabulary format."
        )

    vocab: Dict[str, int] = {}

    for token, token_id in raw_vocab.items():
        if isinstance(token, str) and isinstance(
            token_id,
            int,
        ):
            vocab[token] = token_id

    return vocab


def build_system_prompt(
    functions: List[FunctionDefinition],
) -> str:
    """
    Build the system prompt containing available function definitions.
    """
    lines = [
        (
            "STRICT SYSTEM RULE: Use ONLY a matching "
            "function from the list below."
        ),
        (
            "If NO function matches the user's intent "
            '(even if types match), set name: "none".'
        ),
        (
            "Never use an unrelated function for a "
            "different task."
        ),
        "",
        "Available functions:",
    ]

    for function in functions:
        parameters = []

        for name, info in function.parameters.items():
            description = f"{name}: {info.type}"
            allowed = get_allowed_values(info)

            if allowed is not None:
                description += (
                    f" allowed={json.dumps(allowed)}"
                )

            parameters.append(description)

        params_text = ", ".join(parameters)

        lines.append(
            f"  - {function.name}({params_text}): "
            f"{function.description}"
        )

    lines.append(
        "Use exactly the parameter names, types, and "
        "allowed values defined for the selected function."
    )

    lines.append(
        "\nOutput ONLY valid JSON: "
        '{"name": "<fn>", '
        '"parameters": {<parameters>}}'
    )

    return "\n".join(lines)