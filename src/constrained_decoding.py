from llm_sdk import Small_LLM_Model

import json



def extract_json_from_text(text: str):

    start = text.find("{")

    if start == -1:
        return None

    braces = 0

    for i in range(start , len(text)):
        if text[i] == "{":
            braces += 1
        if text[i] == "}":
            braces -= 1

        if braces == 0:
            return text[start:i+1]

    return None
def get_best_valid_ids(logits, valids):
    return max(
        valids,
        key=lambda i: logits[i] if i < len(logits) else float("-inf")
    )
def build_json_valid_ids(vocabulary:dict):
    """
    Build a set of valid token IDs for JSON characters based on the provided vocabulary.

    Args:
        vocabulary (dict): A dictionary mapping tokens to their corresponding IDs."""

    json_safe = set(
    'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    '0123456789'
    '*_,.:+-/\\\'!?()[]{}" '
    'ĠĊ'
)

    valid = set()

    for token,token_id in vocabulary.items():
        if token and all (c in json_safe for c in token):
            valid.add(token_id)

    return valid


def load_vocabulary(model: Small_LLM_Model) -> dict[str, int]:
    vocab_path = model.get_path_to_tokenizer_file()
    with open(vocab_path, 'r', encoding='utf-8') as f:
        tokenizer_data = json.load(f)
    vocab = tokenizer_data.get('model', {}).get('vocab', {})

    if not isinstance(vocab, dict) or not vocab:
        raise ValueError(f"Tokenizer vocabulary is missing or invalid: {vocab_path}")

    return vocab

def build_system_prompt(functions):
    """
    Build the system prompt for the model based on the provided functions.

    Args:
        functions (list): A list of function definitions."""

    lines = [
        "STRICT SYSTEM RULE: Use ONLY a matching function from the list below.",
        "If NO function matches the user's intent (even if types match), say so.",
        "Never use an unrelated function for a different task.",
        "",
        "Available functions:",
    ]

    for fn in functions:
        params = ", ".join(
            f"{name}: {info.type}"
            for name, info in fn.parameters.items()
        )
        lines.append(f"- {fn.name}({params}): {fn.description}")

    lines.append('\nOutput only valid JSON with the following structure: {"name": "<fn>", "args": {<args>}}')

    return "\n".join(lines)
