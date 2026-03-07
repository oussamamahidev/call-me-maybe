import json
import sys
from typing import List, Any, Dict, Set
from llm_sdk import Small_LLM_Model


def mask_logits(logits: List, allowed_ids: Any) -> Any:
    for i in range(len(logits)):
        if i not in allowed_ids:
            logits[i] = -float("inf")
    return logits


def main_loop(
    model: Small_LLM_Model,
    input_ids: List[int],
    prompt: str
) -> List[int]:

    try:
        with open(model.get_path_to_vocab_file(), "r") as f:
            vocab = json.load(f)
    except IOError:
        print("error with get_path_to_vocab_file method")
        sys.exit()

    id_to_token = {v: k for k, v in vocab.items()}

    string_allowed_ids: List[int] = list()
    for token_str, token_id in vocab.items():
        if "\n" in token_str or "\t" in token_str:
            continue  # no control characters
        elif token_str in prompt:
            string_allowed_ids.append(token_id)

    string_allowed_ids.extend(input_ids)
    allowed_cars = [
        "*", "[", "]", "{", "}", "(", ")", "|", "a", "b", "c", "d",
        "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p",
        "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "A", "B",
        "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N",
        "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
    ]
    for car in allowed_cars:
        string_allowed_ids.extend(model.encode(car)[0].tolist())
    string_allowed_ids = list(set(string_allowed_ids))

    try:
        with open("data/input/functions_definition.json") as f:
            functions: Dict = json.load(f)
    except IOError:
        print("error with the data/input/functions_definition.json file")
        sys.exit()

    function_names: List[str] = [f["name"] for f in functions]
    function_names.append('"')

    state: str = "START"
    selected_function: Any = None
    param_keys: list = []
    param_index: int = 0
    arg_flage: bool = False
    current_param_type: Any = None
    llm_output: list[int] = []
    allowed_functionname_ids: Set = set()

    for token_str, token_id in vocab.items():
        for name in function_names:
            if token_str in name:
                allowed_functionname_ids.add(token_id)

    while True:

        logits = model.get_logits_from_input_ids(input_ids)
        allowed_ids = set()
        print(model.decode(llm_output))

        if state == "START":
            tokens = model.encode('{\n"fn_name": "')[0].tolist()
            input_ids.extend(tokens)
            llm_output.extend(tokens)
            state = "FN_VALUE"
            continue

        elif state == "FN_VALUE":

            logits = mask_logits(logits, allowed_functionname_ids)

            next_token = max(range(len(logits)), key=lambda i: logits[i])
            token_str = id_to_token[next_token]
            input_ids.append(next_token)
            llm_output.append(next_token)

            if token_str == '"':
                # detect selected function
                text = model.decode(llm_output)
                for f in functions:
                    if f["name"] in text:
                        selected_function = f  # dict
                        break
                # print(selected_function)
                param_keys = list(selected_function["parameters"].keys())
                state = "ARG_KEY"

            continue

        elif state == "ARG_KEY":
            if not arg_flage:
                tokens = model.encode(',\n"arguments": {\n')[0].tolist()
                input_ids.extend(tokens)
                llm_output.extend(tokens)
                arg_flage = True

            param: str = param_keys[param_index]
            current_param_type: str = selected_function["parameters"][param]["type"]

            tokens: List[int] = model.encode(f'"{param}": ')[0].tolist()
            input_ids.extend(tokens)
            llm_output.extend(tokens)

            state = "ARG_VALUE"
            continue

        elif state == "ARG_VALUE":

            if current_param_type == "number":
                seen_decimal: bool = False  # kyna . ola la
                max_digits: int = 8
                digit_count: int = 0
                digits_after_decimal: int = 0
                nonzero_after_decimal: bool = False
                decimal_index: Any = None

                # 16.00000 -> 16
                def normalize_fractional_tail() -> None:
                    nonlocal seen_decimal
                    nonlocal digits_after_decimal
                    nonlocal nonzero_after_decimal
                    nonlocal decimal_index
                    nonlocal digit_count

                    if not seen_decimal or decimal_index is None:
                        return

                    while len(input_ids) > decimal_index + 1:
                        last_token_str = id_to_token[input_ids[-1]]
                        if last_token_str != "0":
                            break
                        input_ids.pop()
                        llm_output.pop()
                        if digit_count > 0:
                            digit_count -= 1

                    if len(input_ids) == decimal_index + 1:
                        input_ids.pop()
                        llm_output.pop()
                        seen_decimal = False
                        decimal_index = None
                        digits_after_decimal = 0
                        nonzero_after_decimal = False
                        return

                    remaining_fraction = [
                        id_to_token[token_id]
                        for token_id in input_ids[decimal_index + 1:]
                    ]
                    digits_after_decimal = len(remaining_fraction)
                    nonzero_after_decimal = any(
                        token != "0" for token in remaining_fraction
                    )

                while True:

                    logits = model.get_logits_from_input_ids(input_ids)
                    allowed_ids = set()

                    allowed_digits: List[str] = [
                        ",", "}", "0", "1", "2", "3", "4",
                        "5", "6", "7", "8", "9"
                    ]
                    for car_id in allowed_digits:
                        allowed_ids.add(model.encode(car_id)[0].tolist()[0])

                    logits = mask_logits(logits, allowed_ids)
                    next_token = max(
                        range(len(logits)),
                        key=lambda i: logits[i]
                        )
                    token_str = id_to_token[next_token]

                    input_ids.append(next_token)
                    llm_output.append(next_token)
                    if token_str == "," and param_index == len(param_keys) - 1:
                        llm_output.pop()
                        input_ids.pop()
                        input_ids.extend(model.encode("}")[0].tolist())
                        llm_output.extend(model.encode("}")[0].tolist())
                        state = "END_STRUCT"
                        break

                    if token_str.isdigit():
                        digit_count += 1

                        if seen_decimal:
                            digits_after_decimal += 1
                            if token_str != "0":
                                nonzero_after_decimal = True

                        # stell have some params to add
                        if digit_count >= max_digits:
                            normalize_fractional_tail()
                            if param_index < len(param_keys) - 1:
                                tokens = model.encode(',\n')[0].tolist()
                                input_ids.extend(tokens)
                                llm_output.extend(tokens)
                                param_index += 1
                                state = "ARG_KEY"
                            else:
                                state = "END_STRUCT"
                            break
                        continue

                    if token_str == ".":
                        seen_decimal = True
                        decimal_index = len(input_ids) - 1
                        digits_after_decimal = 0
                        nonzero_after_decimal = False
                        continue

                    if token_str == ",":
                        if seen_decimal:
                            input_ids.pop()
                            llm_output.pop()
                            normalize_fractional_tail()
                            input_ids.append(next_token)
                            llm_output.append(next_token)
                        param_index += 1
                        state = "ARG_KEY"
                        break

                    if token_str == "}":
                        if seen_decimal:
                            input_ids.pop()
                            llm_output.pop()
                            normalize_fractional_tail()
                            input_ids.append(next_token)
                            llm_output.append(next_token)
                        state = "END_STRUCT"
                        break

                continue

            elif current_param_type == "string":
                # force opening quote
                tokens = model.encode('"')[0].tolist()
                input_ids.extend(tokens)
                llm_output.extend(tokens)

                while True:
                    logits = model.get_logits_from_input_ids(input_ids)
                    logits = mask_logits(logits, string_allowed_ids)
                    next_token: int = max(
                        range(len(logits)),
                        key=lambda i: logits[i]
                        )
                    token_str = id_to_token[next_token]
                    input_ids.append(next_token)
                    llm_output.append(next_token)

                    if token_str == '"':
                        break
                    if '"' in token_str:
                        llm_output.pop()
                        input_ids.pop()
                        llm_output.extend(model.encode('"')[0].tolist())
                        input_ids.extend(model.encode('"')[0].tolist())
                        break
                    # print(param_keys, token_str)
                    if param_keys[param_index] == "regex" and token_str == "]":
                        break

                state = "ARG_AFTER_VALUE"
                continue

        elif state == "ARG_AFTER_VALUE":

            param_index += 1

            if param_index < len(param_keys):
                tokens = model.encode(',\n')[0].tolist()
                input_ids.extend(tokens)
                llm_output.extend(tokens)
                state = "ARG_KEY"
            else:
                state = "END_STRUCT"

            continue

        elif state == "END_STRUCT":
            decoded = model.decode(llm_output).rstrip()
            closing_text = '\n}\n' if decoded.endswith('}') else '\n}\n}\n'
            tokens = model.encode(closing_text)[0].tolist()
            input_ids.extend(tokens)
            llm_output.extend(tokens)
            break

    return llm_output
