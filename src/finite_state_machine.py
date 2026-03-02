import json
import sys
from typing import List, Dict, Any

def mask_logits(logits, allowed_ids):
    for i in range(len(logits)):
        if i not in allowed_ids:
            logits[i] = -float("inf")
    return logits


def main_loop(model, input_ids: list[int]) -> list[int]:

    # -------------------------------------------------
    # Load vocab
    # -------------------------------------------------
    try:
        with open(model.get_path_to_vocab_file(), "r") as f:
            vocab = json.load(f)
    except IOError:
        print("error with get_path_to_vocab_file method")
        sys.exit()

    id_to_token = {v: k for k, v in vocab.items()}

    # -------------------------------------------------
    # Load function definitions
    # -------------------------------------------------
    try:
        with open("data/input/functions_definition.json") as f:
            functions = json.load(f)
    except IOError:
        print("error with the data/input/functions_definition.json file")
        sys.exit()

    function_names = [f["name"] for f in functions]
    # -------------------------------------------------
    # FSM STATE
    # -------------------------------------------------
    state = "START"
    selected_function: Any = None
    param_keys: list = []
    param_index = 0
    current_param_name = None
    current_param_type = None

    llm_output: list[int] = []

    # -------------------------------------------------
    # GENERATION LOOP
    # -------------------------------------------------
    while True:

        print(model.decode(llm_output))
        logits = model.get_logits_from_input_ids(input_ids)
        allowed_ids = set()

        # =====================================================
        # STRUCTURAL STATES (we force structure)
        # =====================================================

        if state == "START":
            tokens = model.encode('{\n"fn_name": "')[0].tolist()
            input_ids.extend(tokens)
            llm_output.extend(tokens)
            state = "FN_VALUE"
            continue

        # =====================================================
        # FUNCTION NAME GENERATION
        # =====================================================
        elif state == "FN_VALUE":

            for token_str, token_id in vocab.items():

                # allow letters belonging to any function name
                for name in function_names:
                    if token_str in name:
                        allowed_ids.add(token_id)

                # allow closing quote AFTER something written
                if token_str == '"' and len(model.decode(llm_output)) > 15:
                    allowed_ids.add(token_id)

            logits = mask_logits(logits, allowed_ids)
            # print(logits)

            next_token = max(range(len(logits)), key=lambda i: logits[i])
            token_str = id_to_token[next_token]

            input_ids.append(next_token)
            llm_output.append(next_token)

            if token_str == '"':
                # detect selected function
                text = model.decode(llm_output)
                for f in functions:
                    if f["name"] in text:
                        selected_function = f
                        break

                param_keys = list(selected_function["parameters"].keys())
                state = "ARG_STRUCT_START"

            continue

        # =====================================================
        # FORCE ARGUMENT STRUCTURE
        # =====================================================
        elif state == "ARG_STRUCT_START":

            tokens = model.encode(',\n"arguments": {\n')[0].tolist()
            input_ids.extend(tokens)
            llm_output.extend(tokens)

            param_index = 0
            state = "ARG_KEY"
            continue

        # =====================================================
        # FORCE PARAM KEY
        # =====================================================
        elif state == "ARG_KEY":

            if param_index >= len(param_keys):
                state = "END_STRUCT"
                continue

            param = param_keys[param_index]
            current_param_name = param
            current_param_type = selected_function["parameters"][param]["type"]

            tokens = model.encode(f'"{param}": ')[0].tolist()
            input_ids.extend(tokens)
            llm_output.extend(tokens)

            state = "ARG_VALUE"
            continue

        # =====================================================
        # GENERATE PARAM VALUE
        # =====================================================
        elif state == "ARG_VALUE":

            # -----------------------
            # NUMBER TYPE
            # -----------------------
            if current_param_type == "number":
                # Let the model emit digits (and at most one '.') until a comma (next param)
                # or a closing brace (last param). Caps protect against runaway zeros.
                seen_decimal = False
                max_steps = 16
                max_digits = 12
                steps = 0
                digit_count = 0
                digits_after_decimal = 0
                nonzero_after_decimal = False
                decimal_index: Any = None

                while True:
                    steps += 1

                    logits = model.get_logits_from_input_ids(input_ids)
                    allowed_ids = set()
                    for token_str, token_id in vocab.items():
                        if token_str.isdigit():
                            allowed_ids.add(token_id)
                        elif token_str == "." and not seen_decimal:
                            allowed_ids.add(token_id)

                    if param_index < len(param_keys) - 1:
                        for token_str, token_id in vocab.items():
                            if token_str == ",":
                                allowed_ids.add(token_id)
                    else:
                        for token_str, token_id in vocab.items():
                            if token_str == "}":
                                allowed_ids.add(token_id)

                    logits = mask_logits(logits, allowed_ids)
                    next_token = max(range(len(logits)), key=lambda i: logits[i])
                    token_str = id_to_token[next_token]

                    input_ids.append(next_token)
                    llm_output.append(next_token)

                    if token_str.isdigit():
                        digit_count += 1
                        if (
                            token_str == "0"
                            and not seen_decimal
                            and digit_count >= 2
                            and param_index == len(param_keys) - 1
                        ):
                            # Avoid runaway trailing zeros on the last numeric param
                            input_ids.pop()
                            llm_output.pop()
                            digit_count -= 1
                            state = "END_STRUCT"
                            break

                        if seen_decimal:
                            digits_after_decimal += 1
                            if token_str != "0":
                                nonzero_after_decimal = True

                            # If we have many fractional zeros and no significant digits, strip them.
                            if digits_after_decimal >= 4 and not nonzero_after_decimal:
                                # remove decimal and its trailing zeros
                                while len(input_ids) > decimal_index:
                                    input_ids.pop()
                                    llm_output.pop()
                                    digit_count -= 1 if digit_count > 0 else 0
                                seen_decimal = False
                                digits_after_decimal = 0
                                nonzero_after_decimal = False

                                if param_index < len(param_keys) - 1:
                                    tokens = model.encode(',\n')[0].tolist()
                                    input_ids.extend(tokens)
                                    llm_output.extend(tokens)
                                    param_index += 1
                                    state = "ARG_KEY"
                                else:
                                    state = "END_STRUCT"
                                break

                        if steps >= max_steps or digit_count >= max_digits:
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
                        param_index += 1
                        state = "ARG_KEY"
                        break

                    if token_str == "}":
                        state = "END_STRUCT"
                        break

                continue

            # -----------------------
            # STRING TYPE
            # -----------------------
            elif current_param_type == "string":

                # force opening quote
                tokens = model.encode('"')[0].tolist()
                input_ids.extend(tokens)
                llm_output.extend(tokens)

                # now allow content characters
                # tighter cap for regex strings to avoid runaway patterns
                max_string_steps = 20 if current_param_name == "regex" else 128
                char_limit = 32 if current_param_name == "regex" else None
                char_count = 0
                string_steps = 0
                while True:

                    string_steps += 1
                    logits = model.get_logits_from_input_ids(input_ids)
                    allowed_ids = set()

                    for token_str, token_id in vocab.items():

                        # forbid control characters
                        if "\n" in token_str or "\t" in token_str:
                            continue

                        # allow closing quote always
                        if token_str == '"':
                            allowed_ids.add(token_id)
                            continue

                        # allow readable tokens that do not contain quotes
                        if current_param_name == "regex":
                            # limit to common regex characters to reduce runaway generations
                            if '"' in token_str:
                                continue
                            if (
                                token_str.isalnum()
                                or token_str in [" ", "_", "-", ".", "*", "+", "?", "|", "[", "]", "(", ")", "{", "}", "^", "$", "\\"]
                            ):
                                allowed_ids.add(token_id)
                        else:
                            if '"' not in token_str:
                                allowed_ids.add(token_id)

                    logits = mask_logits(logits, allowed_ids)
                    next_token = max(range(len(logits)), key=lambda i: logits[i])
                    token_str = id_to_token[next_token]

                    input_ids.append(next_token)
                    llm_output.append(next_token)

                    if token_str == '"':
                        break

                    if char_limit is not None:
                        char_count += len(token_str)
                        if char_count >= char_limit:
                            close_tokens = model.encode('"')[0].tolist()
                            input_ids.extend(close_tokens)
                            llm_output.extend(close_tokens)
                            break

                    if string_steps >= max_string_steps:
                        # force close to avoid infinite loops
                        close_tokens = model.encode('"')[0].tolist()
                        input_ids.extend(close_tokens)
                        llm_output.extend(close_tokens)
                        break

                state = "ARG_AFTER_VALUE"
                continue

        # =====================================================
        # COMMA OR NEXT PARAM
        # =====================================================
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

        # =====================================================
        # CLOSE JSON
        # =====================================================
        elif state == "END_STRUCT":
            # If the last token already closed the arguments object (e.g., number path),
            # only close the outer JSON. Otherwise close both arguments and the root.
            decoded = model.decode(llm_output).rstrip()
            closing_text = '\n}\n' if decoded.endswith('}') else '\n}\n}\n'
            tokens = model.encode(closing_text)[0].tolist()
            input_ids.extend(tokens)
            llm_output.extend(tokens)
            break

    return llm_output
