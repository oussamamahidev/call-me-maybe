#export HF_HOME=/goinfre/berrabia/call_me/hf_cache

from llm_sdk import Small_LLM_Model
import json
import sys


import json


def main_loop(model, input_ids: list[int]) -> list[int]:

    # -------------------------------------------------
    # Load vocab
    # -------------------------------------------------
    with open(model.get_path_to_vocab_file(), "r") as f:
        vocab = json.load(f)

    id_to_token = {v: k for k, v in vocab.items()}

    # -------------------------------------------------
    # Load function definitions
    # -------------------------------------------------
    with open("data/input/functions_definition.json") as f:
        functions = json.load(f)

    function_names = [f["name"] for f in functions]

    # -------------------------------------------------
    # FSM STATE
    # -------------------------------------------------
    state = "START"
    selected_function = None
    param_keys = []
    param_index = 0
    current_param_type = None

    llm_output = []

    # -------------------------------------------------
    # Helper: mask everything except allowed tokens
    # -------------------------------------------------
    def mask_logits(logits, allowed_ids):
        for i in range(len(logits)):
            if i not in allowed_ids:
                logits[i] = -float("inf")
        return logits

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
                decimal_index = None

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
                while True:

                    logits = model.get_logits_from_input_ids(input_ids)
                    allowed_ids = set()

                    for token_str, token_id in vocab.items():

                        # forbid control characters
                        if "\n" in token_str or "\t" in token_str:
                            continue

                        # allow closing quote
                        if token_str == '"':
                            allowed_ids.add(token_id)

                        # allow normal characters
                        elif token_str.isalnum() or token_str in [" ", "_"]:
                            allowed_ids.add(token_id)

                    logits = mask_logits(logits, allowed_ids)
                    next_token = max(range(len(logits)), key=lambda i: logits[i])
                    token_str = id_to_token[next_token]

                    input_ids.append(next_token)
                    llm_output.append(next_token)

                    if token_str == '"':
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

def prompt(test: str) -> str:
    with open("data/input/functions_definition.json", "r") as f:
        functions = json.load(f)

    return f"""
    You are a function-calling assistant.

    You must select exactly one function from the list below
    and return a valid JSON object describing the function call.

    Available functions:
    {json.dumps(functions, indent=2)}

    Rules:
    - You must choose one function from the list.
    - Do NOT return text.
    - You must provide all required arguments.
    - Argument types must match the specification.
    - Do NOT return explanations.
    - Return ONLY valid JSON.

    The JSON format must be exactly:

    {{
    "fn_name": "<function_name>",
    "arguments": {{
        "<arg_name>": <value>
    }}
    }}

    User request:
    {test}
    """

def main():
    model = Small_LLM_Model()
    # print(model.get_path_to_vocab_file())

    try:
        if "--input" in sys.argv:
            tests_file = sys.argv[sys.argv.index("--input") + 1]
        else:
            tests_file = "data/input/function_calling_tests.json"
    except IndexError:
            tests_file = "data/input/function_calling_tests.json"

    try:
        if "--output" in sys.argv:
            results_file = sys.argv[sys.argv.index("--output") + 1]
        else:
            results_file = "data/output/function_calling_results.json"
    except IndexError:
            results_file = "data/output/function_calling_results.json"

    try:
        with open(tests_file, "r") as f:
            testes = json.load(f)
    except Exception:
        print("missing testing file bruu ;)")
        sys.exit()

    final_output_list: list[dict] = []
    try:
        for test in testes:
            print(test["prompt"])
            input_ids = model.encode(prompt(test["prompt"]))[0].tolist()
            output_ids = main_loop(model, input_ids)
            llm_output = json.loads(model.decode(output_ids))
            final_output_list.append(
                {
                "prompt": test["prompt"],
                "fn_name": llm_output["fn_name"],
                "args": llm_output["arguments"]
                }
            )
        # print(final_output_list)
    except Exception as e:
        print(f'\nerror : {e}')
        sys.exit()

    try:
        with open(results_file, "w") as f:
            f.write(json.dumps(final_output_list, indent=2))
    except Exception:
        print("error with the output file ;) ")
        sys.exit()

if __name__ == "__main__":
    main()