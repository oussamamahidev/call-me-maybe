from llm_sdk import Small_LLM_Model
import json
import sys
import os
from src.finite_state_machine import main_loop
from typing import Dict, List


def prompt(test: str) -> str:
    try:
        with open("data/input/functions_definition.json", "r") as f:
            functions = json.load(f)
    except IOError:
        print("eror with the data/input/functions_definition.json file")
        sys.exit()

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


def main() -> None:
    model = Small_LLM_Model()
    # print(model.get_path_to_vocab_file())

    try:
        if "--input" in sys.argv:
            tests_file = sys.argv[sys.argv.index("--input") + 1]
        else:
            tests_file = "data/input/function_calling_tests.json"
    except Exception:
        tests_file = "data/input/function_calling_tests.json"

    try:
        if "--output" in sys.argv:
            results_file = sys.argv[sys.argv.index("--output") + 1]
        else:
            results_file = "data/output/function_calling_results.json"
    except Exception:
        results_file = "data/output/function_calling_results.json"

    try:
        with open(tests_file, "r") as f:
            testes = json.load(f)
    except Exception:
        print("missing testing file bruu ;)")
        sys.exit()

    final_output_list: List[Dict] = []
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
    except Exception as e:
        print(f'\nerror : {e}')
        sys.exit()

    try:
        directory = os.path.dirname(results_file)

        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(results_file, "w") as f:
            f.write(json.dumps(final_output_list, indent=2))
    except Exception as e:
        print(f"error with the output file ;) {e}")
        sys.exit()


if __name__ == "__main__":
    main()
