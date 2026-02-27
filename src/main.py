#export HF_HOME=/goinfre/berrabia/call_me/hf_cache

from llm_sdk import Small_LLM_Model
import json
import sys



def main_loop(model: Small_LLM_Model, input_ids: list[int]) -> list[int]:
    input_ids.extend(model.encode('{\n"fn_name": ')[0].tolist())

    # for _ in range(max_new_tokens):
    llm_output = model.encode('{\n"fn_name":')[0].tolist()
    # print(llm_output)
    # print()
    args_flag = True
    while "}" not in model.decode(llm_output):
        output_ids = model.get_logits_from_input_ids(input_ids)
        next_token_id = max(range(len(output_ids)), key=lambda i: output_ids[i])
        input_ids.append(next_token_id)
        llm_output.append(next_token_id)
        # print(model.decode(next_token_id))
        if args_flag and "," in model.decode(llm_output):
            input_ids.extend(model.encode('"arguments": {\n\t')[0].tolist())
            llm_output.extend(model.encode('"arguments": {\n\t')[0].tolist())
            args_flag = False
        print(model.decode(llm_output))
    llm_output.extend(model.encode('\n}')[0].tolist())
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