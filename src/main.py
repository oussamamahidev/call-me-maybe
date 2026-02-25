# we dont build an llm we build an application that uses an llm

#export HF_HOME=/goinfre/berrabia/call_me/hf_cache

from llm_sdk import Small_LLM_Model
import json


def main_loop(model, input_ids: list[int]) -> list[int]:
    input_ids.extend(model.encode('{\n"fn_name": ')[0].tolist())
    
    # for _ in range(max_new_tokens):
    llm_output = model.encode('{\n"fn_name": ')[0].tolist()
    # print(llm_output)
    # print()
    while model.decode(llm_output).count("}") < 2:
        output_ids = model.get_logits_from_input_ids(input_ids)
        next_token_id = max(range(len(output_ids)), key=lambda i: output_ids[i])
        input_ids.append(next_token_id)
        llm_output.append(next_token_id)
        # print(model.decode(llm_output))
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
    print("Loading model...", flush=True)
    model = Small_LLM_Model()
    print("Model loaded.", flush=True)
    # user_input = input("Ask something: ")


    with open("data/input/function_calling_tests.json", "r") as f:
        testes = json.load(f)
    # print(testes)
    for test in testes:
        print(test["prompt"])
        input_ids = model.encode(prompt(test["prompt"]))[0].tolist()
        output_ids = main_loop(model, input_ids)
        print(model.decode(output_ids))

    # print(input_ids)
    # output_ids = model.get_logits_from_input_ids(input_ids)
    # print(model.decode(next_token_id))
    # model.decode(output_ids)
    # print(prompt)

    # print(model.decode(output_ids))
    # generated_text = model.decode(output_ids)
    # print(generated_text)
    # print(output_ids)
if __name__ == "__main__":
    main()