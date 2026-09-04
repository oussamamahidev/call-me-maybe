import argparse
import os
from llm_sdk import Small_LLM_Model
from src.json_loader import (
    load_functions_definition_from_json,
    load_prompts_from_json,
)
import json
import time
from src.constrained_decoding import build_system_prompt, load_vocabulary, build_json_valid_ids, get_best_valid_ids,extract_json_from_text
def parse_args():
    parser = argparse.ArgumentParser(
        description="Translate natural language prompts into function calls"
    )

    parser.add_argument(
        "--input",
        type=str,
        default="data/input/function_calling_tests.json",
        help="Path to the input JSON file containing prompts.",
    )

    parser.add_argument(
        "--function_definitions",
        type=str,
        default="data/input/functions_definition.json",
        help="Path to the JSON file containing function definitions.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/output/function_results.json",
        help="Path to the output JSON file to save the results.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="The model to use for generating function calls.",
    )

    return parser.parse_args()


def main():
    print("Starting function and prompts...")

    args = parse_args()

    print("Loading functions...")

    functions = load_functions_definition_from_json(
        args.function_definitions
    )

    print("Loading prompts...")

    prompts = load_prompts_from_json(
        args.input
    )
    if not functions:
        raise RuntimeError("No functions loaded. Please check the function definitions file.")
    if not prompts:
        raise RuntimeError("No prompts loaded. Please check the input file.")

    print("Processing prompts...")

    system = build_system_prompt(functions)
    print(system)

    print(f"loading model... {args.model}")

    try:
        model = Small_LLM_Model(args.model)
    except OSError as e:
        raise RuntimeError(f"Failed to load model '{args.model}': {e}")


    print(" building valid token IDs.....")
    vocabulary = load_vocabulary(model)
    valid_vocab= build_json_valid_ids(vocabulary)

    all_results= []

    start_time = time.time()

    print("processing prompt.....\n")

    for p in prompts:
        prompt = p.prompt

        print(f"processing prompt {prompt}")
        full_prompt = f"{system}\n\nUser : prompt {prompt}\n\nAssistant : "
        input_ids = model.encode(full_prompt)
        generated_ids = input_ids[0].tolist()

        all_generated = []
        clean_json = None
        all_generated.extend(model.encode('{"name": "')[0].tolist())
        for _ in range(50):
            logits = model.get_logits_from_input_ids(generated_ids + all_generated)
            next_ids = get_best_valid_ids(logits, valid_vocab)
            all_generated.append(next_ids)
            text = model.decode(all_generated)
            clean_json = extract_json_from_text(text)
            print(text)
            

            if clean_json:
                try:
                    parsed = json.loads(clean_json)
                    break
                except Exception:
                    pass
        if not clean_json:
            parsed = {"name": None, "args": {}}

        all_results.append({
            "prompt": prompt,
            "name": parsed.get("name", None),
            "args": parsed.get("args", {}),

            
        })

        if parsed.get("name", None) != None:
            print(f" -> ✅ {parsed['name']} ({parsed.get('args', {})})")
        else:
            print(f" ->   [ERROR] could not generate function call")

    total_time = time.time() - start_time
    all_parsed = [result for result in all_results if result["name"] != "none"]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_parsed,f,ensure_ascii=False,indent=2)


    print(f" Result save to {args.output}")

    print("Completing...")

    print(f" Total time : {total_time :.2} seconds")

    print(f" Average per prompt :{total_time / len(prompt):.2} seconds")
if __name__ == "__main__":
    main()

