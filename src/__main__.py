import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional

from llm_sdk.llm_sdk import Small_LLM_Model

from src.constrained_decoding import (
    build_json_valid_ids,
    build_system_prompt,
    extract_complete_json,
    get_best_valid_token,
    load_vocabulary,
    validate_function_call,
)
from src.json_loader import (
    load_function_definition,
    load_prompts,
)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Translate natural language prompts into "
            "function calls using constrained decoding."
        )
    )

    parser.add_argument(
        "--input",
        type=str,
        default=(
            "data/input/"
            "function_calling_tests.json"
        ),
        help="Path to the JSON file containing user prompts.",
    )

    parser.add_argument(
        "--functions_definition",
        type=str,
        default=(
            "data/input/"
            "functions_definition.json"
        ),
        help=(
            "Path to the JSON file containing "
            "available function definitions."
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=(
            "data/output/"
            "function_calling_results.json"
        ),
        help="Path where generated function calls are saved.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Identifier of the model to use.",
    )

    return parser.parse_args()


def main() -> None:
    """
    Run the function-calling generation pipeline.
    """
    print("🚀 Starting function calling...")

    args = parse_args()

    print("📂 Loading functions and prompts...")

    functions = load_function_definition(
        args.functions_definition
    )

    if not functions:
        raise RuntimeError(
            "No function definitions found. "
            "Please provide at least one."
        )

    prompts = load_prompts(args.input)

    if not prompts:
        raise RuntimeError(
            "No prompts found in input file. "
            "Please provide at least one."
        )

    print("⚙️ Building system prompt...")
    system = build_system_prompt(functions)

    print(f"🤖 Loading model: {args.model}")

    try:
        model = Small_LLM_Model(
            model_name=args.model
        )
    except OSError as error:
        raise RuntimeError(
            f"Model {args.model} not found "
            "or failed to download."
        ) from error

    print("🔤 Building valid token IDs...")

    vocab = load_vocabulary(model)
    valid_ids = build_json_valid_ids(vocab)

    if not valid_ids:
        raise RuntimeError(
            "Could not build valid token IDs."
        )

    all_results: List[Dict[str, Any]] = []
    start_time = time.time()

    print("⚙️ Processing prompts...\n")

    for prompt_object in prompts:
        prompt = prompt_object.prompt

        print(
            f"⏳ Processing prompt: {prompt}"
        )

        full_prompt = (
            f"{system}\n\n"
            f"User prompt: {prompt}\n"
            "Assistant:"
        )

        input_ids = model.encode(full_prompt)
        generated_ids = input_ids[0].tolist()

        all_generated: List[int] = []
        clean_json: Optional[str] = None
        parsed: Optional[Dict[str, Any]] = None

        json_start = model.encode(
            '{"name": "'
        )[0].tolist()

        all_generated.extend(json_start)

        for _ in range(50):
            logits = model.get_logits_from_input_ids(
                generated_ids + all_generated
            )

            next_id = get_best_valid_token(
                logits,
                valid_ids,
            )

            all_generated.append(next_id)

            text = model.decode(all_generated)

            clean_json = extract_complete_json(
                text
            )

            if clean_json is None:
                continue

            try:
                candidate = json.loads(clean_json)
            except json.JSONDecodeError:
                break

            if validate_function_call(
                candidate,
                functions,
            ):
                parsed = candidate

            break

        if parsed is None:
            parsed = {
                "name": "none",
                "parameters": {},
            }

        result: Dict[str, Any] = {
            "prompt": prompt,
            "name": parsed.get(
                "name",
                "none",
            ),
            "parameters": parsed.get(
                "parameters",
                {},
            ),
        }

        all_results.append(result)

        if result["name"] != "none":
            print(
                f"  -> ✅ {result['name']}"
                f"({result['parameters']})"
            )
        else:
            print(
                "  -> ❌ Could not generate "
                "a valid function call."
            )

    total_time = time.time() - start_time

    successful_results = [
        result
        for result in all_results
        if result["name"] != "none"
    ]

    output_directory = os.path.dirname(
        args.output
    )

    if output_directory:
        os.makedirs(
            output_directory,
            exist_ok=True,
        )

    try:
        with open(
            args.output,
            "w",
            encoding="utf-8",
        ) as output_file:
            json.dump(
                all_results,
                output_file,
                ensure_ascii=False,
                indent=2,
            )
    except OSError as error:
        raise RuntimeError(
            "Could not write output file: "
            f"{args.output}"
        ) from error

    print(
        "\n💾 Results saved to: "
        f"{args.output}"
    )

    print("✨ Completed.")

    print(
        "⏱️ Total time: "
        f"{total_time:.2f} seconds"
    )

    average_time = total_time / len(prompts)

    print(
        "📈 Average per prompt: "
        f"{average_time:.2f} seconds"
    )

    success_count = len(successful_results)
    total_count = len(prompts)

    success_rate = (
        success_count
        / total_count
        * 100
    )

    print(
        "🎯 Success rate: "
        f"{success_count}/{total_count} "
        f"({success_rate:.2f}%)"
    )


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print("\nUser stopped the program.")

    except Exception as error:
        print(f"Error: {error}")