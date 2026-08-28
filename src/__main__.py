import argparse

from src.json_loader import (
    load_functions_definition_from_json,
    load_prompts_from_json,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Translate natural language prompts into function calls"
    )

    parser.add_argument(
        "--input",
        type=str,
        default="data/input/function_calling_tests.json",
    )

    parser.add_argument(
        "--function_definitions",
        type=str,
        default="data/input/functions_definition.json",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/output/function_results.json",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
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
    print("\nFunctions:")
    for function in functions:
        print(function)

    print("\nPrompts:")
    for prompt in prompts:
        print(prompt)


if __name__ == "__main__":
    main()
