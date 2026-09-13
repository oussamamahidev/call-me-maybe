import json
from typing import List

from pydantic import ValidationError

from src.models.functions_definition import FunctionDefinition
from src.models.prompts import Prompt


def load_function_definition(path: str) -> List[FunctionDefinition]:
    """
    Load and validate function definitions from a JSON file.
    """
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, list):
            raise RuntimeError(
                f"Expected a JSON array in {path}"
            )

        return [
            FunctionDefinition(**item)
            for item in data
        ]

    except FileNotFoundError as error:
        raise RuntimeError(
            f"File not found: {path}"
        ) from error

    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"Invalid JSON in {path}"
        ) from error

    except (ValidationError, TypeError) as error:
        raise RuntimeError(
            f"Invalid function definition in {path}: {error}"
        ) from error


def load_prompts(path: str) -> List[Prompt]:
    """
    Load and validate natural-language prompts from a JSON file.
    """
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, list):
            raise RuntimeError(
                f"Expected a JSON array in {path}"
            )

        return [
            Prompt(**item)
            for item in data
        ]

    except FileNotFoundError as error:
        raise RuntimeError(
            f"File not found: {path}"
        ) from error

    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"Invalid JSON in {path}"
        ) from error

    except (ValidationError, TypeError) as error:
        raise RuntimeError(
            f"Invalid prompt definition in {path}: {error}"
        ) from error
