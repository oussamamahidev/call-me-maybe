from .models.functions_definition import FunctionDefinition
from .models.prompts import Prompt

import json


def load_functions_definition_from_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return [FunctionDefinition(**item) for item in data]

    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {path}")


def load_prompts_from_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return [Prompt(**item) for item in data]

    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {path}")