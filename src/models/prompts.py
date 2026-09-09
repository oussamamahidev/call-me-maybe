from pydantic import BaseModel


class Prompt(BaseModel):
    """
    Natural-language input prompt.
    """

    prompt: str