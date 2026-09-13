from typing import Dict

from pydantic import BaseModel, ConfigDict


class Parameter(BaseModel):
    """
    Function parameter definition.
    """

    model_config = ConfigDict(extra="allow")

    type: str


class ReturnType(BaseModel):
    """
    Function return type definition.
    """

    model_config = ConfigDict(extra="allow")

    type: str


class FunctionDefinition(BaseModel):
    """
    Function definition loaded from JSON.
    """

    model_config = ConfigDict(extra="allow")

    name: str
    description: str
    parameters: Dict[str, Parameter]
    returns: ReturnType
