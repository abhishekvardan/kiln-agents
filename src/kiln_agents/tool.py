from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Type, Union

from pydantic import BaseModel

ToolParameters = Union[Type[BaseModel], Dict[str, Any], None]


@dataclass
class ToolExecutionContext:
    run_id: str
    log: Callable[[str, Any], None]


@dataclass
class Tool:
    """A callable capability an agent can use. `parameters` is either a pydantic model (validated
    locally before `execute` runs) or a raw JSON Schema dict (skips local validation — this is how
    MCP tools work, since MCP servers validate the call on their own side)."""

    name: str
    description: str
    execute: Callable[[Any, ToolExecutionContext], Awaitable[Any]]
    parameters: ToolParameters = None
    permissions: List[str] = field(default_factory=list)


def define_tool(
    *,
    name: str,
    description: str,
    execute: Callable[[Any, ToolExecutionContext], Awaitable[Any]],
    parameters: ToolParameters = None,
    permissions: Optional[List[str]] = None,
) -> Tool:
    return Tool(name=name, description=description, execute=execute, parameters=parameters, permissions=permissions or [])
