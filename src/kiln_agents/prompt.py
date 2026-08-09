from dataclasses import dataclass
from typing import List, Optional, Union


@dataclass
class PromptOptions:
    persona: Optional[str] = None
    """Who the agent is / its permanent behavior."""
    task: Optional[str] = None
    """What this specific call is asking for."""
    constraints: Optional[Union[str, List[str]]] = None
    """Hard rules — rendered as a bullet list. A single string is treated as one rule."""
    examples: Optional[Union[str, List[str]]] = None
    """Representative input/output pairs — rendered as their own section, in order."""
    output_format: Optional[str] = None
    """Free-form output-shape guidance for non-object() calls (object() already writes its own schema instruction)."""


def _to_list(value: Optional[Union[str, List[str]]]) -> List[str]:
    if not value:
        return []
    return value if isinstance(value, list) else [value]


def define_prompt(options: PromptOptions) -> str:
    """Composes a system prompt from named sections instead of ad-hoc string concatenation.
    Sections merge in a fixed, predictable order (persona, task, constraints, examples,
    output_format); omit whichever you don't need."""
    sections: List[str] = []
    if options.persona:
        sections.append(options.persona.strip())
    if options.task:
        sections.append(f"Task: {options.task.strip()}")
    constraints = _to_list(options.constraints)
    if constraints:
        sections.append("Constraints:\n" + "\n".join(f"- {rule}" for rule in constraints))
    examples = _to_list(options.examples)
    if examples:
        sections.append("Examples:\n" + "\n\n".join(examples))
    if options.output_format:
        sections.append(f"Output format: {options.output_format.strip()}")
    return "\n\n".join(sections)


def combine_prompts(*parts: Union[str, None, bool]) -> str:
    """Merges multiple prompt strings/sections (e.g. a shared safety or house-style block reused
    across agents) into one, dropping any falsy part."""
    return "\n\n".join(part.strip() for part in parts if part and isinstance(part, str) and part.strip())
