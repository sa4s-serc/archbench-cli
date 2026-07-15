"""
Architecture View Generation prompt templates.

Provides zero-shot and few-shot prompts that instruct a model to produce
PlantUML diagrams from a repository summary. A component diagram captures the
static architecture; a sequence diagram captures dynamic runtime interactions.
"""

from typing import Dict, List


# =============================================================================
# Few-Shot Example (minimal valid PlantUML to anchor the output format)
# =============================================================================

FEW_SHOT_EXAMPLE = {
    "summary": (
        "A small web service with a REST API layer that talks to a service layer, "
        "which in turn reads and writes to a relational database."
    ),
    "concern": "control_flow",
    "behavior": "static",
    "diagram": """@startuml
[REST API] --> [Service Layer]
[Service Layer] --> [Database]
@enduml""",
}


# =============================================================================
# System Prompts (ported from the CodeToDiagram viewGeneration prompts)
# =============================================================================

def _component_instruction(concern: str, repo_name: str) -> str:
    return f"""You are expert software architect. Your task is to design a view for the system based on the architectural knowledge provided. Use PlantUML diagrams. Based on the following repository summary, generate a **PlantUML component diagram** to capture the static architecture. Focus on the architectural concern: **{concern}**.

Ensure the diagram:
- Clearly shows system components and their relationships.
- Highlights how the architecture addresses the specified concern.
- Is valid PlantUML code with no explanation.
This is the repository name, so please name the generated image the same: {repo_name}."""


def _sequence_instruction(concern: str, repo_name: str) -> str:
    return f"""You are expert software architect. Your task is to design a view for the system based on the architectural knowledge provided. Use PlantUML diagrams. Based on the following repository summary and system behavior, generate a **PlantUML sequence diagram** to show dynamic interactions.

**Behavioral focus:** {concern}

Ensure the diagram:
- Accurately represents runtime message flow between components or services.
- Matches the described system behavior.
- Is valid PlantUML code with no explanation.
This is the repository name, so please name the generated image the same: {repo_name}."""


def create_system_prompt(
    concern: str,
    behavior: str,
    repo_name: str,
) -> str:
    """
    Build the diagram generation system prompt.

    A ``dynamic`` behavior yields a sequence-diagram instruction; anything else
    yields a component-diagram instruction.
    """
    if behavior == "dynamic":
        return _sequence_instruction(concern, repo_name)
    return _component_instruction(concern, repo_name)


# =============================================================================
# Prompt Creation Functions
# =============================================================================

def create_chat_messages(
    summary: str,
    concern: str,
    behavior: str,
    repo_name: str,
    prompt_style: str = "zero_shot",
) -> List[Dict[str, str]]:
    """
    Create a chat-formatted prompt for diagram generation.

    Args:
        summary: The repository summary (inference input)
        concern: Architectural concern to capture
        behavior: System behavior (static / dynamic)
        repo_name: Repository name (used to name the generated image)
        prompt_style: "zero_shot" or "few_shot"

    Returns:
        List of message dictionaries for chat completion APIs
    """
    system = create_system_prompt(concern, behavior, repo_name)

    messages = [{"role": "system", "content": system}]

    if prompt_style == "few_shot":
        # Add a single example turn to anchor the PlantUML output format
        messages.append({"role": "user", "content": FEW_SHOT_EXAMPLE["summary"]})
        messages.append({"role": "assistant", "content": FEW_SHOT_EXAMPLE["diagram"]})

    messages.append({"role": "user", "content": summary.strip()})

    return messages
