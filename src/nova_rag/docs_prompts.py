"""Prompt templates for documentation generation via Claude CLI."""

from __future__ import annotations

# ── Language mapping ──

LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "uk": "Ukrainian",
    "ru": "Russian",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "it": "Italian",
    "pl": "Polish",
    "nl": "Dutch",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "ar": "Arabic",
    "hi": "Hindi",
    "tr": "Turkish",
    "cs": "Czech",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "no": "Norwegian",
}


def _lang_instruction(language: str) -> str:
    """Return a language instruction line, or empty string for English."""
    if language == "en":
        return ""
    name = LANGUAGE_NAMES.get(language, language)
    return f"\n\nIMPORTANT: Write ALL documentation in {name}. All headings, descriptions, and explanations must be in {name}. Code identifiers and Mermaid labels stay in English."


# ── System prompts ──

LEAF_SYSTEM_PROMPT = """\
You are a senior technical writer generating comprehensive module documentation.

OUTPUT FORMAT:
- Write a single Markdown document for the module.
- Start with a level-1 heading: # {module_name}
- Include these sections (use level-2 headings):
  ## Overview — purpose and responsibility of this module (2-4 sentences)
  ## Architecture — Mermaid diagram showing component relationships
  ## Components — describe each key function/class with its role
  ## Data Flow — how data moves through the module (Mermaid sequence or flowchart if useful)
  ## Dependencies — which other modules this one depends on or is used by

MERMAID GUIDELINES:
- Use ```mermaid code blocks
- Keep diagrams focused — max 15 nodes per diagram
- Use clear, descriptive labels (not just class names)
- Prefer flowchart TD or graph TD for architecture, sequenceDiagram for flows

STYLE:
- Be concise but thorough
- Focus on the "why" and "how", not just listing code
- Reference other modules by name for cross-linking: [Module Name](module-name.md)
- Do NOT include raw source code — summarise and explain instead{lang_instruction}\
"""

PARENT_SYSTEM_PROMPT = """\
You are a senior technical writer generating a module overview that ties together its sub-modules.

OUTPUT FORMAT:
- Write a single Markdown document.
- Start with: # {module_name}
- Sections:
  ## Overview — what this module group does as a whole
  ## Architecture — Mermaid diagram showing how sub-modules relate
  ## Sub-Modules — brief summary of each child with links: [Name](name.md)
  ## Cross-Cutting Concerns — shared patterns, common utilities, design decisions

STYLE:
- Do NOT repeat content from sub-module docs — link to them instead
- Keep it high-level and navigational
- Use Mermaid diagrams to show relationships between sub-modules{lang_instruction}\
"""

OVERVIEW_SYSTEM_PROMPT = """\
You are a senior technical writer generating a repository-level documentation overview.

OUTPUT FORMAT:
- Start with: # {project_name}
- Sections:
  ## Overview — what this project does, its purpose (3-5 sentences)
  ## Architecture — end-to-end Mermaid diagram showing all major modules
  ## Module Index — table with each module name, file link, and one-line description
  ## Tech Stack — languages, frameworks, key dependencies
  ## Getting Started — brief pointers for new developers

STYLE:
- This is the entry point — make it scannable and useful
- Link to every module doc: [Module Name](modules/module-name.md)
- Keep the architecture diagram high-level (top modules only)
- Do NOT dump details — point readers to the right module doc{lang_instruction}\
"""


# ── User prompt builders ──

def build_leaf_prompt(
    module_name: str,
    source_code: str,
    module_tree_summary: str,
    components: list[str],
) -> tuple[str, str]:
    """Build system + user prompt for a leaf module.

    Returns (system_prompt, user_prompt).
    """
    system = LEAF_SYSTEM_PROMPT.format(
        module_name=module_name,
        lang_instruction="",
    )
    user = f"""\
Generate documentation for the **{module_name}** module.

## Module position in the project
{module_tree_summary}

## Components in this module
{chr(10).join(f'- {c}' for c in components)}

## Source code
{source_code}
"""
    return system, user


def build_parent_prompt(
    module_name: str,
    children_docs: dict[str, str],
    module_tree_summary: str,
) -> tuple[str, str]:
    """Build system + user prompt for a parent module overview."""
    system = PARENT_SYSTEM_PROMPT.format(
        module_name=module_name,
        lang_instruction="",
    )
    children_section = ""
    for child_name, doc_content in children_docs.items():
        # Include a truncated version of each child's doc
        truncated = doc_content[:3000]
        if len(doc_content) > 3000:
            truncated += "\n... (truncated)"
        children_section += f"\n### {child_name}\n{truncated}\n"

    user = f"""\
Generate an overview for the **{module_name}** module group.

## Module position in the project
{module_tree_summary}

## Sub-module documentation
{children_section}
"""
    return system, user


def build_overview_prompt(
    project_name: str,
    module_tree_summary: str,
    module_docs_summary: dict[str, str],
) -> tuple[str, str]:
    """Build system + user prompt for the repository overview."""
    system = OVERVIEW_SYSTEM_PROMPT.format(
        project_name=project_name,
        lang_instruction="",
    )
    docs_section = ""
    for mod_name, doc_content in module_docs_summary.items():
        # Only first ~1500 chars of each module doc for context
        truncated = doc_content[:1500]
        if len(doc_content) > 1500:
            truncated += "\n... (truncated)"
        docs_section += f"\n### {mod_name}\n{truncated}\n"

    user = f"""\
Generate the repository overview for **{project_name}**.

## Project module structure
{module_tree_summary}

## Module documentation summaries
{docs_section}
"""
    return system, user


def format_module_tree(module_tree: dict, indent: int = 0) -> str:
    """Format module tree as a readable text summary."""
    lines: list[str] = []
    for name, info in sorted(module_tree.items()):
        prefix = "  " * indent
        n_components = len(info.get("components", []))
        n_files = len(info.get("file_paths", []))
        lines.append(f"{prefix}- **{name}** ({n_files} files, {n_components} components)")

        children = info.get("children", {})
        if children:
            lines.append(format_module_tree(children, indent + 1))

    return "\n".join(lines)


def apply_language(system_prompt: str, language: str) -> str:
    """Inject language instruction into a system prompt."""
    lang_inst = _lang_instruction(language)
    return system_prompt.replace("{lang_instruction}", lang_inst)
