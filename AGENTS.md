# Repository Instructions for Coding Agents (MLAI Workshops)

## 🎯 Purpose
This repository is used for educational exploration of ML foundations. We maintain a "Master Portfolio" on the `main` branch that archives completed and annotated workshops.

## 📋 Core Rules

1.  **Environment**: Always use `uv` for execution.
    - `uv run python ...`
2.  **Portfolio Structure**: 
    - Completed workshops live in `workshops/<name>/`.
    - Session metadata lives in `agents/`.
    - **Never** commit work directly to `main` root; always use the `workshops/` subfolders.
3.  **Context Injection**: When starting a new session from an official `origin/workshop-X` branch, you **MUST** run `git checkout main -- agents/` to restore the session logs and these instructions.
4.  **The "AI Analysis" Pattern**: Every significant demo or simulation in a notebook must be followed by a markdown cell titled `### AI Analysis`. This cell should explain:
    - The conceptual takeaway.
    - The significance of the result.
    - Real-world implications.

## 🛠️ Session Lifecycle
See `agents/WORKFLOW_GUIDE.md` for the exact Git commands to move between workshop branches and the `main` portfolio.
