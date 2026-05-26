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
4.  **Handoff Protocol**: To close a session, the AI agent should commit all annotated work and updated metadata to the active feature branch and push that branch to the `backup` remote. The agent may open a Pull Request (PR) to the `main` branch but **must not** merge it or delete the branch. The final review, PR completion, merge, and branch deletion are reserved for the user to ensure process stability and visibility.
5.  **The "AI Analysis" Pattern**: Every significant demo or simulation in a notebook must be followed by a markdown cell titled `### AI Analysis`. This cell should explain:
    - The conceptual takeaway.
    - The significance of the result.
    - Real-world implications.

## 🛠️ Session Lifecycle
See `agents/WORKFLOW_GUIDE.md` for the exact Git commands to move between workshop branches and the `main` portfolio.
