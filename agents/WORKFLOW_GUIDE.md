# Workshop Agent Workflow Guide

This guide defines how to manage the transition between official workshop branches and our custom "Master Portfolio" on the `main` branch.

## 🏁 Starting a New Session (e.g., Workshop 3)

1.  **Sync Main**: Ensure you are on `main` and it's up to date.
    ```bash
    git checkout main
    ```
2.  **Fetch New Content**: Fetch the next official workshop branch.
    ```bash
    git fetch origin workshop3
    ```
3.  **Create Working Branch**: Create a local branch to do the work.
    ```bash
    git checkout -b feature/workshop3 origin/workshop3
    ```
4.  **Inject Context**: Crucially, bring our `agents/` folder from `main` into the new branch so the AI has access to previous logs and instructions.
    ```bash
    git checkout main -- agents/
    ```

## 🛠️ During the Session

- Work on notebooks in the `notebooks/` folder.
- Follow the **"AI Analysis"** pattern: Add markdown cells after demos to explain the "why" and "how."
- Update `agents/WORKFLOW_STATUS.md` as you complete modules.

## 🏁 Closing the Session

1.  **Commit Work**: Commit your annotated notebooks to the feature branch.
2.  **Archive to Portfolio**:
    - Switch back to `main`.
    ```bash
    git checkout main
    ```
    - Create a new portfolio subfolder.
    ```bash
    mkdir -p workshops/03_validation
    ```
    - Extract the notebooks from the working branch.
    ```bash
    git checkout feature/workshop3 -- notebooks/
    mv notebooks/* workshops/03_validation/
    rm -rf notebooks
    ```
3.  **Finalize Logs**: Update `agents/SESSION_CONTEXT.md` and `agents/SELF_REFLECTION.md` on `main`.
4.  **Commit Archive**:
    ```bash
    git add workshops/ agents/
    git commit -m "docs: archive workshop 3 to portfolio"
    ```

## ⚠️ Important Note
Treat the `main` branch as the **Source of Truth** for the `agents/` folder. Always pull it from `main` when starting a new branch, and always commit updates back to `main` when finishing.
