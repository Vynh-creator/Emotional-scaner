# Contribution Guidelines

Thank you for your interest in contributing to the Emotional Scanner project.  
To maintain code quality and consistency, please follow the rules below.

---

## ✅ Branch Strategy

We use the following branching model:

| Branch Type | Description |
|--------------|-------------|
| `main` | Protected branch. Only Team Lead merges here. |
| `dev` | Development branch (optional). |
| `feature/*` | Feature implementation |
| `fix/*` | Bug fixes |
| `docs/*` | Documentation updates |

Example:

feature/emotion-model
fix/video-reader
docs/update-readme


---

## ✅ Commit Rules

Use clear commit messages in English:

feat: add emotion detection model
fix: fix video reader bug
docs: update installation guide
refactor: clean utils module

---

## ✅ Pull Requests

- Every change must go through Pull Request
- Assign **@Илья Злотников** as reviewer
- PR must include:
  - Description
  - What was changed
  - Linked issue (if exists)

---

## ✅ Code Style

- Follow Python code style (PEP8)
- Use Black formatter
- Max line length = 120
- Write clean and readable code

---

## ✅ Environment

```bash
python -m venv venv
source venv/bin/activate  
pip install -r requirements.txt
