# Simplified Environment & Maintenance Guide

## One-Time Setup
```bash
# From project root
python -m pip install -r requirements.txt
```
If you lack a Groq key and want to demo quickly:
```bash
export ALLOW_FAKE_GROQ=1  # mock LLM responses
```
Then run:
```bash
./scripts/run_streamlit.sh --mock
```

## Daily Usage
```bash
export GROQ_API_KEY=your_real_key_here
./scripts/run_streamlit.sh --port 8501
```
Open http://localhost:8501

## Self-Heal Environment
Audit and auto-fix missing or mismatched pinned packages:
```bash
python scripts/self_heal_env.py         # installs fixes
NO_INSTALL=1 python scripts/self_heal_env.py --dry-run  # report only
```
Outputs `env_audit_report.json` summarizing state.

## Keeping Dependencies Updated
Dependabot config (`.github/dependabot.yml`) creates weekly PRs for safe updates.
You can also manually run:
```bash
pip list --outdated | grep -E 'langchain|groq|pydantic'
```
For selective upgrade with pin preservation:
```bash
python -m pip install --upgrade langchain==0.2.12 langchain-core==0.2.27
```
(Keep core & related libs aligned to avoid plugin conflicts.)

## Troubleshooting
- Groq auth errors: verify `echo $GROQ_API_KEY` length (>20 chars)
- Pytest plugin errors: disable auto loading:
  ```bash
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests
  ```
- Network rate limits: set higher throttle:
  ```bash
  export GROQ_THROTTLE_SECONDS=2
  ```
- Mock mode: `ALLOW_FAKE_GROQ=1` gives deterministic sample rules (no network).

## Minimal Integration Code
```python
from core.production_system import ProductionRuleExtractionSystem
system = ProductionRuleExtractionSystem(groq_api_key=os.environ.get("GROQ_API_KEY"))
results = asyncio.run(system.batch_process_documents(["data/smoke/sample.txt"], concurrency=1))
```

## Recommended Update Flow
1. PR from Dependabot arrives.
2. Run `python scripts/self_heal_env.py --dry-run` to inspect.
3. Merge if tests + smoke pass.
4. If breakage: roll back or pin exact previous versions in `requirements.txt`.

## Security Hygiene
- Keep your real `GROQ_API_KEY` only in `.env` locally; do not commit.
- For CI, store it in repository secrets.

---
This guide reduces setup friction; extend `self_heal_env.py` for extra invariants as the project grows.
