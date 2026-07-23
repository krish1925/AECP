# Contributing to Isotrieve

**Canonical contributing guide:** [`isotrieve-python/CONTRIBUTING.md`](isotrieve-python/CONTRIBUTING.md)

That file covers development setup, testing, code style, PR expectations, and the escalation protocol for design decisions.

## Quick Reference

```bash
# Clone and setup
git clone https://github.com/krish1925/isotrieve.git
cd isotrieve/isotrieve-python
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pip install -e ".[sentence-transformers]"  # optional: image support

# Before submitting a PR
ruff check src/ tests/            # lint
ruff format --check src/ tests/   # format
mypy src/isotrieve                # types
python -m pytest tests/ -q        # tests (~118 pass, 6 skip expected)
```

## Project Structure

- `isotrieve-python/` — the maintained, benchmark-validated Python package ([PyPI](https://pypi.org/project/isotrieve/))
- `isotrieve-npm/` — historical/experimental NPM protocol package (not actively maintained)
- `isotrieve-website/` — GitHub Pages site
- `benchmarks/` — benchmark harness and results
- `spec/` — protocol specification (RFC-001)

## Which Package is Current?

**`isotrieve` on PyPI** is the actively maintained, benchmark-validated package. The NPM package (`isotrieve-npm/`) is historical/experimental — the maintained project is `isotrieve-python/`.

## Branch Strategy

```
main              ← stable, tagged releases, GitHub Pages
  └── development ← integration branch, CI runs on every push
        └── feat/your-branch   ← feature branches
```

All new work branches from `development`. PRs target `development`. Periodically merge `development` → `main` for releases.
