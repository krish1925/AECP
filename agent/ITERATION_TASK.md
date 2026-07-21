# AECP Iteration Task

You are an AI agent iteratively improving the AECP repository. Your job is to read the code, understand it, propose ONE targeted change per iteration, apply it, run tests, and either keep or revert.

## Rules

1. **One change per iteration.** Do not rewrite multiple files or do broad refactors. Pick ONE thing to improve.
2. **Reference the rubric.** Every change should address at least one of:
   - Code still runs / protocol handshake still completes
   - Message schemas still validate
   - No blocking calls in message loop
   - Backward compatible
   - Targeted diff addressing one thing
3. **Run tests after every change.** If tests fail, revert the change and explain why.
4. **Never modify files outside the worktree.** The worktree is your sandbox.
5. **Stop after 20 tool-calling iterations.** Do not exceed this cap.

## Test command

```bash
cd /Users/kpatel/Desktop/agent-communication-agent-worktree/aecp-python && /Users/kpatel/Desktop/agent-communication/.agent-venv/bin/python3.13 -m pytest tests/ -x -q -k "not chroma and not langchain"
```

## What to look for

- Bug fixes in existing code
- Missing error handling
- Type annotation gaps
- Performance improvements in hot paths
- Test coverage gaps
- Documentation improvements in docstrings
- Code clarity improvements

## What NOT to do

- Do not add new features or new modules
- Do not change the public API
- Do not restructure the project
- Do not install new dependencies
- Do not modify configuration files (pyproject.toml, etc.)

## Output format

After each iteration, report:
- What you changed and why
- Whether tests passed or failed
- If failed, what the failure was and that you reverted
