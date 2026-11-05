# Equivalence harness

Proves that in-scope workflows behave exactly like their recorded baseline.
Full description: `docs/testing/equivalence.md`.

```shell
python tests/equivalence/static_check.py check
python tests/equivalence/run_equiv.py check
pytest tests/equivalence
```
