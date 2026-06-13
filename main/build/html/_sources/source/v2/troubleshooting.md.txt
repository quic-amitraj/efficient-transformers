# Troubleshooting

Use this page as the first triage checklist when docs, compile, or runtime behavior is not as expected.

## Common Issues

| Symptom | Likely Cause | Action |
| --- | --- | --- |
| `cannot import name 'AwqBackendPackingMethod'` | `transformers` version mismatch vs repository expectation | Align dependency versions with target release branch and rebuild environment |
| QPC runs fail after compile settings change | Runtime args do not match compile profile | Recompile with intended profile and reuse matching `device_group` |
| Cached artifacts reused unexpectedly | Shared cache path across experiments | Set per-run `QEFF_HOME` for isolated experiments |
| Linkcheck intermittently fails | External site rate-limit/bot guard | Re-run linkcheck; keep persistent offenders in `linkcheck_ignore` |
| Autodoc warnings during docs build | Optional runtime deps unavailable in docs-only env | Expected in minimal docs env; resolve by installing compatible runtime deps |

## Fast Triage Sequence

1. Capture exact command and full stack trace.
2. Confirm active virtualenv and package versions.
3. Confirm compile/runtime profile consistency.
4. Retry with isolated cache (`QEFF_HOME`).
5. Minimize command to a known-good baseline.

## Debug Artifacts to Collect

When filing issues, include:

- command used,
- model name,
- compile profile flags,
- device group,
- environment/package versions,
- complete logs.

## Escalation Path

Open an issue in the project tracker with reproduction steps and logs:

- https://github.com/quic/efficient-transformers/issues
