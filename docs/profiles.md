# Profiles

| Profile | Purpose |
|---|---|
| `stable` | Default profile. Chosen for predictable first-run behavior. |
| `performance-cold` | Opt-in cold throughput path. Preserves the verified 60+ tok/s short/cold result. |
| `exact` | QA and release exactness checks. |
| `max-diagnostic` | Fan-controlled diagnostics only. Not a product headline mode. |

`--max` is separate from profiles. It is opt-in and must restore fan state on exit when supported.
