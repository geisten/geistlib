## Summary

What this changes and **why**.

## Changes

-

## Testing

- [ ] `make test` passes (or skips are explained)
- [ ] `make MODE=asan test` passes for kernel/memory changes
- [ ] `make format-check` clean
- [ ] user-visible change: a fragment `changes/<slug>.md` (see `changes/README.md`), no `CHANGELOG.md` edit

For perf/kernel changes, before/after numbers and host:

| | before | after |
|---|---|---|
| prefill tok/s | | |
| decode tok/s | | |

Host:

## API impact

- [ ] No change to `include/geist.h`
- [ ] New `EXPERIMENTAL` surface only
- [ ] Breaks a `STABLE` symbol (explain why it's necessary)
