### Changed
- **The Mac and Pi host locks queue through a lease broker.** `with-mac-lock`
  and `with-pi-lock` keep their commands, `WAIT`, `status` and exit codes,
  but waiters are now served first come first served within a priority
  (`LOCK_PRIO=ci|interactive|batch`, with aging), a holder declares
  `LOCK_MAX` and shows as OVERTIME/EXPIRED past it, and the `mac-m1max`
  runner no longer stalls silently behind a held lock: its health check asks
  whether a CI lease would come within 10 min, and waiting heavy jobs get the
  next hand-over. Before, lockf/flock waiters woke in random order and a 3 h
  benchmark blocked CI with nothing to show for it. `docs/HARDWARE_CI.md`
  "Host locks"; code in geisten-infra (`host_lease`).
