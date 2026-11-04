### Changed
- **The tree builds with `-Wnull-dereference` under `-Werror` (#343).**
  gcc reports a dereference its optimizer sees on a null path. It found two
  sites, neither reachable today: `vk_linear_t_grouped` re-looked up a
  weight buffer it had validated before (now checked where it is used), and
  the Vulkan parity test's `qfmt_of` returned null for an unlisted dtype
  (now a harness error, exit 99).
