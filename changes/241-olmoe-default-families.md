### Fixed
- **A build that selects no family gets OLMoE too (#241).** `arch_family.c`'s
  "no `-DGEIST_FAMILY_*`, so all of them" block left it out.
