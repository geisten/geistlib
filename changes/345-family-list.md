### Changed
- **A model family is declared in one place, and wiring it wrong fails the
  build (#345).** `src/archs/transformer/family_list.h` is the only list of
  families: the externs, the registry, the `general.architecture` names, the
  default "no `-DGEIST_FAMILY_*`, so all" selection (the block #241 had to
  fix by hand) and `ALL_FAMILIES` in `mk/common.mk` are generated from it,
  and make stops on a unit without a line or a line without a unit. A unit
  defines its table with `TRANSFORMER_FAMILY(sym, populate, populate_layers,
  attention_block, ffn_block, ...)`, so a table without one of those hooks,
  with one of them `nullptr`, or under a name the list does not have, is a
  compile error; `state_create`'s load-time check for the two layer hooks is
  gone. `make check-static` runs `scripts/check_wiring.py`: the committed
  mistakes in `tests/wiring/` must fail to compile (clang and gcc-14), and
  each family must build and link alone. Adding a family is its file and
  one line.
