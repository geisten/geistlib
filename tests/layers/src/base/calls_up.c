/* SEEDED: a call across the layers without an include: the prototype is
 * repeated here, so only the symbol check sees it. */
int engine_step(int x);

int calls_up(int x);
int calls_up(int x) { return engine_step(x); }
