/*
 * test_speculative_int4_norot_int — test_speculative_int4_int with KV
 * rotation off: speculative decode must produce the same token stream as
 * sequential decode_step under packed 4-bit KV without rotation. Its own
 * process so a parallel test run takes it beside the rotated half (#483).
 */
#define SPEC_INT4_ROT_OFF
#include "test_speculative_int4_int.c"
