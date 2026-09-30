/*
 * test_pin_prefix_recurrent_unit — transformer_pin_prefix refuses a prefix
 * on a session with recurrent (Gated-DeltaNet) state: a reset clears that
 * state to the empty sequence, so the prefix would survive in the attention
 * layers' KV cache but not in the recurrence. Pinning nothing (n = 0) only
 * empties the session and stays allowed.
 *
 * White-box: the tree has no DeltaNet model small enough for a unit test,
 * so the session is a zeroed transformer_arch_session with the recurrent
 * state pointer set, on a model with no layers. The refusal has to come
 * before pin_prefix touches anything, so nothing else is needed.
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "test_helpers.h"

#include "src/archs/transformer/arch_state.h"

#include <geist.h>
#include <geist_backend.h>

#include <stdio.h>

int main(void) {
    struct geist_backend *be = nullptr;
    if (geist_backend_create("cpu_scalar", nullptr, nullptr, &be) != GEIST_OK || be == nullptr) {
        GEIST_SKIP("cpu_scalar backend not built");
    }
    static struct transformer_arch_state   st;
    static struct transformer_arch_session sess;
    static struct geist_buffer            *no_buffers[1];
    st.backend                 = be;
    sess.model                 = &st;
    sess.dn_S                  = no_buffers; /* recurrent state present */
    sess.kv_len                = 7;          /* tokens already in the session */
    const geist_token_t ids[3] = {1, 2, 3};

    int fails = 0;
    fails += geist_expect(transformer_pin_prefix(&sess, 3, ids) == GEIST_E_UNSUPPORTED,
                          "a prefix on a session with DeltaNet state is refused");
    fails += geist_expect(sess.kv_len == 7 && sess.prefix_length == 0,
                          "the refused pin leaves the session as it was");
    fails += geist_expect(transformer_pin_prefix(&sess, 0, ids) == GEIST_OK && sess.kv_len == 0 &&
                                  sess.prefix_length == 0,
                          "pinning nothing still empties the session");

    geist_backend_destroy(be);
    if (fails != 0) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("PASS: pin_prefix refuses a prefix a reset could not return to\n");
    return GEIST_TEST_PASS;
}
