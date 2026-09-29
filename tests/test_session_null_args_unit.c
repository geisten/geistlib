/*
 * test_session_null_args_unit — the public session API answers a null
 * array with GEIST_E_INVALID_ARG in every build mode.
 *
 * The array parameters used to be declared GEIST_AT_LEAST(n), i.e.
 * non-null, while the bodies checked them for null anyway. gcc and clang
 * believed the declaration and deleted the check at -O1 and above, so a
 * release build dereferenced the null instead. `make test-unit
 * MODE=release` is the build this test is for; against the old headers it
 * does not even compile (-Wnonnull).
 *
 * Every argument check runs before the session is touched, so a dummy
 * handle is enough: no model, no fixture.
 */
#include "test_helpers.h"

#include <geist.h>
#include <geist_util.h>

#include <stddef.h>
#include <stdio.h>

#define EXPECT_INVALID(call) fails += geist_expect((call) == GEIST_E_INVALID_ARG, #call)

int main(void) {
    static max_align_t          dummy; /* never dereferenced — see above */
    struct geist_session *const s      = (struct geist_session *) &dummy;
    geist_token_t               tok[4] = {0};
    size_t                      n      = 0;
    int                         fails  = 0;

    EXPECT_INVALID(geist_session_tokenize(s, "x", 4, nullptr, &n));
    EXPECT_INVALID(geist_session_prefill_tokens(s, 4, nullptr));
    EXPECT_INVALID(geist_session_pin_prefix(s, 4, nullptr));
    EXPECT_INVALID(geist_session_decode_speculative(s, 2, 4, nullptr, 4, tok, &n));
    EXPECT_INVALID(geist_session_decode_speculative(s, 2, 4, tok, 4, nullptr, &n));
    EXPECT_INVALID(geist_session_attach_audio(s, 4, nullptr, 16000));
    EXPECT_INVALID(geist_session_audio_push(s, 4, nullptr));
    EXPECT_INVALID(geist_session_attach_image(s, 2, 2, nullptr));
    EXPECT_INVALID(geist_session_attach_video(s, 1, 2, 2, nullptr));

    if (fails > 0) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("PASS: a null array is rejected at all nine array parameters\n");
    return GEIST_TEST_PASS;
}
