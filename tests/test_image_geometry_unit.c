/*
 * test_image_geometry_unit — caller-supplied image geometry (issue #330).
 *
 * height and width arrive from geist_session_attach_image / _video and
 * are multiplied out several layers down: height * width * 3 for the
 * pixel extent, width * 3 for a row stride that then narrows to an int
 * on the way into stb_image_resize2. None of those products were checked
 * and nothing bounded the dimensions, so a large enough pair produced a
 * negative stride or a wrapped allocation size.
 *
 * Hermetic: synthesizes its own RGB, needs no model and no fixture.
 */
#include "test_helpers.h"

#include "image_pipeline.h"
#include "heap.h"

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int g_fail = 0;

#define CHECK(cond)                                                         \
    do {                                                                    \
        if (!(cond)) {                                                      \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            g_fail = 1;                                                     \
        }                                                                   \
    } while (0)

int main(void) {
    struct image_plan plan;

    /* ---- the bound ------------------------------------------------------
     * IMAGE_PIPELINE_MAX_DIM is where width * 3 stops fitting in the int
     * row stride stbir_resize takes. Above it the geometry is not merely
     * large, it is unrepresentable. */
    CHECK(IMAGE_PIPELINE_MAX_DIM == (size_t) (INT_MAX / 3));
    CHECK(!image_pipeline_plan(IMAGE_PIPELINE_MAX_DIM + 1, 64, 280, &plan));
    CHECK(!image_pipeline_plan(64, IMAGE_PIPELINE_MAX_DIM + 1, 280, &plan));
    CHECK(!image_pipeline_plan(SIZE_MAX, SIZE_MAX, 280, &plan));
    CHECK(!image_pipeline_plan(SIZE_MAX / 3, 4, 280, &plan));

    /* Degenerate inputs the planner already refused; keep them pinned. */
    CHECK(!image_pipeline_plan(0, 64, 280, &plan));
    CHECK(!image_pipeline_plan(64, 0, 280, &plan));
    CHECK(!image_pipeline_plan(64, 64, 0, &plan));
    CHECK(!image_pipeline_plan(64, 64, 280, nullptr));

    /* ---- and an ordinary image still goes through ------------------------
     * The point of the positive case: a bound that rejects real images
     * would be worse than no bound.
     *
     * One declarator each: GCC rejects a constexpr declaration with more
     * than one, and CI builds with gcc-14 on every Linux target. */
    constexpr size_t H = 224;
    constexpr size_t W = 320;
    CHECK(image_pipeline_plan(H, W, 280, &plan));
    CHECK(plan.resized_h == 672 && plan.resized_w == 960);
    if (g_fail) {
        return GEIST_TEST_FAIL;
    }
    CHECK(plan.resized_h % 48 == 0 && plan.resized_w % 48 == 0);
    CHECK(plan.grid_h == plan.resized_h / 16 && plan.grid_w == plan.resized_w / 16);
    CHECK(plan.soft_tokens > 0 && plan.soft_tokens <= 280);

    /* Preprocess at the planner's own fixed point (672x960 plans to
     * itself), so `src` aliases the input and stbir_resize is not called.
     *
     * Deliberate: stb_image_resize2.h trips UBSan's object-size check on
     * its scanline function-pointer table (stb_image_resize2.h:6270), and
     * vendored third-party UB is not this test's subject — silencing it
     * would mean turning UBSan off for stb project-wide. The cost is that
     * the checked product inside preprocess's resize branch keeps no
     * direct coverage here; it is three lines of ckd_mul over planner
     * output whose inputs image_pipeline_plan has already bounded. */
    struct image_plan idplan;
    CHECK(image_pipeline_plan(plan.resized_h, plan.resized_w, 280, &idplan));
    CHECK(idplan.resized_h == plan.resized_h && idplan.resized_w == plan.resized_w);
    if (g_fail) {
        return GEIST_TEST_FAIL;
    }

    const size_t IH  = idplan.in_h;
    const size_t IW  = idplan.in_w;
    uint8_t     *rgb = heap_alloc_array_aligned(uint8_t, IH * IW * 3);
    CHECK(rgb != nullptr);
    if (rgb == nullptr) {
        return GEIST_TEST_FAIL;
    }
    for (size_t i = 0; i < IH * IW * 3; i++) {
        rgb[i] = (uint8_t) (i * 7u);
    }

    const size_t patch_px  = 16u * 16u * 3u;
    const size_t n_patches = idplan.grid_h * idplan.grid_w;
    float       *patches   = heap_alloc_array_aligned(float, n_patches *patch_px);
    CHECK(patches != nullptr);
    if (patches == nullptr) {
        safe_free((void **) &rgb);
        return GEIST_TEST_FAIL;
    }
    CHECK(image_pipeline_preprocess(rgb, &idplan, patches));
    /* Every patch value is a rescaled byte, so the whole output must land
     * in [0, 1] — a cheap way to notice a stride or bounds mistake in the
     * patchify walk. */
    size_t out_of_range = 0;
    for (size_t i = 0; i < n_patches * patch_px; i++) {
        if (!(patches[i] >= 0.0f && patches[i] <= 1.0f)) {
            out_of_range++;
        }
    }
    if (out_of_range != 0) {
        fprintf(stderr,
                "FAIL: %zu/%zu patch values outside [0,1]\n",
                out_of_range,
                n_patches * patch_px);
        g_fail = 1;
    }

    safe_free((void **) &patches);
    safe_free((void **) &rgb);

    if (g_fail) {
        return GEIST_TEST_FAIL;
    }
    printf("PASS: image geometry — unrepresentable dimensions refused, %zux%zu still "
           "plans (%zux%zu, %zu soft tokens), and %zux%zu preprocesses cleanly\n",
           H,
           W,
           plan.resized_h,
           plan.resized_w,
           plan.soft_tokens,
           IH,
           IW);
    return GEIST_TEST_PASS;
}
