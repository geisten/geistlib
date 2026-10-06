/*
 * mel_pipeline — log-mel spectrogram (vDSP on Apple, vendored radix-2 FFT
 * elsewhere).
 *
 * Per-frame (320 fp32 PCM in → 128 fp32 log-mel out):
 *   1. windowed[i]   = pcm[i] * hann[i]                 for i in [0, 320)
 *   2. zero-pad windowed to 512
 *   3. real FFT (split-format, packed) → 257 complex bins
 *   4. magnitude    = |FFT|                              (sqrt, not square)
 *   5. mel_spec     = magnitude · mel_filters            (1×257 · 257×128)
 *   6. log_mel      = log(mel_spec + 1e-3)
 */
#include "mel_pipeline.h"
#include <geist_types.h>
#include "heap.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__APPLE__)
/* Forward-declare just the vDSP/cblas symbols we need, avoiding the
 * <Accelerate/Accelerate.h> umbrella header — it pulls in vImage →
 * CoreFoundation → /usr/local/include/conv.h on Homebrew systems,
 * which bombs on missing bimg.h. We link against -framework Accelerate. */
typedef unsigned long vDSP_Length;
typedef long          vDSP_Stride;
typedef enum { vDSP_DFT_FORWARD = 1, vDSP_DFT_INVERSE = -1 } vDSP_DFT_Direction;
typedef struct vDSP_DFT_SetupStruct *vDSP_DFT_Setup;
typedef struct DSPComplex {
    float real, imag;
} DSPComplex;
typedef struct DSPSplitComplex {
    float *realp;
    float *imagp;
} DSPSplitComplex;

extern vDSP_DFT_Setup vDSP_DFT_zrop_CreateSetup(vDSP_DFT_Setup, vDSP_Length, vDSP_DFT_Direction);
extern void           vDSP_DFT_DestroySetup(vDSP_DFT_Setup);
extern void           vDSP_DFT_Execute(const struct vDSP_DFT_SetupStruct *,
                                       const float *Ir,
                                       const float *Ii,
                                       float       *Or,
                                       float       *Oi);
extern void           vDSP_vmul(const float *A,
                                vDSP_Stride  IA,
                                const float *B,
                                vDSP_Stride  IB,
                                float       *C,
                                vDSP_Stride  IC,
                                vDSP_Length  N);
extern void           vDSP_ctoz(const DSPComplex      *C,
                                vDSP_Stride            IC,
                                const DSPSplitComplex *Z,
                                vDSP_Stride            IZ,
                                vDSP_Length            N);
#endif

#include "geist_gemm.h"

#define MEL_FLOOR 0.001f
#define HALF_N (MEL_FFT_LENGTH / 2) /* 256 */

#if !defined(__APPLE__)
/* Vendored real FFT for the non-Apple build: a small in-place iterative
 * radix-2 DIT FFT (forward, -2pi i convention; n a power of 2), keeping the
 * BLAS-free build dependency-free. Max rel-magnitude error vs a naive DFT
 * ~7e-6 at N=512. Audio is not perf-critical, so per-butterfly trig is fine. */
static void mel_fft_radix2(float *re, float *im, int n) {
    static const double MEL_PI = 3.14159265358979323846;
    int                 bits   = 0;
    while ((1 << bits) < n)
        bits++;
    for (int i = 0; i < n; i++) {
        int j = 0, x = i;
        for (int b = 0; b < bits; b++) {
            j = (j << 1) | (x & 1);
            x >>= 1;
        }
        if (j > i) {
            float t = re[i];
            re[i]   = re[j];
            re[j]   = t;
            t       = im[i];
            im[i]   = im[j];
            im[j]   = t;
        }
    }
    for (int len = 2; len <= n; len <<= 1) {
        const double ang = -2.0 * MEL_PI / len;
        for (int i = 0; i < n; i += len)
            for (int k = 0; k < len / 2; k++) {
                const double wr = cos(ang * k), wi = sin(ang * k);
                const int    a = i + k, b = a + len / 2;
                const double tr = wr * re[b] - wi * im[b];
                const double ti = wr * im[b] + wi * re[b];
                re[b]           = (float) (re[a] - tr);
                im[b]           = (float) (im[a] - ti);
                re[a]           = (float) (re[a] + tr);
                im[a]           = (float) (im[a] + ti);
            }
    }
}
#endif

struct MelState {
    /* Precomputed constants (loaded from blob). */
    float window[MEL_FRAME_LENGTH];
    float mel[MEL_N_FFT_BINS * MEL_N_MEL]; /* row-major (257, 128) */

#if defined(__APPLE__)
    vDSP_DFT_Setup fft_setup;
    float          fft_real[HALF_N]; /* split-format FFT input/output */
    float          fft_imag[HALF_N];
    float          out_real[HALF_N];
    float          out_imag[HALF_N];
#else
    float fft_re[MEL_FFT_LENGTH]; /* in-place radix-2 FFT scratch */
    float fft_im[MEL_FFT_LENGTH];
#endif

    /* Per-frame scratch (avoids per-call malloc). */
    float padded[MEL_FFT_LENGTH]; /* windowed + zero-padded PCM */
    float magnitude[MEL_N_FFT_BINS];
};

struct MelState *mel_create(const char *constants_bin_path) {
    FILE *f = fopen(constants_bin_path, "rb");
    if (!f) {
        fprintf(stderr, "mel_create: cannot open %s\n", constants_bin_path);
        return nullptr;
    }

    struct MelState *m           = heap_calloc_array_aligned(struct MelState, 1);
    size_t           want_window = MEL_FRAME_LENGTH;
    size_t           want_mel    = (size_t) MEL_N_FFT_BINS * MEL_N_MEL;
    if (fread(m->window, sizeof(float), want_window, f) != want_window ||
        fread(m->mel, sizeof(float), want_mel, f) != want_mel) {
        fprintf(stderr, "mel_create: short read on %s\n", constants_bin_path);
        safe_free((void **) &m);
        fclose(f);
        return nullptr;
    }
    fclose(f);

#if defined(__APPLE__)
    m->fft_setup = vDSP_DFT_zrop_CreateSetup(nullptr, MEL_FFT_LENGTH, vDSP_DFT_FORWARD);
    if (!m->fft_setup) {
        fprintf(stderr, "mel_create: vDSP_DFT_zrop_CreateSetup failed for N=%d\n", MEL_FFT_LENGTH);
        safe_free((void **) &m);
        return nullptr;
    }
#endif
    /* (non-Apple: the radix-2 FFT is stateless — no plan to build.) */
    return m;
}

void mel_destroy(struct MelState *m) {
    if (!m)
        return;
#if defined(__APPLE__)
    if (m->fft_setup)
        vDSP_DFT_DestroySetup(m->fft_setup);
#endif
    safe_free((void **) &m);
}

void mel_frame_compute(struct MelState *m, const float *pcm_320, float *out_mel_128) {
    /* 1+2: window first 320 samples, zero the tail. */
#if defined(__APPLE__)
    vDSP_vmul(pcm_320, 1, m->window, 1, m->padded, 1, MEL_FRAME_LENGTH);
#else
    for (int i = 0; i < MEL_FRAME_LENGTH; i++)
        m->padded[i] = pcm_320[i] * m->window[i];
#endif
    memset(m->padded + MEL_FRAME_LENGTH, 0, (MEL_FFT_LENGTH - MEL_FRAME_LENGTH) * sizeof(float));

#if defined(__APPLE__)
    /* 3a: split into even/odd lanes for vDSP_DFT_zrop input format. */
    DSPSplitComplex split_in = {.realp = m->fft_real, .imagp = m->fft_imag};
    vDSP_ctoz((const DSPComplex *) m->padded, 2, &split_in, 1, HALF_N);

    /* 3b: forward real DFT. vDSP zrop output is split-format with Nyquist
     * packed into out_imag[0]; vDSP also applies a factor-2 forward scaling
     * which we compensate via the 0.5 scale below. */
    vDSP_DFT_Execute(m->fft_setup, m->fft_real, m->fft_imag, m->out_real, m->out_imag);

    /* 4: magnitude per bin, with 0.5 scale to match np.fft.rfft convention. */
    const float scale                = 0.5f;
    m->magnitude[0]                  = fabsf(m->out_real[0]) * scale;
    m->magnitude[MEL_N_FFT_BINS - 1] = fabsf(m->out_imag[0]) * scale;
    for (int k = 1; k < HALF_N; k++) {
        float r         = m->out_real[k] * scale;
        float i         = m->out_imag[k] * scale;
        m->magnitude[k] = sqrtf(r * r + i * i);
    }
#else
    /* Radix-2 real FFT: real input, imag 0; output bins 0..N/2 (no scaling,
     * Nyquist at N/2 — same convention as np.fft.rfft). */
    for (int i = 0; i < MEL_FFT_LENGTH; i++) {
        m->fft_re[i] = m->padded[i];
        m->fft_im[i] = 0.0f;
    }
    mel_fft_radix2(m->fft_re, m->fft_im, MEL_FFT_LENGTH);
    for (int k = 0; k < MEL_N_FFT_BINS; k++) {
        const float r = m->fft_re[k], i = m->fft_im[k];
        m->magnitude[k] = sqrtf(r * r + i * i);
    }
#endif

    /* 5: mel_spec = magnitude · mel_filters
     *     (1,257) · (257,128) → (1,128). cblas_sgemv with M=128, N=257.
     *     mel is row-major (257,128) → as a matrix with leading dim 128,
     *     transposed view is (128,257). */
    geist_sgemv(GEIST_OP_T,
                MEL_N_FFT_BINS,
                MEL_N_MEL,
                1.0f,
                m->mel,
                MEL_N_MEL,
                m->magnitude,
                1,
                0.0f,
                out_mel_128,
                1);

    /* 6: log(x + floor). */
    for (int i = 0; i < MEL_N_MEL; i++) {
        out_mel_128[i] = logf(out_mel_128[i] + MEL_FLOOR);
    }
}
