/*
 * test_wav_reader_unit — audio_test_read_wav must parse WAVs by walking
 * RIFF chunks, not by skipping a fixed 44-byte header.
 *
 * The regression this pins: ffmpeg writes a LIST/INFO chunk between
 * 'fmt ' and 'data'; the old fixed-offset readers returned metadata
 * bytes as PCM and shifted the whole clip (#268 — measured as ~77 %
 * median LibriSpeech WER on a reference-exact pipeline).
 *
 * Writes two temp WAVs with identical PCM — one minimal 44-byte layout,
 * one with a LIST chunk before 'data' — and asserts both parse to the
 * same samples.
 */
#include "audio_test_util.h"
#include "test_helpers.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_SAMPLES 800 /* 50 ms */

static void put_u32(unsigned char *p, unsigned v) {
    p[0] = (unsigned char) v;
    p[1] = (unsigned char) (v >> 8);
    p[2] = (unsigned char) (v >> 16);
    p[3] = (unsigned char) (v >> 24);
}

/* Write a 16 kHz mono 16-bit WAV; with_list inserts an ffmpeg-style
 * LIST/INFO chunk between 'fmt ' and 'data'. */
static bool write_wav(const char *path, const int16_t *pcm, size_t n, bool with_list) {
    static const unsigned char list_chunk[] = {'L', 'I', 'S', 'T', 14,  0,   0,   0,
                                               'I', 'N', 'F', 'O', 'I', 'S', 'F', 'T',
                                               2,   0,   0,   0,   'x', '\0'};
    const unsigned data_bytes = (unsigned) (n * 2);
    const unsigned riff_bytes = 4 + 24 + (with_list ? (unsigned) sizeof(list_chunk) : 0) + 8 +
                                data_bytes;
    unsigned char hdr[36] = {'R', 'I', 'F', 'F', 0, 0, 0, 0, 'W', 'A', 'V', 'E',
                             'f', 'm', 't', ' ', 16, 0, 0, 0, 1, 0, 1, 0};
    put_u32(hdr + 4, riff_bytes);
    put_u32(hdr + 24, 16000);
    put_u32(hdr + 28, 32000);
    hdr[32] = 2;
    hdr[34] = 16;

    FILE *f = fopen(path, "wb");
    if (f == nullptr) {
        return false;
    }
    bool ok = fwrite(hdr, 1, sizeof hdr, f) == sizeof hdr;
    if (with_list) {
        ok = ok && fwrite(list_chunk, 1, sizeof list_chunk, f) == sizeof list_chunk;
    }
    unsigned char data_hdr[8] = {'d', 'a', 't', 'a'};
    put_u32(data_hdr + 4, data_bytes);
    ok = ok && fwrite(data_hdr, 1, sizeof data_hdr, f) == sizeof data_hdr &&
         fwrite(pcm, 2, n, f) == n;
    fclose(f);
    return ok;
}

int main(void) {
    int16_t pcm[N_SAMPLES];
    audio_test_synth_sweep(pcm, N_SAMPLES, 0.05);

    const char *plain  = "/tmp/geist_wav_plain_test.wav";
    const char *listed = "/tmp/geist_wav_list_test.wav";
    if (!write_wav(plain, pcm, N_SAMPLES, false) || !write_wav(listed, pcm, N_SAMPLES, true)) {
        return GEIST_TEST_ERROR;
    }

    int fails = 0;
    for (int variant = 0; variant < 2; variant++) {
        const char *path = variant == 0 ? plain : listed;
        size_t      n    = 0;
        int         sr   = 0;
        int16_t    *got  = audio_test_read_wav(path, &n, &sr);
        if (got == nullptr || n != N_SAMPLES || sr != 16000) {
            fprintf(stderr, "FAIL: %s parsed as n=%zu sr=%d\n", path, n, sr);
            fails++;
        } else if (memcmp(got, pcm, sizeof pcm) != 0) {
            fprintf(stderr, "FAIL: %s PCM mismatch\n", path);
            fails++;
        }
        free(got);
    }

    /* Truncated/garbage input must refuse, not crash. */
    size_t   n  = 0;
    int      sr = 0;
    int16_t *p  = audio_test_read_wav("/dev/null", &n, &sr);
    if (p != nullptr) {
        fprintf(stderr, "FAIL: /dev/null accepted\n");
        free(p);
        fails++;
    }

    remove(plain);
    remove(listed);
    printf(fails == 0 ? "PASS\n" : "FAIL (%d)\n", fails);
    return fails == 0 ? GEIST_TEST_PASS : GEIST_TEST_FAIL;
}
