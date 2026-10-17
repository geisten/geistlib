/* js_streams A.bin B.bin n_vocab [per_step.txt] — two streams of logit rows
 * (float32, n_vocab each, as tools/dump_geist_tf_logits and
 * benchmark/dump_llamacpp_tf.cpp write them) compared row by row: the
 * Jensen-Shannon divergence of the two softmaxes (nats) and whether the
 * argmax agrees. Prints the distribution (mean, median, p90, p99, max,
 * share above 0.02) and, with a 4th argument, one "step js top1_same" line
 * per row.
 *
 *   cc -O2 -o js_streams tools/js_streams.c -lm
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void softmax(size_t n, const float *x, double *p) {
    double mx = x[0], z = 0.0;
    for (size_t i = 1; i < n; i++)
        mx = x[i] > mx ? x[i] : mx;
    for (size_t i = 0; i < n; i++)
        z += (p[i] = exp((double) x[i] - mx));
    for (size_t i = 0; i < n; i++)
        p[i] /= z;
}

static int cmp(const void *a, const void *b) {
    const double x = *(const double *) a, y = *(const double *) b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv) {
    if (argc < 4)
        return 2;
    const size_t V  = (size_t) atol(argv[3]);
    FILE        *fa = fopen(argv[1], "rb"), *fb = fopen(argv[2], "rb");
    FILE        *fo = argc > 4 ? fopen(argv[4], "w") : nullptr;
    float       *a = malloc(V * sizeof *a), *b = malloc(V * sizeof *b);
    double      *pa = malloc(V * sizeof *pa), *pb = malloc(V * sizeof *pb);
    double      *js = nullptr;
    size_t       n = 0, cap = 0, same = 0;
    if (fa == nullptr || fb == nullptr || a == nullptr || b == nullptr || pa == nullptr ||
        pb == nullptr)
        return 1;
    while (fread(a, sizeof *a, V, fa) == V && fread(b, sizeof *b, V, fb) == V) {
        softmax(V, a, pa);
        softmax(V, b, pb);
        double d  = 0.0;
        size_t ia = 0, ib = 0;
        for (size_t i = 0; i < V; i++) {
            const double m = 0.5 * (pa[i] + pb[i]);
            if (pa[i] > 0.0)
                d += 0.5 * pa[i] * log(pa[i] / m);
            if (pb[i] > 0.0)
                d += 0.5 * pb[i] * log(pb[i] / m);
            ia = a[i] > a[ia] ? i : ia;
            ib = b[i] > b[ib] ? i : ib;
        }
        if (n == cap) {
            cap = cap ? 2 * cap : 256;
            js  = realloc(js, cap * sizeof *js);
        }
        js[n] = d;
        same += ia == ib;
        if (fo != nullptr)
            fprintf(fo, "%zu %.6f %d\n", n, d, ia == ib);
        n++;
    }
    if (n == 0)
        return 1;
    double sum  = 0.0;
    size_t over = 0;
    for (size_t i = 0; i < n; i++) {
        sum += js[i];
        over += js[i] > 0.02;
    }
    qsort(js, n, sizeof *js, cmp);
    printf("n=%zu mean=%.5f median=%.5f p90=%.5f p99=%.5f max=%.5f over0.02=%.1f%% "
           "top1_same=%.1f%%\n",
           n,
           sum / (double) n,
           js[n / 2],
           js[(n * 9) / 10],
           js[(n * 99) / 100],
           js[n - 1],
           100.0 * (double) over / (double) n,
           100.0 * (double) same / (double) n);
    if (fo != nullptr)
        fclose(fo);
    return 0;
}
