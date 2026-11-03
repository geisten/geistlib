// Metal quant linear microbench (#401): rows 1..8, GPU time per call.
#include <geist.h>
#include <geist_backend.h>
#include <geist_weight.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now_ms(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e3+t.tv_nsec/1e6;}
static unsigned rs=1;static unsigned char r8(void){rs=rs*1664525u+1013904223u;return rs>>24;}
static void*dev(struct geist_backend*be,size_t b,struct geist_buffer**o){if(be->desc->vtbl->buffer_create(be,b,GEIST_BUFFER_SCRATCH,GEIST_MEMORY_AUTO,o)!=GEIST_OK)return NULL;return be->desc->vtbl->buffer_map(*o);}
int main(int argc,char**argv){
  int dtype=!strcmp(argv[1],"q40")?GEIST_DTYPE_Q4_0:!strcmp(argv[1],"q41")?GEIST_DTYPE_Q4_1:!strcmp(argv[1],"q80")?GEIST_DTYPE_Q8_0:!strcmp(argv[1],"q5k")?GEIST_DTYPE_Q5_K:!strcmp(argv[1],"q4k")?GEIST_DTYPE_Q4_K:GEIST_DTYPE_Q6_K;size_t n_in=atol(argv[2]),n_out=atol(argv[3]);int bb=atoi(argv[4]),be_el=atoi(argv[5]);int reps=argc>6?atoi(argv[6]):50;
  struct geist_backend*be;if(geist_backend_create("metal",NULL,NULL,&be)!=GEIST_OK)return 1;
  const struct geist_backend_vtbl*v=be->desc->vtbl;const struct geist_backend_fused*f=geist_backend_fused_tbl(be);
  struct geist_buffer*bx,*bw,*by;float*x=dev(be,8*n_in*4,&bx);unsigned char*w=dev(be,n_out*(n_in/be_el)*bb,&bw);dev(be,8*n_out*4,&by);
  for(size_t i=0;i<8*n_in;i++)x[i]=((float)r8()-127.5f)*0.01f;
  for(size_t i=0;i<n_out*(n_in/be_el)*bb;i++)w[i]=r8();
  for(size_t b=0;b<n_out*(n_in/be_el);b++){w[b*bb]=r8();w[b*bb+1]=0x20|(r8()&3);}
  struct geist_tensor tw={.buffer=bw,.dtype=dtype,.layout=GEIST_LAYOUT_BLOCK_QUANTIZED,.ndim=2,.shape={n_out,n_in}};
  struct geist_weight ww={.n_in=n_in,.n_out=n_out,.dtype=dtype};
  float tmp[4];
  for(int rows=1;rows<=8;rows++){
    struct geist_tensor tx={.buffer=bx,.dtype=GEIST_DTYPE_F32,.layout=GEIST_LAYOUT_DENSE,.ndim=2,.shape={rows,n_in},.stride={n_in,1}};
    struct geist_tensor ty={.buffer=by,.dtype=GEIST_DTYPE_F32,.layout=GEIST_LAYOUT_DENSE,.ndim=2,.shape={rows,n_out},.stride={n_out,1}};
    double best=1e9;
    for(int t=0;t<5;t++){
      int tok=v->parallel_region_begin(be,GEIST_REGION_DECODE_STEP);
      double a=now_ms();
      for(int i=0;i<reps;i++) if(f->linear_t(be,&tx,&ww,&tw,rows,&ty)!=GEIST_OK){fprintf(stderr,"fail rows %d: %s\n",rows,geist_backend_errmsg(be));return 1;}
      v->parallel_region_end(be,tok);
      (void)v->buffer_download(sizeof tmp,(unsigned char*)tmp,by);
      double ms=(now_ms()-a)/reps; if(ms<best)best=ms;}
    printf("dtype %d %zux%zu rows %d: %.3f ms/call (%.2f x rows1) GB/s %.0f\n",dtype,n_in,n_out,rows,best,0.0,(double)n_out*(n_in/be_el)*bb/best/1e6);
  }
  return 0;}
