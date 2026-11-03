// verify-cost probe (#401): decode_step vs verify_drafts(k) on one session.
#include <geist.h>
#include <geist_util.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now_ms(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e3+t.tv_nsec/1e6;}
static unsigned rs=12345;static int rnd(void){rs=rs*1103515245u+12345u;return 1000+(int)((rs>>8)%30000u);}
int main(int argc,char**argv){
  const char*path=argv[1];int rounds=argc>2?atoi(argv[2]):3;int kmax=argc>3?atoi(argv[3]):4;
  setenv("GEIST_TEXT_ONLY","1",0);
  struct geist_backend*be=NULL;if(geist_backend_create(getenv("GEIST_BENCH_BACKEND")?getenv("GEIST_BENCH_BACKEND"):"metal",NULL,NULL,&be)!=GEIST_OK){fprintf(stderr,"be\n");return 1;}
  struct geist_session_opts lo={0};lo.max_seq_len=2048;
  struct geist_model*m=NULL;if(geist_model_load_with_opts(path,be,&lo,&m)!=GEIST_OK){fprintf(stderr,"load\n");return 1;}
  struct geist_session_opts o={.max_seq_len=2048,.random_seed=1};
  struct geist_session*s=NULL;if(geist_session_create(m,be,&o,&s)!=GEIST_OK){fprintf(stderr,"sess\n");return 1;}
  geist_token_t ids[512];for(int i=0;i<512;i++)ids[i]=rnd();
  if(geist_session_prefill_tokens(s,512,ids)!=GEIST_OK){fprintf(stderr,"prefill\n");return 1;}
  geist_token_t t;for(int i=0;i<4;i++)(void)geist_session_decode_step(s,&t);
  geist_token_t out[32],dr[16];size_t n;
  for(int k=2;k<=kmax;k++){for(int j=0;j<k-1;j++)dr[j]=rnd();(void)geist_session_verify_drafts(s,k-1,dr,32,out,&n);} // warm pipelines
  for(int r=0;r<rounds;r++){
    double t0=now_ms();for(int i=0;i<6;i++)(void)geist_session_decode_step(s,&t);double dec=(now_ms()-t0)/6;
    printf("round %d decode_step %.2f ms\n",r,dec);fflush(stdout);
    for(int k=2;k<=kmax;k++){for(int rep=0;rep<3;rep++){for(int j=0;j<k-1;j++)dr[j]=rnd();
      fprintf(stderr,"K%d ",k);double a=now_ms();(void)geist_session_verify_drafts(s,k-1,dr,32,out,&n);
      printf("round %d k %d call %.2f\n",r,k,now_ms()-a);}}
  }
  geist_session_destroy(s);geist_model_destroy(m);geist_backend_destroy(be);return 0;}
