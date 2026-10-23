/* metal_shaders.h — the embedded MSL kernel sources for the Metal backend.
 *
 * Pure data: every symbol is a `static const char[]` shader string compiled
 * at run time. Included from metal_internal.h, after the param structs. */
#ifndef GEIST_METAL_SHADERS_H
#define GEIST_METAL_SHADERS_H

static const char metal_q4k_source[] =
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "struct Params{uint "
        "n_in,n_out,rows,blocks_per_row,x_offset,w_byte_offset,y_offset,x_row_stride,y_row_stride;}"
        ";\n"
        "static inline float load_f16(device const uchar*w,uint o){ushort "
        "b=ushort(w[o])|(ushort(w[o+1])<<8);return float(as_type<half>(b));}\n"
        "static inline void get_scale_min(uint j,device const uchar*q,thread uint&s,thread "
        "uint&m){if(j<4){s=uint(q[j])&63u;m=uint(q[j+4])&63u;}else{uint "
        "a=uint(q[j+4]),b=uint(q[j-4]),c=uint(q[j]);s=(a&15u)|((b>>6u)<<4u);m=(a>>4u)|((c>>6u)<<4u)"
        ";}}\n"
        "static inline float deq(device const uchar*w,constant Params&p,uint row,uint k){uint "
        "br=k/256u,ib=k-br*256u,sub=ib/32u,idx=ib-sub*32u;uint "
        "bo=p.w_byte_offset+(row*p.blocks_per_row+br)*144u;float "
        "d=load_f16(w,bo),dm=load_f16(w,bo+2u);uint s,m;get_scale_min(sub,w+bo+4u,s,m);uint "
        "qb=uint(w[bo+16u+(sub/2u)*32u+idx]);uint q=(sub&1u)==0u?(qb&15u):(qb>>4u);return "
        "d*float(s)*float(q)-dm*float(m);}\n"
        "kernel void matmul_q4k_m8(device const float *x [[buffer(0)]],\n"
        "                         device const uchar *w [[buffer(1)]],\n"
        "                         device float *y [[buffer(2)]],\n"
        "                         constant Params &p [[buffer(3)]],\n"
        "                         uint3 tg [[threadgroup_position_in_grid]],\n"
        "                         uint lid [[thread_index_in_threadgroup]]) {\n"
        "    threadgroup float partial[8][256];\n"
        "    uint row = tg.x;\n"
        "    uint batch_base = tg.y * 8u;\n"
        "    if (row >= p.n_out || batch_base >= p.rows) { return; }\n"
        "    float sum[8];\n"
        "    for (uint m = 0u; m < 8u; m++) { sum[m] = 0.0f; }\n"
        "    for (uint k = lid; k < p.n_in; k += 256u) {\n"
        "        float wv = deq(w,p,row,k);\n"
        "        for (uint m = 0u; m < 8u; m++) {\n"
        "            uint batch = batch_base + m;\n"
        "            if (batch < p.rows) {\n"
        "                sum[m] += x[p.x_offset + batch * p.x_row_stride + k] * wv;\n"
        "            }\n"
        "        }\n"
        "    }\n"
        "    for (uint m = 0u; m < 8u; m++) { partial[m][lid] = sum[m]; }\n"
        "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "    for (uint stride = 128u; stride > 0u; stride >>= 1u) {\n"
        "        if (lid < stride) {\n"
        "            for (uint m = 0u; m < 8u; m++) {\n"
        "                partial[m][lid] += partial[m][lid + stride];\n"
        "            }\n"
        "        }\n"
        "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "    }\n"
        "    if (lid == 0u) {\n"
        "        for (uint m = 0u; m < 8u; m++) {\n"
        "            uint batch = batch_base + m;\n"
        "            if (batch < p.rows) {\n"
        "                y[p.y_offset + batch * p.y_row_stride + row] = partial[m][0];\n"
        "            }\n"
        "        }\n"
        "    }\n"
        "}\n";

/* Shared 64x32 simdgroup GEMM template (matmul_q4k_mm_sg tile structure,
 * bounds-checked variant). Instantiated per quant format by splicing in the
 * block struct name BLK, a 16-element dequant function DQ (device const
 * BLK*, short il, thread half4x4&) and QKNL = 16-element chunks per block,
 * all defined in the surrounding shader source. The staging bounds check
 * uses il0, the k position inside the tile (il drifts ahead of it across a
 * block). Unlike the q4k kernel, the partial-tile epilogue copies scalars
 * only: the float4 path needs a 16-byte-aligned y row, which arbitrary
 * n_out does not give. */
#define GEIST_METAL_MM_SG_KERNEL(NAME, BLK, DQ, QKNL)                                         \
    "kernel void matmul_" NAME "_mm_sg(device const float*x[[buffer(0)]],device const "       \
    "uchar*w[[buffer(1)]],device float*y[[buffer(2)]],constant P&p[[buffer(3)]],threadgroup " \
    "char*shmem[[threadgroup(0)]],uint3 tg[[threadgroup_position_in_grid]],ushort "           \
    "ti[[thread_index_in_threadgroup]],ushort sg[[simdgroup_index_in_threadgroup]]){\n"       \
    "threadgroup half*sa=(threadgroup half*)shmem;threadgroup half*sb=(threadgroup "          \
    "half*)(shmem+4096);constexpr short NR0=64;constexpr short NR1=32;constexpr short "       \
    "NK=32;constexpr short NL0=2;constexpr short NL1=4;constexpr short QK_NL=" QKNL ";uint "  \
    "r0=tg.y*uint(NR0);uint r1=tg.x*uint(NR1);if(r0>=p.no||r1>=p.rows)return;short "          \
    "nr0=short(min(p.no-r0,uint(NR0)));short nr1=short(min(p.rows-r1,uint(NR1)));short "      \
    "lr0=min(short(ti/NL0),short(nr0-1));short lr1=min(short(ti/NL1),short(nr1-1));short "    \
    "il0=short(ti%NL0);short il=il0;device const " BLK "*wp=(device const " BLK               \
    "*)(w+p.wo)+(r0+uint(lr0))*p.bpr;short iy=short(8*(ti%NL1));device const "                \
    "float*xp=x+p.xo+(r1+uint(lr1))*p.xs+uint(iy);simdgroup_half8x8 ma[4];simdgroup_half8x8 " \
    "mb[2];simdgroup_float8x8 mc[8];FOR_UNROLL(short "                                        \
    "i=0;i<8;i++){mc[i]=make_filled_simdgroup_matrix<float,8>(0.0f);}for(uint "               \
    "loop_k=0u;loop_k<p.ni;loop_k+=uint(NK)){half4x4 ta;" DQ "(wp,il,ta);"                    \
    "threadgroup_barrier(mem_flags::mem_threadgroup);FOR_UNROLL(short "                       \
    "i=0;i<16;i++){short sx=short(2*il0+i/8);short sy=short((ti/NL0)/8);short "               \
    "lx=short((ti/NL0)%8);short ly=short(i%8);short "                                         \
    "ib=short(8*sx+sy);*(sa+64*ib+8*ly+lx)=(loop_k+uint(16*il0+i)<p.ni)?ta[i/"                \
    "4][i%4]:half(0.0);}short sx=short(ti%NL1);short sy=short((ti/NL1)/8);short "             \
    "ly=short((ti/NL1)%8);short ib=short(4*sx+sy);FOR_UNROLL(short i=0;i<8;i++){uint "        \
    "kk=loop_k+uint(iy+i);*(sb+64*ib+8*ly+i)=kk<p.ni?half(xp[i]):half(0.0);}"                 \
    "il=(il+2<QK_NL)?il+2:il%2;wp=(il<2)?wp+1:wp;xp+=NK;"                                     \
    "threadgroup_barrier(mem_flags::mem_threadgroup);"                                        \
    "threadgroup const half*lsma=sa+4*64*(sg%2);threadgroup const "                           \
    "half*lsmb=sb+2*64*(sg/2);FOR_UNROLL(short "                                              \
    "ik=0;ik<NK/8;ik++){simdgroup_barrier(mem_flags::mem_none);FOR_UNROLL(short "             \
    "i=0;i<4;i++){simdgroup_load(ma[i],lsma+64*i,8,0,false);}simdgroup_barrier(mem_flags::"   \
    "mem_none);FOR_UNROLL(short "                                                             \
    "i=0;i<2;i++){simdgroup_load(mb[i],lsmb+64*i,8,0,false);}simdgroup_barrier(mem_flags::"   \
    "mem_none);FOR_UNROLL(short "                                                             \
    "i=0;i<8;i++){simdgroup_multiply_accumulate(mc[i],mb[i/"                                  \
    "4],ma[i%4],mc[i]);}lsma+=8*64;lsmb+=4*64;}}\n"                                           \
    "if(r0+uint(NR0)<=p.no&&r1+uint(NR1)<=p.rows){device "                                    \
    "float*C=y+p.yo+(r1+16u*uint(sg>>1))*p.ys+r0+32u*uint(sg&1);FOR_UNROLL(short "            \
    "i=0;i<8;i++){simdgroup_store(mc[i],C+8*(i%4)+8*p.ys*(i/"                                 \
    "4),p.ys,0,false);}}else{threadgroup_barrier(mem_flags::mem_threadgroup);threadgroup "    \
    "float*tmp=((threadgroup float*)shmem)+32*(sg&1)+16*(sg>>1)*NR0;FOR_UNROLL(short "        \
    "i=0;i<8;i++){simdgroup_store(mc[i],tmp+8*(i%4)+8*NR0*(i/"                                \
    "4),NR0,0,false);}threadgroup_barrier(mem_flags::mem_threadgroup);if(sg==0){for(uint "    \
    "j=ti;j<uint(nr1);j+=uint(NR1)){device float*D=y+p.yo+(r1+j)*p.ys+r0;threadgroup "        \
    "float*C=((threadgroup float*)shmem)+j*NR0;for(uint "                                     \
    "i=0u;i<uint(nr0);i++){D[i]=C[i];}}}}\n"                                                  \
    "}\n"

/* Interior fast path of the template above (matmul_q4k_mm_sg_fast
 * structure): no bounds checks, vectorized 8-wide activation staging,
 * direct simdgroup_store output. Dispatched only when rows%32==0,
 * n_out%64==0, n_in%32==0 and x is 8-float aligned. */
#define GEIST_METAL_MM_SG_FAST_KERNEL(NAME, BLK, DQ, QKNL)                                        \
    "kernel void matmul_" NAME "_mm_sg_fast(device const float*x[[buffer(0)]],device const "      \
    "uchar*w[[buffer(1)]],device float*y[[buffer(2)]],constant P&p[[buffer(3)]],threadgroup "     \
    "char*shmem[[threadgroup(0)]],uint3 tg[[threadgroup_position_in_grid]],ushort "               \
    "ti[[thread_index_in_threadgroup]],ushort sg[[simdgroup_index_in_threadgroup]]){\n"           \
    "threadgroup half*sa=(threadgroup half*)shmem;threadgroup half*sb=(threadgroup "              \
    "half*)(shmem+4096);"                                                                         \
    "constexpr short NK=32;constexpr short NL0=2;constexpr short NL1=4;constexpr short "          \
    "QK_NL=" QKNL ";"                                                                             \
    "uint r0=tg.y*64u;uint r1=tg.x*32u;"                                                          \
    "short lr0=short(ti/NL0);short lr1=short(ti/NL1);short il0=short(ti%NL0);short il=il0;"       \
    "device const " BLK "*wp=(device const " BLK "*)(w+p.wo)+(r0+uint(lr0))*p.bpr;"               \
    "short iy=short(8*(ti%NL1));"                                                                 \
    "device const float*xp=x+p.xo+min(r1+uint(lr1),p.rows-1u)*p.xs+uint(iy);"                     \
    "simdgroup_half8x8 ma[4];simdgroup_half8x8 mb[2];simdgroup_float8x8 mc[8];"                   \
    "FOR_UNROLL(short i=0;i<8;i++){mc[i]=make_filled_simdgroup_matrix<float,8>(0.0f);}"           \
    "short sx=short(ti%NL1);short sy=short((ti/NL1)/8);short lyb=short((ti/NL1)%8);short "        \
    "ibb=short(4*sx+sy);"                                                                         \
    "for(uint loop_k=0u;loop_k<p.ni;loop_k+=uint(NK)){"                                           \
    "half4x4 ta;" DQ "(wp,il,ta);"                                                                \
    "threadgroup_barrier(mem_flags::mem_threadgroup);"                                            \
    "FOR_UNROLL(short i=0;i<16;i++){short ax=short(2*il0+i/8);short ay=short((ti/NL0)/8);"        \
    "short lx=short((ti/NL0)%8);short ly=short(i%8);short "                                       \
    "ib=short(8*ax+ay);*(sa+64*ib+8*ly+lx)=ta[i/4][i%4];}"                                        \
    "*(threadgroup half2x4*)(sb+64*ibb+8*lyb)=half2x4(*((device const float2x4*)xp));"            \
    "il=(il+2<QK_NL)?il+2:il%2;wp=(il<2)?wp+1:wp;xp+=NK;"                                         \
    "threadgroup_barrier(mem_flags::mem_threadgroup);"                                            \
    "threadgroup const half*lsma=sa+4*64*(sg%2);threadgroup const half*lsmb=sb+2*64*(sg/2);"      \
    "FOR_UNROLL(short ik=0;ik<NK/8;ik++){"                                                        \
    "simdgroup_barrier(mem_flags::mem_none);"                                                     \
    "FOR_UNROLL(short i=0;i<4;i++){simdgroup_load(ma[i],lsma+64*i,8,0,false);}"                   \
    "simdgroup_barrier(mem_flags::mem_none);"                                                     \
    "FOR_UNROLL(short i=0;i<2;i++){simdgroup_load(mb[i],lsmb+64*i,8,0,false);}"                   \
    "simdgroup_barrier(mem_flags::mem_none);"                                                     \
    "FOR_UNROLL(short i=0;i<8;i++){simdgroup_multiply_accumulate(mc[i],mb[i/4],ma[i%4],"          \
    "mc[i]);}"                                                                                    \
    "lsma+=8*64;lsmb+=4*64;}}"                                                                    \
    "if(r1+32u<=p.rows){device float*C=y+p.yo+(r1+16u*uint(sg>>1))*p.ys+r0+32u*uint(sg&1);"       \
    "FOR_UNROLL(short i=0;i<8;i++){simdgroup_store(mc[i],C+8*(i%4)+8*p.ys*(i/4),p.ys,0,false);}}" \
    "else{threadgroup_barrier(mem_flags::mem_threadgroup);threadgroup float*T=(threadgroup "      \
    "float*)shmem;"                                                                               \
    "FOR_UNROLL(short "                                                                           \
    "i=0;i<8;i++){simdgroup_store(mc[i],T+32*(sg&1)+1024*(sg>>1)+8*(i%4)+512*(i/4),64,0,false);}" \
    "threadgroup_barrier(mem_flags::mem_threadgroup);for(uint "                                   \
    "j=uint(ti);j<(p.rows-r1)*64u;j+=128u)"                                                       \
    "{y[p.yo+(r1+j/64u)*p.ys+r0+j%64u]=T[j];}}"                                                   \
    "}\n"

/* Correctness-first row-major Q4_0/Q8_0 linear kernels. Both formats use
 * 32-element blocks with an fp16 scale followed by packed nibbles or int8s.
 * One threadgroup computes one output row for one activation row. */
static const char metal_q40_q80_source[] =
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "struct P{uint ni,no,rows,bpr,xo,wo,yo,xs,ys;};\n"
        "static inline float h(device const uchar*w,uint o){ushort "
        "b=ushort(w[o])|(ushort(w[o+1])<<8);return float(as_type<half>(b));}\n"
        "kernel void linear_q40(device const float*x[[buffer(0)]],device const "
        "uchar*w[[buffer(1)]],device float*y[[buffer(2)]],constant P&p[[buffer(3)]],uint3 "
        "tg[[threadgroup_position_in_grid]],uint lid[[thread_index_in_threadgroup]]){threadgroup "
        "float part[256];uint r=tg.x,b=tg.y;if(r>=p.no||b>=p.rows)return;float s=0.0f;for(uint "
        "k=lid;k<p.ni;k+=256u){uint bi=k>>5u,j=k&31u,bo=p.wo+(r*p.bpr+bi)*18u;uchar "
        "q=w[bo+2u+(j&15u)];int "
        "v=int(j<16u?(q&15u):(q>>4u))-8;s+=x[p.xo+b*p.xs+k]*h(w,bo)*float(v);}part[lid]=s;"
        "threadgroup_barrier(mem_flags::mem_threadgroup);for(uint "
        "z=128u;z>0u;z>>=1u){if(lid<z)part[lid]+=part[lid+z];threadgroup_barrier(mem_flags::mem_"
        "threadgroup);}if(lid==0u)y[p.yo+b*p.ys+r]=part[0];}\n"
        "kernel void linear_q80(device const float*x[[buffer(0)]],device const "
        "uchar*w[[buffer(1)]],device float*y[[buffer(2)]],constant P&p[[buffer(3)]],uint3 "
        "tg[[threadgroup_position_in_grid]],uint lid[[thread_index_in_threadgroup]]){threadgroup "
        "float part[256];uint r=tg.x,b=tg.y;if(r>=p.no||b>=p.rows)return;float s=0.0f;for(uint "
        "k=lid;k<p.ni;k+=256u){uint bi=k>>5u,j=k&31u,bo=p.wo+(r*p.bpr+bi)*34u;char "
        "q=as_type<char>(w[bo+2u+j]);s+=x[p.xo+b*p.xs+k]*h(w,bo)*float(q);}part[lid]=s;threadgroup_"
        "barrier(mem_flags::mem_threadgroup);for(uint "
        "z=128u;z>0u;z>>=1u){if(lid<z)part[lid]+=part[lid+z];threadgroup_barrier(mem_flags::mem_"
        "threadgroup);}if(lid==0u)y[p.yo+b*p.ys+r]=part[0];}\n"
        "kernel void linear_q40_m8(device const float*x[[buffer(0)]],device const "
        "uchar*w[[buffer(1)]],device float*y[[buffer(2)]],constant P&p[[buffer(3)]],uint3 "
        "tg[[threadgroup_position_in_grid]],uint lid[[thread_index_in_threadgroup]]){threadgroup "
        "float part[8][256];uint r=tg.x,bb=tg.y*8u;if(r>=p.no||bb>=p.rows)return;float "
        "s[8];for(uint m=0u;m<8u;m++)s[m]=0.0f;for(uint k=lid;k<p.ni;k+=256u){uint "
        "bi=k>>5u,j=k&31u,bo=p.wo+(r*p.bpr+bi)*18u;uchar q=w[bo+2u+(j&15u)];int "
        "v=int(j<16u?(q&15u):(q>>4u))-8;float wv=h(w,bo)*float(v);for(uint "
        "m=0u;m<8u;m++){uint b=bb+m;if(b<p.rows)s[m]+=x[p.xo+b*p.xs+k]*wv;}}for(uint "
        "m=0u;m<8u;m++)part[m][lid]=s[m];threadgroup_barrier(mem_flags::mem_threadgroup);for(uint "
        "z=128u;z>0u;z>>=1u){if(lid<z){for(uint m=0u;m<8u;m++)part[m][lid]+=part[m][lid+z];}"
        "threadgroup_barrier(mem_flags::mem_threadgroup);}if(lid==0u){for(uint "
        "m=0u;m<8u;m++){uint b=bb+m;if(b<p.rows)y[p.yo+b*p.ys+r]=part[m][0];}}}\n"
        "kernel void linear_q80_m8(device const float*x[[buffer(0)]],device const "
        "uchar*w[[buffer(1)]],device float*y[[buffer(2)]],constant P&p[[buffer(3)]],uint3 "
        "tg[[threadgroup_position_in_grid]],uint lid[[thread_index_in_threadgroup]]){threadgroup "
        "float part[8][256];uint r=tg.x,bb=tg.y*8u;if(r>=p.no||bb>=p.rows)return;float "
        "s[8];for(uint m=0u;m<8u;m++)s[m]=0.0f;for(uint k=lid;k<p.ni;k+=256u){uint "
        "bi=k>>5u,j=k&31u,bo=p.wo+(r*p.bpr+bi)*34u;char q=as_type<char>(w[bo+2u+j]);float "
        "wv=h(w,bo)*float(q);for(uint m=0u;m<8u;m++){uint "
        "b=bb+m;if(b<p.rows)s[m]+=x[p.xo+b*p.xs+k]*wv;}}for(uint "
        "m=0u;m<8u;m++)part[m][lid]=s[m];threadgroup_barrier(mem_flags::mem_threadgroup);for(uint "
        "z=128u;z>0u;z>>=1u){if(lid<z){for(uint m=0u;m<8u;m++)part[m][lid]+=part[m][lid+z];}"
        "threadgroup_barrier(mem_flags::mem_threadgroup);}if(lid==0u){for(uint "
        "m=0u;m<8u;m++){uint b=bb+m;if(b<p.rows)y[p.yo+b*p.ys+r]=part[m][0];}}}\n";

/* Simdgroup quant kernels, one MSL unit built by concatenating the
 * metal_qsg_* arrays at library-create time (each C literal stays under the
 * 4095-char ISO limit). Common part: FOR_UNROLL, params struct, the q5_K
 * scale/min decoder and the per-format block structs + 16-element dequant
 * chunk loaders used by the shared GEMM template. */
static const char metal_qsg_common_source[] =
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "#define FOR_UNROLL(x) _Pragma(\"clang loop unroll(full)\") for(x)\n"
        "struct P{uint ni,no,rows,bpr,xo,wo,yo,xs,ys;};\n"
        "static inline void sm5(uint j,device const uchar*q,thread uint&s,thread uint&m){if(j<4u){"
        "s=uint(q[j])&63u;m=uint(q[j+4u])&63u;}else{uint a=uint(q[j+4u]),b=uint(q[j-4u]),"
        "c=uint(q[j]);s=(a&15u)|((b>>6u)<<4u);m=(a>>4u)|((c>>6u)<<4u);}}\n"
        "struct bq40{half d;uchar qs[16];};\n"
        "static inline void dq40(device const bq40*xb,short il,thread half4x4&r){"
        "device const ushort*qs=(device const ushort*)(xb->qs);"
        "float d1=il?float(xb->d)*(1.0f/16.0f):float(xb->d);float d2=d1*(1.0f/256.0f);"
        "float md=-8.0f*float(xb->d);"
        "ushort m0=il?0x00F0:0x000F;ushort m1=m0<<8;"
        "FOR_UNROLL(short i=0;i<8;i++){"
        "r[i/2][2*(i%2)+0]=half(d1*float(qs[i]&m0)+md);"
        "r[i/2][2*(i%2)+1]=half(d2*float(qs[i]&m1)+md);}}\n"
        "struct bq80{half d;char qs[32];};\n"
        "static inline void dq80(device const bq80*xb,short il,thread half4x4&r){"
        "float d=float(xb->d);device const char*q=xb->qs+16*il;"
        "FOR_UNROLL(short i=0;i<16;i++){r[i/4][i%4]=half(d*float(q[i]));}}\n"
        "struct bq41{half d;half m;uchar qs[16];};\n"
        "static inline void dq41(device const bq41*xb,short il,thread half4x4&r){"
        "device const ushort*qs=(device const ushort*)(xb->qs);"
        "float d1=il?float(xb->d)*(1.0f/16.0f):float(xb->d);float d2=d1*(1.0f/256.0f);"
        "float ml=float(xb->m);"
        "ushort m0=il?0x00F0:0x000F;ushort m1=m0<<8;"
        "FOR_UNROLL(short i=0;i<8;i++){"
        "r[i/2][2*(i%2)+0]=half(d1*float(qs[i]&m0)+ml);"
        "r[i/2][2*(i%2)+1]=half(d2*float(qs[i]&m1)+ml);}}\n"
        "constant float kiq4[16]={-127.0f,-104.0f,-83.0f,-65.0f,-49.0f,-35.0f,-22.0f,"
        "-10.0f,1.0f,13.0f,25.0f,38.0f,53.0f,69.0f,89.0f,113.0f};\n"
        "struct biq4nl{half d;uchar qs[16];};\n"
        "static inline void dqiq4nl(device const biq4nl*xb,short il,thread half4x4&r){"
        "float d=float(xb->d);ushort sh=ushort(4*il);"
        "FOR_UNROLL(short i=0;i<16;i++){"
        "r[i/4][i%4]=half(d*kiq4[(xb->qs[i]>>sh)&15]);}}\n"
        "struct biq4xs{half d;ushort sh;uchar sl[4];uchar qs[128];};\n"
        "static inline void dqiq4xs(device const biq4xs*xb,short il,thread half4x4&r){"
        "short b32=il/2;ushort nsh=ushort(4*(il&1));"
        "uint ls=(uint(xb->sl[b32/2])>>(4u*uint(b32&1))&15u)|"
        "(((uint(xb->sh)>>(2u*uint(b32)))&3u)<<4u);"
        "float dl=float(xb->d)*(float(ls)-32.0f);"
        "device const uchar*q=xb->qs+16*b32;"
        "FOR_UNROLL(short i=0;i<16;i++){"
        "r[i/4][i%4]=half(dl*kiq4[(q[i]>>nsh)&15]);}}\n"
        "struct bq5k{half d;half dmin;uchar scales[12];uchar qh[32];uchar qs[128];};\n"
        "static inline void dq5k(device const bq5k*xb,short il,thread half4x4&r){"
        "short sb=il/4;short lo16=il&1;short hi=(il&3)/2;"
        "device const uchar*q=xb->qs+sb*32+lo16*16;"
        "device const uchar*qh=xb->qh+lo16*16;"
        "uint j=uint(sb*2+hi);uint s,m;sm5(j,xb->scales,s,m);"
        "float dl=float(xb->d)*float(s);float ml=float(xb->dmin)*float(m);"
        "ushort sh=hi?4:0;uchar hb=uchar(1u<<j);"
        "FOR_UNROLL(short i=0;i<16;i++){uint v=(uint(q[i])>>sh)&15u;"
        "if((qh[i]&hb)!=0)v+=16u;r[i/4][i%4]=half(dl*float(v)-ml);}}\n";

/* Simdgroup decode GEMVs, llama mul_mv structure: 2 simdgroups per
 * threadgroup, 2 rows each. Q4_0/Q4_1: a thread covers 8 q bytes (elements
 * il..il+7 low nibbles, il+16..il+23 high) with pre-scaled y so the masked
 * ushort products need no shifts; the block offset folds into sumy. Q5_K: a
 * thread owns one 8-element run of every 256-block, so d/dmin and the 6-bit
 * scale pair load once per run. */
static const char metal_qsg_n4_q40_source[] =
        "kernel void matvec_q40_n4(device const float*x[[buffer(0)]],device const "
        "uchar*w[[buffer(1)]],device float*y[[buffer(2)]],constant P&p[[buffer(3)]],uint3 "
        "tg[[threadgroup_position_in_grid]],ushort ti[[thread_index_in_simdgroup]],ushort "
        "sg[[simdgroup_index_in_threadgroup]]){"
        "uint b=tg.y,fr=(tg.x*2u+uint(sg))*4u;if(fr>=p.no||b>=p.rows)return;"
        "uint nb=p.ni>>5u,ix=uint(ti)>>1u,il=(uint(ti)&1u)*8u;"
        "uint nr=min(p.no-fr,4u);"
        "device const float*yb=x+p.xo+b*p.xs+ix*32u+il;"
        "float sf[4]={0.0f,0.0f,0.0f,0.0f};"
        "for(uint ib=ix;ib<nb;ib+=16u){"
        "float yl[16];float sumy=0.0f;"
        "for(uint i=0u;i<8u;i+=2u){"
        "float a0=yb[i],a1=yb[i+1u],c0=yb[i+16u],c1=yb[i+17u];"
        "sumy+=a0+a1+c0+c1;"
        "yl[i]=a0;yl[i+1u]=a1*(1.0f/256.0f);"
        "yl[i+8u]=c0*(1.0f/16.0f);yl[i+9u]=c1*(1.0f/4096.0f);}"
        "for(uint rr=0u;rr<nr;rr++){"
        "uint bo=p.wo+((fr+rr)*p.bpr+ib)*18u;"
        "device const ushort*qs=(device const ushort*)(w+bo+2u+il);"
        "float acc0=0.0f,acc1=0.0f;"
        "for(uint i=0u;i<4u;i++){uint q=uint(qs[i]);"
        "acc0+=yl[2u*i]*float(q&0x000Fu)+yl[2u*i+1u]*float(q&0x0F00u);"
        "acc1+=yl[2u*i+8u]*float(q&0x00F0u)+yl[2u*i+9u]*float(q&0xF000u);}"
        "float d=float(*((device const half*)(w+bo)));"
        "sf[rr]+=d*(acc0+acc1-8.0f*sumy);}"
        "yb+=512u;}"
        "for(uint rr=0u;rr<4u;rr++){float a=simd_sum(sf[rr]);"
        "if(ti==0&&fr+rr<p.no)y[p.yo+b*p.ys+fr+rr]=a;}}\n";

static const char metal_mv_q80_body[] =
        "static inline void mv_q80(device const float*x,device const uchar*w,device "
        "float*y,constant P&p,uint tx,uint b,ushort ti,ushort sg){"
        "uint fr=(tx*2u+uint(sg))*4u;if(fr>=p.no||b>=p.rows)return;"
        "uint nb=p.ni>>5u,ix=uint(ti)>>1u,il=(uint(ti)&1u)*16u;"
        "uint nr=min(p.no-fr,4u);"
        "device const float*yb=x+p.xo+b*p.xs+ix*32u+il;"
        "float sf[4]={0.0f,0.0f,0.0f,0.0f};"
        "for(uint ib=ix;ib<nb;ib+=16u){"
        "float yl[16];for(uint i=0u;i<16u;i++)yl[i]=yb[i];"
        "for(uint rr=0u;rr<nr;rr++){"
        "uint bo=p.wo+((fr+rr)*p.bpr+ib)*34u;"
        "device const char*qs=(device const char*)(w+bo+2u+il);"
        "float acc=0.0f;for(uint i=0u;i<16u;i++)acc+=yl[i]*float(qs[i]);"
        "float d=float(*((device const half*)(w+bo)));"
        "sf[rr]+=d*acc;}"
        "yb+=512u;}"
        "for(uint rr=0u;rr<4u;rr++){float a=simd_sum(sf[rr]);"
        "if(ti==0&&fr+rr<p.no)y[p.yo+b*p.ys+fr+rr]=a;}}\n";

static const char metal_qsg_n4_q80_source[] =
        "kernel void matvec_q80_n4(device const float*x[[buffer(0)]],device const "
        "uchar*w[[buffer(1)]],device float*y[[buffer(2)]],constant P&p[[buffer(3)]],uint3 "
        "tg[[threadgroup_position_in_grid]],ushort ti[[thread_index_in_simdgroup]],ushort "
        "sg[[simdgroup_index_in_threadgroup]]){mv_q80(x,w,y,p,tg.x,tg.y,ti,sg);}\n";

/* GEMM instances of the shared 64x32 simdgroup tile, one array each to
 * stay under the 4095-char literal limit. */
static const char metal_qsg_mm_q40_source[] = GEIST_METAL_MM_SG_KERNEL("q40", "bq40", "dq40", "2");
static const char metal_qsg_mm_q80_source[] = GEIST_METAL_MM_SG_KERNEL("q80", "bq80", "dq80", "2");
static const char metal_qsg_mm_q40_fast_source[] =
        GEIST_METAL_MM_SG_FAST_KERNEL("q40", "bq40", "dq40", "2");
static const char metal_qsg_mm_q80_fast_source[] =
        GEIST_METAL_MM_SG_FAST_KERNEL("q80", "bq80", "dq80", "2");

static const char metal_q5k_source[] =
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "struct P{uint ni,no,rows,bpr,xo,wo,yo,xs,ys;};\n"
        "static inline float h(device const uchar*w,uint o){ushort "
        "b=ushort(w[o])|(ushort(w[o+1])<<8);return float(as_type<half>(b));}\n"
        "static inline void sm5(uint j,device const uchar*q,thread uint&s,thread uint&m){if(j<4u){"
        "s=uint(q[j])&63u;m=uint(q[j+4u])&63u;}else{uint a=uint(q[j+4u]),b=uint(q[j-4u]),"
        "c=uint(q[j]);s=(a&15u)|((b>>6u)<<4u);m=(a>>4u)|((c>>6u)<<4u);}}\n"
        "static inline float dq5(device const uchar*w,constant P&p,uint r,uint k){uint br=k>>8u,"
        "ib=k&255u,sb=ib>>6u,j=ib&63u,hi=j>>5u,idx=j&31u,bo=p.wo+(r*p.bpr+br)*176u;"
        "uint sc,mn;sm5(sb*2u+hi,w+bo+4u,sc,mn);uint lo=uint(w[bo+48u+sb*32u+idx]);"
        "uint q=hi==0u?(lo&15u):(lo>>4u);uint mask=1u<<(sb*2u+hi);"
        "if((uint(w[bo+16u+idx])&mask)!=0u)q+=16u;return h(w,bo)*float(sc)*float(q)-"
        "h(w,bo+2u)*float(mn);}\n"
        "kernel void linear_q5k(device const float*x[[buffer(0)]],device const "
        "uchar*w[[buffer(1)]],"
        "device float*y[[buffer(2)]],constant P&p[[buffer(3)]],uint3 tg[[threadgroup_position_in_"
        "grid]],uint lid[[thread_index_in_threadgroup]]){threadgroup float part[256];uint r=tg.x,"
        "b=tg.y;if(r>=p.no||b>=p.rows)return;float s=0.0f;for(uint k=lid;k<p.ni;k+=256u)"
        "s+=x[p.xo+b*p.xs+k]*dq5(w,p,r,k);part[lid]=s;threadgroup_barrier(mem_flags::mem_"
        "threadgroup);for(uint z=128u;z>0u;z>>=1u){if(lid<z)part[lid]+=part[lid+z];threadgroup_"
        "barrier(mem_flags::mem_threadgroup);}if(lid==0u)y[p.yo+b*p.ys+r]=part[0];}\n"
        "kernel void linear_q5k_m8(device const float*x[[buffer(0)]],device const "
        "uchar*w[[buffer(1)"
        "]],device float*y[[buffer(2)]],constant P&p[[buffer(3)]],uint3 "
        "tg[[threadgroup_position_in_"
        "grid]],uint lid[[thread_index_in_threadgroup]]){threadgroup float part[8][256];uint "
        "r=tg.x,"
        "bb=tg.y*8u;if(r>=p.no||bb>=p.rows)return;float s[8];for(uint m=0u;m<8u;m++)s[m]=0.0f;"
        "for(uint k=lid;k<p.ni;k+=256u){float wv=dq5(w,p,r,k);for(uint m=0u;m<8u;m++){uint b=bb+m;"
        "if(b<p.rows)s[m]+=x[p.xo+b*p.xs+k]*wv;}}for(uint m=0u;m<8u;m++)part[m][lid]=s[m];"
        "threadgroup_barrier(mem_flags::mem_threadgroup);for(uint z=128u;z>0u;z>>=1u){if(lid<z){"
        "for(uint m=0u;m<8u;m++)part[m][lid]+=part[m][lid+z];}threadgroup_barrier(mem_flags::mem_"
        "threadgroup);}if(lid==0u){for(uint m=0u;m<8u;m++){uint b=bb+m;if(b<p.rows)"
        "y[p.yo+b*p.ys+r]=part[m][0];}}}\n";

static const char metal_qsg_n4_q5k_source[] =
        "kernel void matvec_q5k_n4(device const float*x[[buffer(0)]],device const "
        "uchar*w[[buffer(1)]],device float*y[[buffer(2)]],constant P&p[[buffer(3)]],uint3 "
        "tg[[threadgroup_position_in_grid]],ushort ti[[thread_index_in_simdgroup]],ushort "
        "sg[[simdgroup_index_in_threadgroup]]){"
        "uint b=tg.y,fr=(tg.x*2u+uint(sg))*4u;if(fr>=p.no||b>=p.rows)return;"
        "uint nb=p.ni>>8u;uint nr=min(p.no-fr,4u);"
        "uint sb=uint(ti)>>3u,rr8=uint(ti)&7u,hi=rr8>>2u,i0=(rr8&3u)*8u;"
        "uint sj=sb*2u+hi,hm=1u<<sj,qsh=hi*4u;"
        "uint qso=48u+sb*32u+i0,qho=16u+i0;"
        "device const float*yb=x+p.xo+b*p.xs+sb*64u+hi*32u+i0;"
        "float sf[4]={0.0f,0.0f,0.0f,0.0f};"
        "for(uint ib=0u;ib<nb;ib++){"
        "float yl[8];float sumy=0.0f;"
        "for(uint i=0u;i<8u;i++){yl[i]=yb[i];sumy+=yl[i];}"
        "for(uint rr=0u;rr<nr;rr++){"
        "uint bo=p.wo+((fr+rr)*p.bpr+ib)*176u;"
        "uint sc,mn;sm5(sj,w+bo+4u,sc,mn);"
        "float acc=0.0f;"
        "for(uint i=0u;i<8u;i++){uint v=(uint(w[bo+qso+i])>>qsh)&15u;"
        "if((uint(w[bo+qho+i])&hm)!=0u)v+=16u;acc+=yl[i]*float(v);}"
        "float d=float(*((device const half*)(w+bo)));"
        "float dm=float(*((device const half*)(w+bo+2u)));"
        "sf[rr]+=d*float(sc)*acc-dm*float(mn)*sumy;}"
        "yb+=256u;}"
        "for(uint rr=0u;rr<4u;rr++){float a=simd_sum(sf[rr]);"
        "if(ti==0&&fr+rr<p.no)y[p.yo+b*p.ys+fr+rr]=a;}}\n";

/* Q3_K and IQ3_S chunk loaders + structs for the shared GEMM template.
 * Their own MSL fragment: the qsg common literal is near the 4095-char
 * limit. IQ3_S references iq3sg[512], a constant array generated into
 * the source string at library-build time from quant/iq_grids.h. */
static const char metal_qsg_common2_source[] =
        "struct bq3k{uchar hmask[32];uchar qs[64];uchar scales[12];half d;};\n"
        "static inline float q3ksc(device const uchar*sc,short j){"
        "uint lo=(uint(sc[j%8])>>((j/8)*4))&15u;"
        "uint hi=(uint(sc[8+(j%4)])>>((j/4)*2))&3u;"
        "return float(int(lo|(hi<<4u))-32);}\n"
        "static inline void dq3k(device const bq3k*xb,short il,thread half4x4&r){"
        "float dl=float(xb->d)*q3ksc(xb->scales,il);"
        "ushort sl=ushort(((il/2)&3)*2);ushort sh=ushort(il/2);"
        "device const uchar*ql=xb->qs+(il/8)*32+16*(il&1);"
        "device const uchar*hm=xb->hmask+16*(il&1);"
        "FOR_UNROLL(short i=0;i<16;i++){"
        "int lo2=int((ql[i]>>sl)&3u);int hi1=int((hm[i]>>sh)&1u);"
        "r[i/4][i%4]=half(dl*float(lo2+4*hi1-4));}}\n"
        "struct biq3s{half d;uchar qs[64];uchar qh[8];uchar signs[32];uchar scales[4];};\n"
        "static inline void dqiq3s(device const biq3s*xb,short il,thread half4x4&r){"
        "short k=il/2;short h8=(il&1)*4;"
        "float db=float(xb->d)*(1.0f+2.0f*float((uint(xb->scales[k/2])>>(4*(k&1)))&15u));"
        "device const uchar*q=xb->qs+8*k+h8;"
        "uint qh=uint(xb->qh[k])>>ushort(h8);"
        "device const uchar*sg=xb->signs+4*k+2*(il&1);"
        "FOR_UNROLL(short i=0;i<16;i++){"
        "short j2=i/4;uint gi=uint(q[j2])|(((qh>>ushort(j2))&1u)<<8u);"
        "float gb=float((iq3sg[gi]>>(8u*uint(i&3)))&255u);"
        "float sn=((uint(sg[i>>3])>>ushort(i&7))&1u)!=0u?-1.0f:1.0f;"
        "r[i/4][i%4]=half(db*gb*sn);}}\n";

/* Simdgroup decode GEMVs: 2 simdgroups x 2 rows, a thread owns one
 * 8-element run of every 256-block. */
static const char metal_qsg_n4_q3k_source[] =
        "kernel void matvec_q3k_n4(device const float*x[[buffer(0)]],device const "
        "uchar*w[[buffer(1)]],device float*y[[buffer(2)]],constant P&p[[buffer(3)]],uint3 "
        "tg[[threadgroup_position_in_grid]],ushort ti[[thread_index_in_simdgroup]],ushort "
        "sg[[simdgroup_index_in_threadgroup]]){"
        "uint b=tg.y,fr=(tg.x*2u+uint(sg))*2u;if(fr>=p.no||b>=p.rows)return;"
        "uint nb=p.ni>>8u;"
        "short il2=short(ti>>1u);uint i0=(uint(ti)&1u)*8u;"
        "ushort sl=ushort(((il2/2)&3)*2);ushort shh=ushort(il2/2);"
        "uint qlo=32u+uint(il2/8)*32u+16u*uint(il2&1)+i0;"
        "uint hmo=16u*uint(il2&1)+i0;"
        "device const float*yb=x+p.xo+b*p.xs+uint(il2)*16u+i0;"
        "float s0=0.0f,s1=0.0f;"
        "for(uint ib=0u;ib<nb;ib++){"
        "float yl[8];for(uint i=0u;i<8u;i++)yl[i]=yb[i];"
        "for(uint rr=0u;rr<2u;rr++){uint row=fr+rr;if(row>=p.no)break;"
        "uint bo=p.wo+(row*p.bpr+ib)*110u;"
        "float sc=q3ksc(w+bo+96u,il2);"
        "float d=float(*((device const half*)(w+bo+108u)));"
        "float acc=0.0f;"
        "for(uint i=0u;i<8u;i++){"
        "int lo2=int((w[bo+qlo+i]>>sl)&3u);int hi1=int((w[bo+hmo+i]>>shh)&1u);"
        "acc+=yl[i]*float(lo2+4*hi1-4);}"
        "if(rr==0u)s0+=d*sc*acc;else s1+=d*sc*acc;}"
        "yb+=256u;}"
        "float a0=simd_sum(s0),a1=simd_sum(s1);"
        "if(ti==0){uint o=p.yo+b*p.ys+fr;y[o]=a0;if(fr+1u<p.no)y[o+1u]=a1;}}\n";

static const char metal_qsg_n4_iq3s_source[] =
        "kernel void matvec_iq3s_n4(device const float*x[[buffer(0)]],device const "
        "uchar*w[[buffer(1)]],device float*y[[buffer(2)]],constant P&p[[buffer(3)]],uint3 "
        "tg[[threadgroup_position_in_grid]],ushort ti[[thread_index_in_simdgroup]],ushort "
        "sg[[simdgroup_index_in_threadgroup]]){"
        "uint b=tg.y,fr=(tg.x*2u+uint(sg))*2u;if(fr>=p.no||b>=p.rows)return;"
        "uint nb=p.ni>>8u;"
        "uint k=uint(ti)>>2u,r4=uint(ti)&3u;"
        "device const float*yb=x+p.xo+b*p.xs+k*32u+r4*8u;"
        "float s0=0.0f,s1=0.0f;"
        "for(uint ib=0u;ib<nb;ib++){"
        "float yl[8];for(uint i=0u;i<8u;i++)yl[i]=yb[i];"
        "for(uint rr=0u;rr<2u;rr++){uint row=fr+rr;if(row>=p.no)break;"
        "uint bo=p.wo+(row*p.bpr+ib)*110u;"
        "float d=float(*((device const half*)(w+bo)));"
        "float db=d*(1.0f+2.0f*float((uint(w[bo+106u+k/2u])>>(4u*(k&1u)))&15u));"
        "uint qh=uint(w[bo+66u+k]);"
        "uint gi0=uint(w[bo+2u+8u*k+2u*r4])|(((qh>>(2u*r4))&1u)<<8u);"
        "uint gi1=uint(w[bo+2u+8u*k+2u*r4+1u])|(((qh>>(2u*r4+1u))&1u)<<8u);"
        "uint g0=iq3sg[gi0],g1=iq3sg[gi1];"
        "uint s8=uint(w[bo+74u+4u*k+r4]);"
        "float acc=0.0f;"
        "for(uint i=0u;i<8u;i++){"
        "uint gb=((i<4u?g0:g1)>>(8u*(i&3u)))&255u;"
        "float sn=((s8>>i)&1u)!=0u?-1.0f:1.0f;"
        "acc+=yl[i]*float(gb)*sn;}"
        "if(rr==0u)s0+=db*acc;else s1+=db*acc;}"
        "yb+=256u;}"
        "float a0=simd_sum(s0),a1=simd_sum(s1);"
        "if(ti==0){uint o=p.yo+b*p.ys+fr;y[o]=a0;if(fr+1u<p.no)y[o+1u]=a1;}}\n";

static const char metal_qsg_mm_q3k_source[] = GEIST_METAL_MM_SG_KERNEL("q3k", "bq3k", "dq3k", "16");
static const char metal_qsg_mm_iq3s_source[] =
        GEIST_METAL_MM_SG_KERNEL("iq3s", "biq3s", "dqiq3s", "16");

/* IQ4 simdgroup decode GEMVs. IQ4_NL shares the Q4_0 18-byte geometry
 * (4 rows per simdgroup); IQ4_XS follows the q5k_n4 structure (2 rows,
 * a thread owns one 8-element nibble run of every 256-block). */
static const char metal_qsg_n4_iq4nl_source[] =
        "kernel void matvec_iq4nl_n4(device const float*x[[buffer(0)]],device const "
        "uchar*w[[buffer(1)]],device float*y[[buffer(2)]],constant P&p[[buffer(3)]],uint3 "
        "tg[[threadgroup_position_in_grid]],ushort ti[[thread_index_in_simdgroup]],ushort "
        "sg[[simdgroup_index_in_threadgroup]]){"
        "uint b=tg.y,fr=(tg.x*2u+uint(sg))*4u;if(fr>=p.no||b>=p.rows)return;"
        "uint nb=p.ni>>5u,ix=uint(ti)>>1u,il=(uint(ti)&1u)*8u;"
        "uint nr=min(p.no-fr,4u);"
        "device const float*yb=x+p.xo+b*p.xs+ix*32u+il;"
        "float sf[4]={0.0f,0.0f,0.0f,0.0f};"
        "for(uint ib=ix;ib<nb;ib+=16u){"
        "float yl[16];for(uint i=0u;i<8u;i++){yl[i]=yb[i];yl[i+8u]=yb[i+16u];}"
        "for(uint rr=0u;rr<nr;rr++){"
        "uint bo=p.wo+((fr+rr)*p.bpr+ib)*18u;"
        "device const uchar*qs=w+bo+2u+il;"
        "float acc=0.0f;"
        "for(uint i=0u;i<8u;i++){uint q=uint(qs[i]);"
        "acc+=yl[i]*kiq4[q&15u]+yl[i+8u]*kiq4[q>>4u];}"
        "float d=float(*((device const half*)(w+bo)));"
        "sf[rr]+=d*acc;}"
        "yb+=512u;}"
        "for(uint rr=0u;rr<4u;rr++){float a=simd_sum(sf[rr]);"
        "if(ti==0&&fr+rr<p.no)y[p.yo+b*p.ys+fr+rr]=a;}}\n";

static const char metal_qsg_n4_iq4xs_source[] =
        "kernel void matvec_iq4xs_n4(device const float*x[[buffer(0)]],device const "
        "uchar*w[[buffer(1)]],device float*y[[buffer(2)]],constant P&p[[buffer(3)]],uint3 "
        "tg[[threadgroup_position_in_grid]],ushort ti[[thread_index_in_simdgroup]],ushort "
        "sg[[simdgroup_index_in_threadgroup]]){"
        "uint b=tg.y,fr=(tg.x*2u+uint(sg))*4u;if(fr>=p.no||b>=p.rows)return;"
        "uint nb=p.ni>>8u;uint nr=min(p.no-fr,4u);"
        "uint b32=uint(ti)>>2u,r4=uint(ti)&3u,nib=r4>>1u,j0=(r4&1u)*8u;"
        "ushort nsh=ushort(4u*nib);"
        "uint qso=8u+b32*16u+j0;"
        "device const float*yb=x+p.xo+b*p.xs+b32*32u+nib*16u+j0;"
        "float sf[4]={0.0f,0.0f,0.0f,0.0f};"
        "for(uint ib=0u;ib<nb;ib++){"
        "float yl[8];for(uint i=0u;i<8u;i++)yl[i]=yb[i];"
        "for(uint rr=0u;rr<nr;rr++){"
        "uint bo=p.wo+((fr+rr)*p.bpr+ib)*136u;"
        "uint sh=uint(*((device const ushort*)(w+bo+2u)));"
        "uint ls=((uint(w[bo+4u+b32/2u])>>(4u*(b32&1u)))&15u)|(((sh>>(2u*b32))&3u)<<4u);"
        "float dl=float(*((device const half*)(w+bo)))*(float(ls)-32.0f);"
        "device const ushort*q2=(device const ushort*)(w+bo+qso);"
        "float acc=0.0f;"
        "for(uint i=0u;i<4u;i++){uint q=uint(q2[i]);"
        "acc+=yl[2u*i]*kiq4[(q>>nsh)&15u]+yl[2u*i+1u]*kiq4[(q>>(nsh+8u))&15u];}"
        "sf[rr]+=dl*acc;}"
        "yb+=256u;}"
        "for(uint rr=0u;rr<4u;rr++){float a=simd_sum(sf[rr]);"
        "if(ti==0&&fr+rr<p.no)y[p.yo+b*p.ys+fr+rr]=a;}}\n";

/* PQ2_0 (PrismML ternary, Ternary-Bonsai): 128-element blocks of
 * [half d][32 bytes of 2-bit codes], element j at byte j/4 bits 2*(j%4),
 * value (code - 1) * d. dqpq2 serves the shared GEMM template (QK_NL 8:
 * eight 16-element chunks per block). The decode GEMV follows q40_n4 (4
 * rows per simdgroup, a thread owning a 32-element quarter block), but a
 * byte of codes is four weights, so per-weight mask + convert + FMA is
 * ALU-latency bound. A 256-entry half4 table in threadgroup memory maps
 * each code byte to its four codes, so a byte is one lookup and one float4
 * dot; unrolling the four rows (the tail clamps its row index, stores stay
 * guarded) lets their loads overlap. The -1 bias folds into sumy. What is
 * left is bound by the 2-byte-aligned block layout, not the arithmetic.
 * (Slower or over tolerance: x staged in threadgroup memory, half
 * activations, a float4 table, half the bytes through the ALU.) */
static const char metal_qsg_pq2_dq_source[] =
        "struct bpq2{half d;uchar qs[32];};\n"
        "static inline void dqpq2(device const bpq2*xb,short il,thread half4x4&r){"
        "half d=xb->d;device const uchar*q=xb->qs+4*il;"
        "FOR_UNROLL(short i=0;i<16;i++){"
        "r[i/4][i%4]=d*half(short((q[i/4]>>(2*(i%4)))&3)-1);}}\n";

/* R rows per simdgroup. R trades registers for activation traffic: a
 * thread loads 128 bytes of x per iteration and R*8 bytes of weights, so
 * x traffic is 15/R times the weight traffic. */
#define GEIST_METAL_PQ2_N_KERNEL(R)                                                       \
    "kernel void matvec_pq2_n" R "(device const float*x[[buffer(0)]],device const "       \
    "uchar*w[[buffer(1)]],device float*y[[buffer(2)]],constant P&p[[buffer(3)]],"         \
    "threadgroup float*shm[[threadgroup(0)]],uint3 tg[[threadgroup_position_in_grid]],"   \
    "uint3 nt3[[threads_per_threadgroup]],"                                               \
    "ushort ti[[thread_index_in_simdgroup]],ushort sg[[simdgroup_index_in_threadgroup]]," \
    "ushort tid[[thread_index_in_threadgroup]]){"                                         \
    "uint nsg=nt3.x>>5u;"                                                                 \
    "threadgroup half4*lut=(threadgroup half4*)shm;"                                      \
    "for(uint j=tid;j<256u;j+=nt3.x){lut[j]=half4(half(j&3u),half((j>>2u)&3u),"           \
    "half((j>>4u)&3u),half(j>>6u));}"                                                     \
    "threadgroup_barrier(mem_flags::mem_threadgroup);"                                    \
    "uint b=tg.y,fr=(tg.x*nsg+uint(sg))*" R "u;if(fr>=p.no||b>=p.rows)return;"            \
    "uint nb=p.ni>>7u,ix=uint(ti)>>2u,il=uint(ti)&3u;"                                    \
    "uint nr=min(p.no-fr," R "u);"                                                        \
    "device const float4*yb=(device const float4*)(x+p.xo+b*p.xs)+ix*32u+il*8u;"          \
    "float sf[" R "];FOR_UNROLL(uint i=0u;i<" R "u;i++)sf[i]=0.0f;"                       \
    "for(uint ib=ix;ib<nb;ib+=8u){"                                                       \
    "float4 yl[8];float sumy=0.0f;"                                                       \
    "FOR_UNROLL(uint i=0u;i<8u;i++){yl[i]=yb[i];sumy+=yl[i].x+yl[i].y+yl[i].z+yl[i].w;}"  \
    "FOR_UNROLL(uint rr=0u;rr<" R "u;rr++){"                                              \
    "uint bo=p.wo+((fr+min(rr,nr-1u))*p.bpr+ib)*34u;"                                     \
    "device const ushort*qs=(device const ushort*)(w+bo+2u+il*8u);"                       \
    "float a0=0.0f,a1=0.0f;"                                                              \
    "FOR_UNROLL(uint i=0u;i<4u;i++){uint q=uint(qs[i]);"                                  \
    "a0+=dot(yl[2u*i],float4(lut[q&255u]));a1+=dot(yl[2u*i+1u],float4(lut[q>>8u]));}"     \
    "float d=float(*((device const half*)(w+bo)));"                                       \
    "sf[rr]+=d*(a0+a1-sumy);}"                                                            \
    "yb+=256u;}"                                                                          \
    "FOR_UNROLL(uint rr=0u;rr<" R "u;rr++){float a=simd_sum(sf[rr]);"                     \
    "if(ti==0&&fr+rr<p.no)y[p.yo+b*p.ys+fr+rr]=a;}}\n"

static const char metal_qsg_pq2_source[]    = GEIST_METAL_PQ2_N_KERNEL("4");
static const char metal_qsg_pq2_n8_source[] = GEIST_METAL_PQ2_N_KERNEL("8");

/* PQ2_0 for 2..8 activation rows (#105: speculative verify). The 32-row
 * GEMM pays its staging for 2 rows as for 32, and a multi-row GEMV pays the
 * ALU-bound code decode once per row. Here every weight is decoded once,
 * straight into a simdgroup_matrix's thread elements (lane l holds row fm,
 * columns fn and fn+1: two codes of one byte, one 16-entry half2 table
 * lookup), and the 8 rows go through the MMA. A threadgroup is 16 outputs;
 * its four simdgroups split the 128-element blocks and add their 8x8 tiles
 * at the end. x arrives as f16 (cvt_f32_f16_rows) with 8 rows readable;
 * columns >= rows of the tile are never stored, so their contents do not
 * matter. */
static const char metal_qsg_mm_pq2_m8_source[] =
        "kernel void matmul_pq2_m8(device const half*x[[buffer(0)]],device const "
        "uchar*w[[buffer(1)]],device float*y[[buffer(2)]],constant P&p[[buffer(3)]],threadgroup "
        "char*shmem[[threadgroup(0)]],uint3 tg[[threadgroup_position_in_grid]],ushort "
        "ti[[thread_index_in_simdgroup]],ushort sg[[simdgroup_index_in_threadgroup]],ushort "
        "tid[[thread_index_in_threadgroup]]){\n"
        "threadgroup half2*lut=(threadgroup half2*)shmem;threadgroup float*red=(threadgroup "
        "float*)(shmem+64);"
        "if(tid<16){lut[tid]=half2(half(short(tid&3)-1),half(short(tid>>2)-1));}"
        "threadgroup_barrier(mem_flags::mem_threadgroup);"
        "uint qid=uint(ti)/4u;uint fm=(qid&4u)+(uint(ti)/2u)%4u;uint "
        "fn=(qid&2u)*2u+(uint(ti)%2u)*2u;"
        "uint o0=tg.x*16u;uint sh=2u*fn;"
        "uint ra=min(o0+fm,p.no-1u),rb=min(o0+8u+fm,p.no-1u);"
        "device const uchar*wa=w+p.wo+ra*p.bpr*34u;device const uchar*wb=w+p.wo+rb*p.bpr*34u;"
        "simdgroup_float8x8 ca=make_filled_simdgroup_matrix<float,8>(0.0f);"
        "simdgroup_float8x8 cb=make_filled_simdgroup_matrix<float,8>(0.0f);"
        "simdgroup_half8x8 ma,mb,mx;"
        "for(uint ib=uint(sg);ib<p.bpr;ib+=4u){"
        "device const half*xb=x+p.xo+ib*128u;"
        "half da=*(device const half*)(wa+ib*34u);half db=*(device const half*)(wb+ib*34u);"
        "device const ushort*qa=(device const ushort*)(wa+ib*34u+2u);"
        "device const ushort*qb=(device const ushort*)(wb+ib*34u+2u);"
        "FOR_UNROLL(short s=0;s<16;s++){"
        "simdgroup_load(mx,xb+8*s,p.xs,ulong2(0,0),true);"
        "half2 va=da*lut[(uint(qa[s])>>sh)&15u];half2 vb=db*lut[(uint(qb[s])>>sh)&15u];"
        "ma.thread_elements()[0]=va.x;ma.thread_elements()[1]=va.y;"
        "mb.thread_elements()[0]=vb.x;mb.thread_elements()[1]=vb.y;"
        "simdgroup_multiply_accumulate(ca,ma,mx,ca);simdgroup_multiply_accumulate(cb,mb,mx,cb);}}"
        "simdgroup_store(ca,red+128u*uint(sg),8);simdgroup_store(cb,red+128u*uint(sg)+64u,8);"
        "threadgroup_barrier(mem_flags::mem_threadgroup);"
        "uint e=uint(tid);float a=red[e]+red[128u+e]+red[256u+e]+red[384u+e];"
        "uint o=o0+(e/64u)*8u+(e%64u)/8u;uint c=e%8u;"
        "if(c<p.rows&&o<p.no){y[p.yo+c*p.ys+o]=a;}}\n";

static const char metal_qsg_mm_pq2_source[] = GEIST_METAL_MM_SG_KERNEL("pq2", "bpq2", "dqpq2", "8");
/* PQ2_0 interior fast GEMM (#77, #103): a 64-output x 64-row tile, four
 * simdgroups owning a 32 x 32 block each (16 accumulators): per 8-wide k
 * step each loads 4 weight and 4 activation 8x8 tiles for 16 MMAs. The
 * per-output accumulation order is the 64 x 32 template's (k steps in
 * order, one MMA per 8 k).
 *
 * The MMAs alone run at ~10 TFLOP/s on the M1 Max (96 % of the fp16
 * peak); what costs the rest is the staging between the two barriers of
 * each 32-k step, which issues on the same ALUs. So the staging is cut down
 * (#103), every step bit-identical to before:
 *  - x arrives as f16 (cvt_f32_f16_rows, one dispatch per GEMM into a
 *    backend scratch), the same round-to-nearest the staging did, so the
 *    loop no longer loads f32 and converts 16 values per thread;
 *  - a 256-entry half4 table maps a code byte to its four (code - 1)
 *    values (the matvec_pq2 table), one multiply by d each: d * (code - 1)
 *    as before, without the per-weight shift, mask and convert;
 *  - the next step's code bytes, scale and x are loaded into registers
 *    before this step's MMAs, so their latency hides behind them.
 * Interleaved on Bonsai 2 27B shapes at 320 rows: -6 to -7 % against the
 * f32 kernel, -12 % on the narrow (1024-output, 64-row) ones.
 * Threadgroup memory: 8 KB tiles + 2 KB table; a ragged last row tile
 * stages its output in 16 KB. (Tried and not kept, #103: larger tiles
 * 64x128/128x64/128x128 and 8 or 16 simdgroups, +10-30 %; weights decoded
 * straight into simdgroup_matrix thread elements, +14-30 %; activations
 * read by simdgroup_load from device memory, +6 %; double-buffered
 * staging, a 64-wide k step, transposed weight staging, 2x8 or 4x4 decode
 * blocks with vector stores: all slower; a branch-free bit decode: no
 * gain.) */
static const char metal_qsg_mm_pq2_fast_source[] =
        "kernel void matmul_pq2_mm_sg_fast(device const half*x[[buffer(0)]],device const "
        "uchar*w[[buffer(1)]],device float*y[[buffer(2)]],constant P&p[[buffer(3)]],threadgroup "
        "char*shmem[[threadgroup(0)]],uint3 tg[[threadgroup_position_in_grid]],ushort "
        "ti[[thread_index_in_threadgroup]],ushort sg[[simdgroup_index_in_threadgroup]]){\n"
        "threadgroup half*sa=(threadgroup half*)shmem;threadgroup half*sb=(threadgroup "
        "half*)(shmem+4096);threadgroup half4*lut=(threadgroup half4*)(shmem+8192);"
        "for(ushort j=ti;j<256;j+=128){lut[j]=half4(half(short(j&3)-1),half(short((j>>2)&3)-1),"
        "half(short((j>>4)&3)-1),half(short(j>>6)-1));}"
        "threadgroup_barrier(mem_flags::mem_threadgroup);"
        "uint r0=tg.y*64u;uint r1=tg.x*64u;"
        "short lr0=short(ti/2);short il0=short(ti%2);short il=il0;"
        "device const bpq2*wp=(device const bpq2*)(w+p.wo)+(r0+uint(lr0))*p.bpr;"
        "device const half*xp=x+p.xo+min(r1+uint(lr0),p.rows-1u)*p.xs+uint(16*il0);"
        "threadgroup half*sbp=sb+64*(16*il0+lr0/8)+8*(lr0%8);"
        "simdgroup_half8x8 ma[4];simdgroup_half8x8 mb[4];simdgroup_float8x8 mc[16];"
        "FOR_UNROLL(short i=0;i<16;i++){mc[i]=make_filled_simdgroup_matrix<float,8>(0.0f);}"
        "half wd=wp->d;device const ushort*wq=(device const ushort*)(wp->qs+4*il);"
        "ushort q0=wq[0],q1=wq[1];"
        "half2x4 xa=*((device const half2x4*)xp),xb=*((device const half2x4*)(xp+8));"
        "for(uint loop_k=0u;loop_k<p.ni;loop_k+=32u){"
        "half4x4 ta;ta[0]=wd*lut[q0&255];ta[1]=wd*lut[q0>>8];ta[2]=wd*lut[q1&255];"
        "ta[3]=wd*lut[q1>>8];"
        "threadgroup_barrier(mem_flags::mem_threadgroup);"
        "FOR_UNROLL(short i=0;i<16;i++){short ib=short(8*(2*il0+i/8)+lr0/8);"
        "*(sa+64*ib+8*(i%8)+lr0%8)=ta[i/4][i%4];}"
        "*(threadgroup half2x4*)sbp=xa;*(threadgroup half2x4*)(sbp+512)=xb;"
        "il=(il+2<8)?il+2:il%2;wp=(il<2)?wp+1:wp;xp+=32;"
        "if(loop_k+32u<p.ni){wd=wp->d;wq=(device const ushort*)(wp->qs+4*il);q0=wq[0];q1=wq[1];"
        "xa=*((device const half2x4*)xp);xb=*((device const half2x4*)(xp+8));}"
        "threadgroup_barrier(mem_flags::mem_threadgroup);"
        "threadgroup const half*lsma=sa+4*64*(sg%2);threadgroup const half*lsmb=sb+4*64*(sg/2);"
        "FOR_UNROLL(short ik=0;ik<4;ik++){"
        "simdgroup_barrier(mem_flags::mem_none);"
        "FOR_UNROLL(short i=0;i<4;i++){simdgroup_load(ma[i],lsma+64*i,8,0,false);}"
        "simdgroup_barrier(mem_flags::mem_none);"
        "FOR_UNROLL(short i=0;i<4;i++){simdgroup_load(mb[i],lsmb+64*i,8,0,false);}"
        "simdgroup_barrier(mem_flags::mem_none);"
        "FOR_UNROLL(short i=0;i<16;i++){simdgroup_multiply_accumulate(mc[i],mb[i/4],ma[i%4],"
        "mc[i]);}"
        "lsma+=8*64;lsmb+=8*64;}}"
        "if(r1+64u<=p.rows){device float*C=y+p.yo+(r1+32u*uint(sg>>1))*p.ys+r0+32u*uint(sg&1);"
        "FOR_UNROLL(short i=0;i<16;i++){simdgroup_store(mc[i],C+8*(i%4)+8*p.ys*(i/4),p.ys,0,"
        "false);}}"
        "else{threadgroup_barrier(mem_flags::mem_threadgroup);threadgroup float*T=(threadgroup "
        "float*)shmem;"
        "FOR_UNROLL(short i=0;i<16;i++){simdgroup_store(mc[i],T+32*(sg&1)+2048*(sg>>1)+8*(i%4)+"
        "512*(i/4),64,0,false);}"
        "threadgroup_barrier(mem_flags::mem_threadgroup);"
        "for(uint j=uint(ti);j<(p.rows-r1)*64u;j+=128u){y[p.yo+(r1+j/64u)*p.ys+r0+j%64u]=T[j];}}"
        "}\n";

/* The fast PQ2_0 GEMM's x as f16: rows of n4 float4 at xo4 + row * xs4
 * (in float4 units) to a dense rows x n4 half4 copy. half4(float4) rounds to
 * nearest even, as the GEMM's own staging did. */
static const char metal_qsg_cvt_f16_source[] =
        "struct PH{uint n4,xo4,xs4;};\n"
        "kernel void cvt_f32_f16_rows(device const float4*x[[buffer(0)]],device "
        "half4*y[[buffer(1)]],constant PH&p[[buffer(2)]],uint3 g[[thread_position_in_grid]]){"
        "if(g.x>=p.n4)return;y[g.y*p.n4+g.x]=half4(x[p.xo4+g.y*p.xs4+g.x]);}\n";

/* Decode GEMV for the two 2-bit ternary formats that share TQ2_0's
 * element layout (TQ2_0 and I2_S), after q3k_n4: 2 rows per simdgroup, a
 * thread owns one 8-element run of every 256-element block, so its 8 codes share one
 * shift, and the -1 bias folds into sumy. The formats differ in the shift
 * (SH), the block bytes (BB) and where the scale lives: TQ2_0 has a half
 * per block (BLK_D, applied per block through D_MUL), I2_S one f32 per
 * tensor (PRE, applied once to each row's sum through POST). */
#define GEIST_METAL_TERNARY_N4_KERNEL(NAME, PRE, SH, BB, BLK_D, D_MUL, POST)            \
    "kernel void matvec_" NAME "_n4(device const float*x[[buffer(0)]],device const "    \
    "uchar*w[[buffer(1)]],device float*y[[buffer(2)]],constant P&p[[buffer(3)]],uint3 " \
    "tg[[threadgroup_position_in_grid]],ushort ti[[thread_index_in_simdgroup]],ushort " \
    "sg[[simdgroup_index_in_threadgroup]]){"                                            \
    "uint b=tg.y,fr=(tg.x*2u+uint(sg))*2u;if(fr>=p.no||b>=p.rows)return;"               \
    "uint nb=p.ni>>8u,t=uint(ti);" PRE "uint qo=(t>>4u)*32u+(t&3u)*8u;uint sh=" SH ";"  \
    "device const float*yb=x+p.xo+b*p.xs+t*8u;"                                         \
    "float s0=0.0f,s1=0.0f;"                                                            \
    "for(uint ib=0u;ib<nb;ib++){"                                                       \
    "float yl[8];float sumy=0.0f;"                                                      \
    "for(uint i=0u;i<8u;i++){yl[i]=yb[i];sumy+=yl[i];}"                                 \
    "for(uint rr=0u;rr<2u;rr++){uint row=fr+rr;if(row>=p.no)break;"                     \
    "uint bo=p.wo+(row*p.bpr+ib)*" BB "u;"                                              \
    "float acc=0.0f;"                                                                   \
    "for(uint i=0u;i<8u;i++)acc+=yl[i]*float((uint(w[bo+qo+i])>>sh)&3u);" BLK_D         \
    "if(rr==0u)s0+=" D_MUL "(acc-sumy);else s1+=" D_MUL "(acc-sumy);}"                  \
    "yb+=256u;}"                                                                        \
    "float a0=simd_sum(s0)" POST ",a1=simd_sum(s1)" POST ";"                            \
    "if(ti==0){uint o=p.yo+b*p.ys+fr;y[o]=a0;if(fr+1u<p.no)y[o+1u]=a1;}}\n"

/* TQ2_0 (BitNet ternary, #559): 256-element blocks of [64 bytes of 2-bit
 * codes][half d], value (code - 1) * d. Element e of a block sits in byte
 * (e/128)*32 + e%32 at bits 2*((e%128)/32) — dequant_tq2_0_row. dqtq2
 * serves the shared GEMM template (QK_NL 16); the decode GEMV, which
 * follows q3k_n4, is GEIST_METAL_TERNARY_N4_KERNEL. */
static const char metal_qsg_tq2_source[] =
        "struct btq2{uchar qs[64];half d;};\n"
        "static inline void dqtq2(device const btq2*xb,short il,thread half4x4&r){"
        "half d=xb->d;device const uchar*q=xb->qs+(il/8)*32+(il&1)*16;"
        "short sh=((il&7)/2)*2;"
        "FOR_UNROLL(short i=0;i<16;i++){r[i/4][i%4]=d*half(short((q[i]>>sh)&3)-1);}}\n";
static const char metal_qsg_tq2_n4_source[] =
        GEIST_METAL_TERNARY_N4_KERNEL("tq2",
                                      "",
                                      "((t>>2u)&3u)*2u",
                                      "66",
                                      "float d=float(*((device const half*)(w+bo+64u)));",
                                      "d*",
                                      "");

static const char metal_qsg_mm_tq2_source[] =
        GEIST_METAL_MM_SG_KERNEL("tq2", "btq2", "dqtq2", "16");

/* I2_S (bitnet.cpp's BitNet b1.58 format, #560): TQ2_0's element layout
 * with the four fields of a byte in reverse order (shift 6 - 2g), 64-byte
 * blocks and no block scale — one f32 per tensor right after the packed
 * bytes (i2_s_scale_offset). The GEMM template hands its dequant only the
 * block pointer, so dqi2s_t is a macro that reads the scale through the
 * kernel's own `w` and `p`; both kernels therefore need the view to start
 * at the tensor (metal_tensor_is_q40_q80_matrix checks the tail fits). */
static const char metal_qsg_i2s_source[] =
        "struct bi2s{uchar qs[64];};\n"
        "static inline void dqi2s(device const bi2s*xb,short il,thread half4x4&r,float s){"
        "half hs=half(s);device const uchar*q=xb->qs+(il/8)*32+(il&1)*16;"
        "short sh=6-((il&7)/2)*2;"
        "FOR_UNROLL(short i=0;i<16;i++){r[i/4][i%4]=hs*half(short((q[i]>>sh)&3)-1);}}\n"
        "#define dqi2s_t(xb,il,r) "
        "dqi2s(xb,il,r,*((device const float*)(w+p.wo+p.no*p.bpr*64u)))\n";
static const char metal_qsg_i2s_n4_source[] =
        GEIST_METAL_TERNARY_N4_KERNEL("i2s",
                                      "float d=*((device const float*)(w+p.wo+p.no*p.bpr*64u));",
                                      "6u-((t>>2u)&3u)*2u",
                                      "64",
                                      "",
                                      "",
                                      "*d");

static const char metal_qsg_mm_i2s_source[] =
        GEIST_METAL_MM_SG_KERNEL("i2s", "bi2s", "dqi2s_t", "16");

static const char metal_qsg_mm_iq4nl_source[] =
        GEIST_METAL_MM_SG_KERNEL("iq4nl", "biq4nl", "dqiq4nl", "2");
static const char metal_qsg_mm_iq4xs_source[] =
        GEIST_METAL_MM_SG_KERNEL("iq4xs", "biq4xs", "dqiq4xs", "16");
static const char metal_qsg_mm_iq4xs_fast_source[] =
        GEIST_METAL_MM_SG_FAST_KERNEL("iq4xs", "biq4xs", "dqiq4xs", "16");

static const char metal_qsg_mm_q5k_source[] = GEIST_METAL_MM_SG_KERNEL("q5k", "bq5k", "dq5k", "16");
static const char metal_qsg_mm_q5k_fast_source[] =
        GEIST_METAL_MM_SG_FAST_KERNEL("q5k", "bq5k", "dq5k", "16");

static const char metal_q41_source[] =
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "struct P{uint ni,no,rows,bpr,xo,wo,yo,xs,ys;};\n"
        "static inline float h(device const uchar*w,uint o){ushort b=ushort(w[o])|(ushort(w[o+1])"
        "<<8);return float(as_type<half>(b));}\n"
        "static inline float dq(device const uchar*w,constant P&p,uint r,uint k){uint bi=k>>5u,"
        "j=k&31u,bo=p.wo+(r*p.bpr+bi)*20u;uint q=uint(w[bo+4u+(j&15u)]);q=j<16u?(q&15u):(q>>4u);"
        "return h(w,bo)*float(q)+h(w,bo+2u);}\n"
        "kernel void linear_q41(device const float*x[[buffer(0)]],device const "
        "uchar*w[[buffer(1)]],"
        "device float*y[[buffer(2)]],constant P&p[[buffer(3)]],uint3 tg[[threadgroup_position_in_"
        "grid]],uint lid[[thread_index_in_threadgroup]]){threadgroup float part[256];uint r=tg.x,"
        "b=tg.y;if(r>=p.no||b>=p.rows)return;float s=0.0f;for(uint k=lid;k<p.ni;k+=256u)"
        "s+=x[p.xo+b*p.xs+k]*dq(w,p,r,k);part[lid]=s;threadgroup_barrier(mem_flags::mem_"
        "threadgroup);for(uint z=128u;z>0u;z>>=1u){if(lid<z)part[lid]+=part[lid+z];threadgroup_"
        "barrier(mem_flags::mem_threadgroup);}if(lid==0u)y[p.yo+b*p.ys+r]=part[0];}\n"
        "kernel void linear_q41_m8(device const float*x[[buffer(0)]],device const "
        "uchar*w[[buffer(1)"
        "]],device float*y[[buffer(2)]],constant P&p[[buffer(3)]],uint3 "
        "tg[[threadgroup_position_in_"
        "grid]],uint lid[[thread_index_in_threadgroup]]){threadgroup float part[8][256];uint "
        "r=tg.x,"
        "bb=tg.y*8u;if(r>=p.no||bb>=p.rows)return;float s[8];for(uint m=0u;m<8u;m++)s[m]=0.0f;"
        "for(uint k=lid;k<p.ni;k+=256u){float wv=dq(w,p,r,k);for(uint m=0u;m<8u;m++){uint b=bb+m;"
        "if(b<p.rows)s[m]+=x[p.xo+b*p.xs+k]*wv;}}for(uint m=0u;m<8u;m++)part[m][lid]=s[m];"
        "threadgroup_barrier(mem_flags::mem_threadgroup);for(uint z=128u;z>0u;z>>=1u){if(lid<z){"
        "for(uint m=0u;m<8u;m++)part[m][lid]+=part[m][lid+z];}threadgroup_barrier(mem_flags::mem_"
        "threadgroup);}if(lid==0u){for(uint m=0u;m<8u;m++){uint b=bb+m;if(b<p.rows)"
        "y[p.yo+b*p.ys+r]=part[m][0];}}}\n";

/* Simdgroup decode GEMV, matvec_q40_n4 structure; v = d*q + m, so the
 * offset term is +m*sumy instead of -8d*sumy. */
static const char metal_qsg_n4_q41_source[] =
        "kernel void matvec_q41_n4(device const float*x[[buffer(0)]],device const "
        "uchar*w[[buffer(1)]],device float*y[[buffer(2)]],constant P&p[[buffer(3)]],uint3 "
        "tg[[threadgroup_position_in_grid]],ushort ti[[thread_index_in_simdgroup]],ushort "
        "sg[[simdgroup_index_in_threadgroup]]){"
        "uint b=tg.y,fr=(tg.x*2u+uint(sg))*2u;if(fr>=p.no||b>=p.rows)return;"
        "uint nb=p.ni>>5u,ix=uint(ti)>>1u,il=(uint(ti)&1u)*8u;"
        "device const float*yb=x+p.xo+b*p.xs+ix*32u+il;"
        "float s0=0.0f,s1=0.0f;"
        "for(uint ib=ix;ib<nb;ib+=16u){"
        "float yl[16];float sumy=0.0f;"
        "for(uint i=0u;i<8u;i+=2u){"
        "float a0=yb[i],a1=yb[i+1u],c0=yb[i+16u],c1=yb[i+17u];"
        "sumy+=a0+a1+c0+c1;"
        "yl[i]=a0;yl[i+1u]=a1*(1.0f/256.0f);"
        "yl[i+8u]=c0*(1.0f/16.0f);yl[i+9u]=c1*(1.0f/4096.0f);}"
        "for(uint rr=0u;rr<2u;rr++){uint row=fr+rr;if(row>=p.no)break;"
        "uint bo=p.wo+(row*p.bpr+ib)*20u;"
        "device const ushort*qs=(device const ushort*)(w+bo+4u+il);"
        "float acc0=0.0f,acc1=0.0f;"
        "for(uint i=0u;i<4u;i++){uint q=uint(qs[i]);"
        "acc0+=yl[2u*i]*float(q&0x000Fu)+yl[2u*i+1u]*float(q&0x0F00u);"
        "acc1+=yl[2u*i+8u]*float(q&0x00F0u)+yl[2u*i+9u]*float(q&0xF000u);}"
        "float d=float(*((device const half*)(w+bo)));"
        "float mv=float(*((device const half*)(w+bo+2u)));"
        "float r=d*(acc0+acc1)+mv*sumy;"
        "if(rr==0u)s0+=r;else s1+=r;}"
        "yb+=512u;}"
        "float a0=simd_sum(s0),a1=simd_sum(s1);"
        "if(ti==0){uint o=p.yo+b*p.ys+fr;y[o]=a0;if(fr+1u<p.no)y[o+1u]=a1;}}\n";

static const char metal_qsg_mm_q41_source[] = GEIST_METAL_MM_SG_KERNEL("q41", "bq41", "dq41", "2");
static const char metal_qsg_mm_q41_fast_source[] =
        GEIST_METAL_MM_SG_FAST_KERNEL("q41", "bq41", "dq41", "2");

static const char metal_mv_hdr_source[] = "#include <metal_stdlib>\n"
                                          "using namespace metal;\n"
                                          "struct P{uint ni,no,rows,bpr,xo,wo,yo,xs,ys;};\n";

static const char metal_mv_q4k_body[] =
        "static inline void mv_q4k(device const float*x,device const uchar*w,device "
        "float*y,constant P&p,uint tx,uint b,ushort ti,ushort sg){constexpr ushort "
        "km1=0x3f3f,km2=0x0f0f,km3=0xc0c0;uint "
        "ix=uint(ti)/"
        "8u,it=uint(ti)&7u,iq=it>>2u,ir=it&3u,fr=(tx*2u+uint(sg))*2u;if(fr>=p.no||b>=p."
        "rows)return;uint nb=p.ni>>8u;float sum0=0.0f,sum1=0.0f;device const "
        "float*y4=x+p.xo+b*p.xs+ix*256u+64u*iq+8u*ir;ushort sc16[4];thread const uchar*sc8=(thread "
        "const uchar*)sc16;for(uint ib=ix;ib<nb;ib+=4u){float yl[16],yh[16];float4 "
        "sumy=float4(0.0f);for(uint "
        "i=0u;i<8u;i++){yl[i]=y4[i];sumy[0]+=yl[i];yl[i+8u]=y4[i+32u];sumy[1]+=yl[i+8u];yh[i]=y4[i+"
        "128u];sumy[2]+=yh[i];yh[i+8u]=y4[i+160u];sumy[3]+=yh[i+8u];}for(uint "
        "rr=0u;rr<2u;rr++){uint row=fr+rr;if(row>=p.no)continue;device const "
        "uchar*wb=w+p.wo+(row*p.bpr+ib)*144u;device const ushort*sc=(device const "
        "ushort*)(wb+4u)+iq;device const ushort*q1=(device const "
        "ushort*)(wb+16u)+16u*iq+4u*ir;device const half*dh=(device const "
        "half*)wb;sc16[0]=sc[0]&km1;sc16[1]=sc[2]&km1;sc16[2]=((sc[4]>>0)&km2)|((sc[0]&km3)>>2);"
        "sc16[3]=((sc[4]>>4)&km2)|((sc[2]&km3)>>2);device const ushort*q2=q1+32u;float4 "
        "a1=float4(0.0f),a2=float4(0.0f);for(uint i=0u;i<4u;i++){ushort "
        "q=q1[i];a1[0]+=yl[2u*i+0u]*float(q&0x000f);a1[1]+=yl[2u*i+1u]*float(q&0x0f00);a1[2]+=yl["
        "2u*i+8u]*float(q&0x00f0);a1[3]+=yl[2u*i+9u]*float(q&0xf000);q=q2[i];a2[0]+=yh[2u*i+0u]*"
        "float(q&0x000f);a2[1]+=yh[2u*i+1u]*float(q&0x0f00);a2[2]+=yh[2u*i+8u]*float(q&0x00f0);a2["
        "3]+=yh[2u*i+9u]*float(q&0xf000);}float "
        "acc=float(dh[0])*((a1[0]+(1.0f/256.0f)*a1[1])*float(sc8[0])+(a1[2]+(1.0f/"
        "256.0f)*a1[3])*float(sc8[1])*(1.0f/16.0f)+(a2[0]+(1.0f/"
        "256.0f)*a2[1])*float(sc8[4])+(a2[2]+(1.0f/256.0f)*a2[3])*float(sc8[5])*(1.0f/"
        "16.0f))-float(dh[1])*(sumy[0]*float(sc8[2])+sumy[1]*float(sc8[3])+sumy[2]*float(sc8[6])+"
        "sumy[3]*float(sc8[7]));if(rr==0u)sum0+=acc;else sum1+=acc;}y4+=1024u;}float "
        "all0=simd_sum(sum0),all1=simd_sum(sum1);if(ti==0){uint "
        "o=p.yo+b*p.ys+fr;y[o]=all0;if(fr+1u<p.no)y[o+1u]=all1;}}\n";

static const char metal_q4k_n4_kernel[] =
        "kernel void matvec_q4k_n4(device const float*x[[buffer(0)]],device const "
        "uchar*w[[buffer(1)]],device float*y[[buffer(2)]],constant P&p[[buffer(3)]],uint3 "
        "tg[[threadgroup_position_in_grid]],ushort ti[[thread_index_in_simdgroup]],ushort "
        "sg[[simdgroup_index_in_threadgroup]]){mv_q4k(x,w,y,p,tg.x,tg.y,ti,sg);}\n";

static const char metal_q4k_m16_source[] =
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "struct P{uint ni,no,rows,bpr,xo,wo,yo,xs,ys;};\n"
        "static inline float h(device const uchar*w,uint o){ushort "
        "b=ushort(w[o])|(ushort(w[o+1])<<8);return float(as_type<half>(b));}\n"
        "static inline void sm(uint j,device const uchar*q,thread uint&s,thread "
        "uint&m){if(j<4){s=uint(q[j])&63u;m=uint(q[j+4])&63u;}else{uint "
        "a=uint(q[j+4]),b=uint(q[j-4]),c=uint(q[j]);s=(a&15u)|((b>>6u)<<4u);m=(a>>4u)|((c>>6u)<<4u)"
        ";}}\n"
        "static inline float dq(device const uchar*w,constant P&p,uint r,uint k){uint "
        "br=k>>8u,ib=k&255u,sub=ib>>5u,idx=ib&31u,bo=p.wo+(r*p.bpr+br)*144u;float "
        "d=h(w,bo),dm=h(w,bo+2u);uint s,m;sm(sub,w+bo+4u,s,m);uint "
        "qb=uint(w[bo+16u+(sub>>1u)*32u+idx]);uint q=(sub&1u)==0u?(qb&15u):(qb>>4u);return "
        "d*float(s)*float(q)-dm*float(m);}\n"
        "kernel void matmul_q4k_m16(device const float*x[[buffer(0)]],device const "
        "uchar*w[[buffer(1)]],device float*y[[buffer(2)]],constant P&p[[buffer(3)]],uint3 "
        "tg[[threadgroup_position_in_grid]],uint lid[[thread_index_in_threadgroup]]){threadgroup "
        "float part[16][256];uint row=tg.x,bb=tg.y*16u;if(row>=p.no||bb>=p.rows)return;float "
        "s[16];for(uint m=0u;m<16u;m++)s[m]=0.0f;for(uint k=lid;k<p.ni;k+=256u){float "
        "wv=dq(w,p,row,k);for(uint m=0u;m<16u;m++){uint "
        "b=bb+m;if(b<p.rows)s[m]+=x[p.xo+b*p.xs+k]*wv;}}for(uint "
        "m=0u;m<16u;m++)part[m][lid]=s[m];threadgroup_barrier(mem_flags::mem_threadgroup);for(uint "
        "st=128u;st>0u;st>>=1u){if(lid<st){for(uint "
        "m=0u;m<16u;m++)part[m][lid]+=part[m][lid+st];}threadgroup_barrier(mem_flags::mem_"
        "threadgroup);}if(lid==0u){for(uint m=0u;m<16u;m++){uint "
        "b=bb+m;if(b<p.rows)y[p.yo+b*p.ys+row]=part[m][0];}}}\n";

static const char metal_q4k_mm_sg_source[] =
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "#define FOR_UNROLL(x) _Pragma(\"clang loop unroll(full)\") for(x)\n"
        "struct P{uint ni,no,rows,bpr,xo,wo,yo,xs,ys;};\n"
        "struct block_q4_K{half d;half dmin;uchar scales[12];uchar qs[128];};\n"
        "static inline uchar2 sm2(short j,short k,device const uchar*q){return "
        "j<4?uchar2(uchar(q[j+k]&63),uchar(q[j+4+k]&63)):uchar2(uchar((q[j+4+k]&15)|((q[j-4+k]&192)"
        ">>2)),uchar((q[j+4+k]>>4)|((q[j+k]&192)>>2)));}\n"
        "static inline void dq4(device const block_q4_K*xb,short il,thread half4x4&r){device const "
        "uchar*q=xb->qs;short is=(il/4)*2;q=q+(il/4)*32+16*(il&1);il=il&3;uchar2 "
        "sc=sm2(is,il/2,xb->scales);float d=il<2?float(xb->d):float(xb->d)/16.0f;float "
        "mn=float(xb->dmin);float dl=d*float(sc[0]);float ml=mn*float(sc[1]);ushort "
        "mask=il<2?0x0f:0xf0;FOR_UNROLL(short "
        "i=0;i<16;i++){r[i/4][i%4]=half(dl*float(q[i]&mask)-ml);}}\n"
        "kernel void matmul_q4k_mm_sg(device const float*x[[buffer(0)]],device const "
        "uchar*w[[buffer(1)]],device float*y[[buffer(2)]],constant P&p[[buffer(3)]],threadgroup "
        "char*shmem[[threadgroup(0)]],uint3 tg[[threadgroup_position_in_grid]],ushort "
        "ti[[thread_index_in_threadgroup]],ushort sg[[simdgroup_index_in_threadgroup]]){\n"
        "threadgroup half*sa=(threadgroup half*)shmem;threadgroup half*sb=(threadgroup "
        "half*)(shmem+4096);constexpr short NR0=64;constexpr short NR1=32;constexpr short "
        "NK=32;constexpr short NL0=2;constexpr short NL1=4;constexpr short QK_NL=16;uint "
        "r0=tg.y*uint(NR0);uint r1=tg.x*uint(NR1);if(r0>=p.no||r1>=p.rows)return;short "
        "nr0=short(min(p.no-r0,uint(NR0)));short nr1=short(min(p.rows-r1,uint(NR1)));short "
        "lr0=min(short(ti/NL0),short(nr0-1));short lr1=min(short(ti/NL1),short(nr1-1));short "
        "il0=short(ti%NL0);short il=il0;device const block_q4_K*wp=(device const "
        "block_q4_K*)(w+p.wo)+(r0+uint(lr0))*p.bpr;short iy=short(8*(ti%NL1));device const "
        "float*xp=x+p.xo+(r1+uint(lr1))*p.xs+uint(iy);simdgroup_half8x8 ma[4];simdgroup_half8x8 "
        "mb[2];simdgroup_float8x8 mc[8];FOR_UNROLL(short "
        "i=0;i<8;i++){mc[i]=make_filled_simdgroup_matrix<float,8>(0.0f);}for(uint "
        "loop_k=0u;loop_k<p.ni;loop_k+=uint(NK)){half4x4 "
        "ta;dq4(wp,il,ta);threadgroup_barrier(mem_flags::mem_threadgroup);FOR_UNROLL(short "
        "i=0;i<16;i++){short sx=short(2*il0+i/8);short sy=short((ti/NL0)/8);short "
        "lx=short((ti/NL0)%8);short ly=short(i%8);short "
        "ib=short(8*sx+sy);*(sa+64*ib+8*ly+lx)=(loop_k+uint(16*il0+i)<p.ni)?ta[i/"
        "4][i%4]:half(0.0);}short sx=short(ti%NL1);short sy=short((ti/NL1)/8);short "
        "ly=short((ti/NL1)%8);short ib=short(4*sx+sy);FOR_UNROLL(short i=0;i<8;i++){uint "
        "kk=loop_k+uint(iy+i);*(sb+64*ib+8*ly+i)=kk<p.ni?half(xp[i]):half(0.0);}il=(il+2<QK_NL)?il+"
        "2:il%2;wp=(il<2)?wp+1:wp;xp+=NK;threadgroup_barrier(mem_flags::mem_threadgroup);"
        "threadgroup const half*lsma=sa+4*64*(sg%2);threadgroup const "
        "half*lsmb=sb+2*64*(sg/2);FOR_UNROLL(short "
        "ik=0;ik<NK/8;ik++){simdgroup_barrier(mem_flags::mem_none);FOR_UNROLL(short "
        "i=0;i<4;i++){simdgroup_load(ma[i],lsma+64*i,8,0,false);}simdgroup_barrier(mem_flags::mem_"
        "none);FOR_UNROLL(short "
        "i=0;i<2;i++){simdgroup_load(mb[i],lsmb+64*i,8,0,false);}simdgroup_barrier(mem_flags::mem_"
        "none);FOR_UNROLL(short "
        "i=0;i<8;i++){simdgroup_multiply_accumulate(mc[i],mb[i/"
        "4],ma[i%4],mc[i]);}lsma+=8*64;lsmb+=4*64;}}\n"
        "if(r0+uint(NR0)<=p.no&&r1+uint(NR1)<=p.rows){device "
        "float*C=y+p.yo+(r1+16u*uint(sg>>1))*p.ys+r0+32u*uint(sg&1);FOR_UNROLL(short "
        "i=0;i<8;i++){simdgroup_store(mc[i],C+8*(i%4)+8*p.ys*(i/"
        "4),p.ys,0,false);}}else{threadgroup_barrier(mem_flags::mem_threadgroup);threadgroup "
        "float*tmp=((threadgroup float*)shmem)+32*(sg&1)+16*(sg>>1)*NR0;FOR_UNROLL(short "
        "i=0;i<8;i++){simdgroup_store(mc[i],tmp+8*(i%4)+8*NR0*(i/"
        "4),NR0,0,false);}threadgroup_barrier(mem_flags::mem_threadgroup);if(sg==0){for(uint "
        "j=ti;j<uint(nr1);j+=uint(NR1)){device float*D=y+p.yo+(r1+j)*p.ys+r0;threadgroup "
        "float*C=((threadgroup float*)shmem)+j*NR0;uint i=0u;for(;i+3u<uint(nr0);i+=4u){*((device "
        "float4*)(D+i))=*((threadgroup float4*)(C+i));}for(;i<uint(nr0);i++){D[i]=C[i];}}}}\n"
        "}\n";

/* Interior fast path of matmul_q4k_mm_sg (llama's FC_mul_mm bc_inp/bc_out
 * = false variant): no per-element bounds checks, vectorized 8-wide
 * activation staging, direct device-memory output. Dispatched only when
 * rows%32==0, n_out%64==0, n_in%32==0 and x is 8-float aligned. */
static const char metal_q4k_mm_sg_fast_source[] =
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "#define FOR_UNROLL(x) _Pragma(\"clang loop unroll(full)\") for(x)\n"
        "struct P{uint ni,no,rows,bpr,xo,wo,yo,xs,ys;};\n"
        "struct block_q4_K{half d;half dmin;uchar scales[12];uchar qs[128];};\n"
        "static inline uchar2 sm2(short j,short k,device const uchar*q){return "
        "j<4?uchar2(uchar(q[j+k]&63),uchar(q[j+4+k]&63)):uchar2(uchar((q[j+4+k]&15)|((q[j-4+k]&192)"
        ">>2)),uchar((q[j+4+k]>>4)|((q[j+k]&192)>>2)));}\n"
        "static inline void dq4(device const block_q4_K*xb,short il,thread half4x4&r){device const "
        "uchar*q=xb->qs;short is=(il/4)*2;q=q+(il/4)*32+16*(il&1);il=il&3;uchar2 "
        "sc=sm2(is,il/2,xb->scales);float d=il<2?float(xb->d):float(xb->d)/16.0f;float "
        "mn=float(xb->dmin);float dl=d*float(sc[0]);float ml=mn*float(sc[1]);ushort "
        "mask=il<2?0x0f:0xf0;FOR_UNROLL(short "
        "i=0;i<16;i++){r[i/4][i%4]=half(dl*float(q[i]&mask)-ml);}}\n"
        "kernel void matmul_q4k_mm_sg_fast(device const float*x[[buffer(0)]],device const "
        "uchar*w[[buffer(1)]],device float*y[[buffer(2)]],constant P&p[[buffer(3)]],threadgroup "
        "char*shmem[[threadgroup(0)]],uint3 tg[[threadgroup_position_in_grid]],ushort "
        "ti[[thread_index_in_threadgroup]],ushort sg[[simdgroup_index_in_threadgroup]]){\n"
        "threadgroup half*sa=(threadgroup half*)shmem;threadgroup half*sb=(threadgroup "
        "half*)(shmem+4096);"
        "constexpr short NK=32;constexpr short NL0=2;constexpr short NL1=4;constexpr short "
        "QK_NL=16;"
        "uint r0=tg.y*64u;uint r1=tg.x*32u;"
        "short lr0=short(ti/NL0);short lr1=short(ti/NL1);short il0=short(ti%NL0);short il=il0;"
        "device const block_q4_K*wp=(device const block_q4_K*)(w+p.wo)+(r0+uint(lr0))*p.bpr;"
        "short iy=short(8*(ti%NL1));"
        "device const float*xp=x+p.xo+min(r1+uint(lr1),p.rows-1u)*p.xs+uint(iy);"
        "simdgroup_half8x8 ma[4];simdgroup_half8x8 mb[2];simdgroup_float8x8 mc[8];"
        "FOR_UNROLL(short i=0;i<8;i++){mc[i]=make_filled_simdgroup_matrix<float,8>(0.0f);}"
        "short sx=short(ti%NL1);short sy=short((ti/NL1)/8);short lyb=short((ti/NL1)%8);short "
        "ibb=short(4*sx+sy);"
        "for(uint loop_k=0u;loop_k<p.ni;loop_k+=uint(NK)){"
        "half4x4 ta;dq4(wp,il,ta);"
        "threadgroup_barrier(mem_flags::mem_threadgroup);"
        "FOR_UNROLL(short i=0;i<16;i++){short ax=short(2*il0+i/8);short ay=short((ti/NL0)/8);short "
        "lx=short((ti/NL0)%8);short ly=short(i%8);short "
        "ib=short(8*ax+ay);*(sa+64*ib+8*ly+lx)=ta[i/4][i%4];}"
        "*(threadgroup half2x4*)(sb+64*ibb+8*lyb)=half2x4(*((device const float2x4*)xp));"
        "il=(il+2<QK_NL)?il+2:il%2;wp=(il<2)?wp+1:wp;xp+=NK;"
        "threadgroup_barrier(mem_flags::mem_threadgroup);"
        "threadgroup const half*lsma=sa+4*64*(sg%2);threadgroup const half*lsmb=sb+2*64*(sg/2);"
        "FOR_UNROLL(short ik=0;ik<NK/8;ik++){"
        "simdgroup_barrier(mem_flags::mem_none);"
        "FOR_UNROLL(short i=0;i<4;i++){simdgroup_load(ma[i],lsma+64*i,8,0,false);}"
        "simdgroup_barrier(mem_flags::mem_none);"
        "FOR_UNROLL(short i=0;i<2;i++){simdgroup_load(mb[i],lsmb+64*i,8,0,false);}"
        "simdgroup_barrier(mem_flags::mem_none);"
        "FOR_UNROLL(short i=0;i<8;i++){simdgroup_multiply_accumulate(mc[i],mb[i/4],ma[i%4],mc[i]);}"
        "lsma+=8*64;lsmb+=4*64;}}"
        "if(r1+32u<=p.rows){device float*C=y+p.yo+(r1+16u*uint(sg>>1))*p.ys+r0+32u*uint(sg&1);"
        "FOR_UNROLL(short "
        "i=0;i<8;i++){simdgroup_store(mc[i],C+8*(i%4)+8*p.ys*(i/4),p.ys,0,false);}}"
        "else{threadgroup_barrier(mem_flags::mem_threadgroup);threadgroup float*T=(threadgroup "
        "float*)shmem;"
        "FOR_UNROLL(short "
        "i=0;i<8;i++){simdgroup_store(mc[i],T+32*(sg&1)+1024*(sg>>1)+8*(i%4)+512*(i/"
        "4),64,0,false);}"
        "threadgroup_barrier(mem_flags::mem_threadgroup);for(uint "
        "j=uint(ti);j<(p.rows-r1)*64u;j+=128u)"
        "{y[p.yo+(r1+j/64u)*p.ys+r0+j%64u]=T[j];}}"
        "}\n";

/* matmul_q4k_mm_sg_fast's structure on Q6_K blocks (210 bytes), for full
 * 64-output x 32-row tiles (#17): 32-wide K steps, every thread
 * dequantizes. dq6 gives the 16 consecutive elements of il as
 * matmul_q6k_sg_fast does, half(d * sc * (q - 32)). */
static const char metal_q6k_mm32_source[] =
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "#define FOR_UNROLL(x) _Pragma(\"clang loop unroll(full)\") for(x)\n"
        "struct P{uint ni,no,rows,bpr,xo,wo,yo,xs,ys;};\n"
        "static inline void dq6(device const uchar*b,short il,thread half4x4&r){short k0=16*il;"
        "short hi=k0>>7;short ih=k0&127;short st=ih>>5;short l0=ih&31;"
        "device const uchar*ql=b+hi*64+l0+((st&1)?32:0);device const uchar*qh=b+128+hi*32+l0;"
        "float d=float(as_type<half>(ushort(ushort(b[208])|(ushort(b[209])<<8))));"
        "float dl=d*float(as_type<char>(b[192+hi*8+(l0>>4)+2*st]));short qs=(st>=2)?4:0;short "
        "hs=2*st;"
        "FOR_UNROLL(short i=0;i<16;i++){int "
        "q=int(((uint(ql[i])>>qs)&15u)|(((uint(qh[i])>>hs)&3u)<<4u));"
        "r[i/4][i%4]=half(dl*float(q-32));}}\n"
        "kernel void matmul_q6k_mm_sg_fast(device const float*x[[buffer(0)]],device const "
        "uchar*w[[buffer(1)]],device float*y[[buffer(2)]],constant P&p[[buffer(3)]],threadgroup "
        "char*shmem[[threadgroup(0)]],uint3 tg[[threadgroup_position_in_grid]],ushort "
        "ti[[thread_index_in_threadgroup]],ushort sg[[simdgroup_index_in_threadgroup]]){\n"
        "threadgroup half*sa=(threadgroup half*)shmem;threadgroup half*sb=(threadgroup "
        "half*)(shmem+4096);"
        "constexpr short NK=32;constexpr short NL0=2;constexpr short NL1=4;constexpr short "
        "QK_NL=16;"
        "uint r0=tg.y*64u;uint r1=tg.x*32u;"
        "short lr0=short(ti/NL0);short lr1=short(ti/NL1);short il0=short(ti%NL0);short il=il0;"
        "device const uchar*wp=w+p.wo+(r0+uint(lr0))*p.bpr*210u;"
        "short iy=short(8*(ti%NL1));"
        "device const float*xp=x+p.xo+min(r1+uint(lr1),p.rows-1u)*p.xs+uint(iy);"
        "simdgroup_half8x8 ma[4];simdgroup_half8x8 mb[2];simdgroup_float8x8 mc[8];"
        "FOR_UNROLL(short i=0;i<8;i++){mc[i]=make_filled_simdgroup_matrix<float,8>(0.0f);}"
        "short sx=short(ti%NL1);short sy=short((ti/NL1)/8);short lyb=short((ti/NL1)%8);short "
        "ibb=short(4*sx+sy);"
        "for(uint loop_k=0u;loop_k<p.ni;loop_k+=uint(NK)){"
        "half4x4 ta;dq6(wp,il,ta);"
        "threadgroup_barrier(mem_flags::mem_threadgroup);"
        "FOR_UNROLL(short i=0;i<16;i++){short ax=short(2*il0+i/8);short ay=short((ti/NL0)/8);short "
        "lx=short((ti/NL0)%8);short ly=short(i%8);short "
        "ib=short(8*ax+ay);*(sa+64*ib+8*ly+lx)=ta[i/4][i%4];}"
        "*(threadgroup half2x4*)(sb+64*ibb+8*lyb)=half2x4(*((device const float2x4*)xp));"
        "il=(il+2<QK_NL)?il+2:il%2;wp=(il<2)?wp+210:wp;xp+=NK;"
        "threadgroup_barrier(mem_flags::mem_threadgroup);"
        "threadgroup const half*lsma=sa+4*64*(sg%2);threadgroup const half*lsmb=sb+2*64*(sg/2);"
        "FOR_UNROLL(short ik=0;ik<NK/8;ik++){"
        "simdgroup_barrier(mem_flags::mem_none);"
        "FOR_UNROLL(short i=0;i<4;i++){simdgroup_load(ma[i],lsma+64*i,8,0,false);}"
        "simdgroup_barrier(mem_flags::mem_none);"
        "FOR_UNROLL(short i=0;i<2;i++){simdgroup_load(mb[i],lsmb+64*i,8,0,false);}"
        "simdgroup_barrier(mem_flags::mem_none);"
        "FOR_UNROLL(short i=0;i<8;i++){simdgroup_multiply_accumulate(mc[i],mb[i/4],ma[i%4],mc[i]);}"
        "lsma+=8*64;lsmb+=4*64;}}"
        "if(r1+32u<=p.rows){device float*C=y+p.yo+(r1+16u*uint(sg>>1))*p.ys+r0+32u*uint(sg&1);"
        "FOR_UNROLL(short "
        "i=0;i<8;i++){simdgroup_store(mc[i],C+8*(i%4)+8*p.ys*(i/4),p.ys,0,false);}}"
        "else{threadgroup_barrier(mem_flags::mem_threadgroup);threadgroup float*T=(threadgroup "
        "float*)shmem;"
        "FOR_UNROLL(short "
        "i=0;i<8;i++){simdgroup_store(mc[i],T+32*(sg&1)+1024*(sg>>1)+8*(i%4)+512*(i/"
        "4),64,0,false);}"
        "threadgroup_barrier(mem_flags::mem_threadgroup);for(uint "
        "j=uint(ti);j<(p.rows-r1)*64u;j+=128u)"
        "{y[p.yo+(r1+j/64u)*p.ys+r0+j%64u]=T[j];}}"
        "}\n";

static const char metal_q6k_source[] =
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "struct Params{uint "
        "n_in,n_out,rows,blocks_per_row,x_offset,w_byte_offset,y_offset,x_row_stride,y_row_stride;}"
        ";\n"
        "static inline float load_f16(device const uchar*w,uint o){ushort "
        "b=ushort(w[o])|(ushort(w[o+1])<<8);return float(as_type<half>(b));}\n"
        "static inline int load_i8(device const uchar*w,uint o){uint u=uint(w[o]);return "
        "u<128u?int(u):int(u)-256;}\n"
        "static inline float deq6(device const uchar*w,constant Params&p,uint row,uint k){uint "
        "br=k/256u,ib=k-br*256u,hi=ib/128u,ih=ib-hi*128u,stream=ih/32u,l=ih-stream*32u,is=l/"
        "16u;uint bo=p.w_byte_offset+(row*p.blocks_per_row+br)*210u;uint "
        "qlb=bo+hi*64u,qhb=bo+128u+hi*32u,scb=bo+192u+hi*8u;uint "
        "ql0=uint(w[qlb+l]),ql1=uint(w[qlb+l+32u]),qh=uint(w[qhb+l]);uint "
        "qu,si;if(stream==0u){qu=(ql0&15u)|(((qh>>0u)&3u)<<4u);si=is+0u;}else "
        "if(stream==1u){qu=(ql1&15u)|(((qh>>2u)&3u)<<4u);si=is+2u;}else "
        "if(stream==2u){qu=(ql0>>4u)|(((qh>>4u)&3u)<<4u);si=is+4u;}else{qu=(ql1>>4u)|(((qh>>6u)&3u)"
        "<<4u);si=is+6u;}float d=load_f16(w,bo+208u);return "
        "d*float(load_i8(w,scb+si))*float(int(qu)-32);}\n"
        "kernel void matvec_q6k(device const float*x[[buffer(0)]],device const "
        "uchar*w[[buffer(1)]],device float*y[[buffer(2)]],constant Params&p[[buffer(3)]],uint3 "
        "tg[[threadgroup_position_in_grid]],uint lid[[thread_index_in_threadgroup]]){threadgroup "
        "float partial[256];uint row=tg.x,batch=tg.y;if(row>=p.n_out||batch>=p.rows){return;}float "
        "sum=0.0f;for(uint "
        "k=lid;k<p.n_in;k+=256u){sum+=x[p.x_offset+batch*p.x_row_stride+k]*deq6(w,p,row,k);}"
        "partial[lid]=sum;threadgroup_barrier(mem_flags::mem_threadgroup);for(uint "
        "stride=128u;stride>0u;stride>>=1u){if(lid<stride){partial[lid]+=partial[lid+stride];}"
        "threadgroup_barrier(mem_flags::mem_threadgroup);}if(lid==0u){y[p.y_offset+batch*p.y_row_"
        "stride+row]=partial[0];}}\n";

/* 64x32 q6k GEMM (NK=8). Weight staging dequants one 8-elem k-run per
 * thread: d and the int8 scale load once per run instead of per element. */
static const char metal_q6k_mm_sg_source[] =
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "struct Params{uint "
        "n_in,n_out,rows,blocks_per_row,x_offset,w_byte_offset,y_offset,x_row_stride,y_row_stride;}"
        ";\n"
        "static inline float load_f16(device const uchar*w,uint o){ushort "
        "b=ushort(w[o])|(ushort(w[o+1])<<8);return float(as_type<half>(b));}\n"
        "static inline int load_i8(device const uchar*w,uint o){uint u=uint(w[o]);return "
        "u<128u?int(u):int(u)-256;}\n"
        "kernel void matmul_q6k_sg(device const float*x[[buffer(0)]],device const "
        "uchar*w[[buffer(1)]],device float*y[[buffer(2)]],constant Params&p[[buffer(3)]],uint3 "
        "tg[[threadgroup_position_in_grid]],uint lid[[thread_index_in_threadgroup]],ushort "
        "sgitg[[simdgroup_index_in_threadgroup]],ushort tiisg[[thread_index_in_simdgroup]]){"
        "threadgroup half as[512];threadgroup half bs[256];threadgroup float cs[256];"
        "uint b0=tg.y*64u,o0=tg.x*32u;"
        "simdgroup_float8x8 acc[8];for(uint "
        "i=0u;i<8u;i++)acc[i]=make_filled_simdgroup_matrix<float,8>(0.0f);"
        "for(uint k0=0u;k0<p.n_in;k0+=8u){"
        "for(uint i=lid;i<512u;i+=128u){uint "
        "r=i/"
        "8u,c=i%8u;as[i]=(b0+r<p.rows&&k0+c<p.n_in)?half(x[p.x_offset+(b0+r)*p.x_row_stride+k0+c]):"
        "half(0.0f);}"
        "if(lid<32u){uint o=o0+lid;"
        "if(o<p.n_out&&k0<p.n_in){"
        "uint br=k0>>8u,ib=k0&255u,hi=ib>>7u,ih=ib&127u,stream=ih>>5u,l=ih&31u,is=l>>4u;"
        "uint bo=p.w_byte_offset+(o*p.blocks_per_row+br)*210u;"
        "uint "
        "qlb=bo+hi*64u+l+((stream&1u)?32u:0u),qhb=bo+128u+hi*32u+l,scb=bo+192u+hi*8u+is+2u*stream;"
        "uint qsh=(stream>=2u)?4u:0u,hsh=2u*stream;"
        "float ds=load_f16(w,bo+208u)*float(load_i8(w,scb));"
        "for(uint j=0u;j<8u;j++){uint "
        "qu=((uint(w[qlb+j])>>qsh)&15u)|(((uint(w[qhb+j])>>hsh)&3u)<<4u);bs[j*32u+lid]=half(ds*"
        "float(int(qu)-32));}"
        "}else{for(uint j=0u;j<8u;j++)bs[j*32u+lid]=half(0.0f);}}"
        "threadgroup_barrier(mem_flags::mem_threadgroup);"
        "simdgroup_half8x8 ma,mb;simdgroup_load(mb,bs+8u*uint(sgitg),32);"
        "for(uint "
        "rj=0u;rj<8u;rj++){simdgroup_load(ma,as+64u*rj,8);simdgroup_multiply_accumulate(acc[rj],ma,"
        "mb,acc[rj]);}"
        "threadgroup_barrier(mem_flags::mem_threadgroup);}"
        "threadgroup float*lcs=cs+64u*uint(sgitg);"
        "for(uint "
        "rj=0u;rj<8u;rj++){simdgroup_store(acc[rj],lcs,8);simdgroup_barrier(mem_flags::mem_"
        "threadgroup);"
        "for(uint i=uint(tiisg);i<64u;i+=32u){uint "
        "r=i/"
        "8u,c=i%8u;if(b0+8u*rj+r<p.rows&&o0+8u*uint(sgitg)+c<p.n_out)y[p.y_offset+(b0+8u*rj+r)*p.y_"
        "row_stride+o0+8u*uint(sgitg)+c]=lcs[i];}simdgroup_barrier(mem_flags::mem_threadgroup);}}"
        "\n";

/* Interior fast path of matmul_q6k_sg: no per-element bounds checks,
 * vectorized 8-wide activation staging, direct simdgroup_store output.
 * Dispatched only when rows%64==0, n_out%32==0 and x is 8-float aligned. */
static const char metal_q6k_mm_sg_fast_source[] =
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "struct Params{uint "
        "n_in,n_out,rows,blocks_per_row,x_offset,w_byte_offset,y_offset,x_row_stride,y_row_stride;}"
        ";\n"
        "static inline float load_f16(device const uchar*w,uint o){ushort "
        "b=ushort(w[o])|(ushort(w[o+1])<<8);return float(as_type<half>(b));}\n"
        "static inline int load_i8(device const uchar*w,uint o){uint u=uint(w[o]);return "
        "u<128u?int(u):int(u)-256;}\n"
        "kernel void matmul_q6k_sg_fast(device const float*x[[buffer(0)]],device const "
        "uchar*w[[buffer(1)]],device float*y[[buffer(2)]],constant Params&p[[buffer(3)]],uint3 "
        "tg[[threadgroup_position_in_grid]],uint lid[[thread_index_in_threadgroup]],ushort "
        "sgitg[[simdgroup_index_in_threadgroup]],ushort tiisg[[thread_index_in_simdgroup]]){"
        "threadgroup half as[512];threadgroup half bs[256];"
        "uint b0=tg.y*64u,o0=tg.x*32u;"
        "simdgroup_float8x8 acc[8];for(uint "
        "i=0u;i<8u;i++)acc[i]=make_filled_simdgroup_matrix<float,8>(0.0f);"
        "for(uint k0=0u;k0<p.n_in;k0+=8u){"
        "if(lid<64u){*(threadgroup half2x4*)(as+lid*8u)=half2x4(*((device const "
        "float2x4*)(x+p.x_offset+(b0+lid)*p.x_row_stride+k0)));}"
        "else if(lid<96u){uint o=o0+(lid-64u);"
        "uint br=k0>>8u,ib=k0&255u,hi=ib>>7u,ih=ib&127u,stream=ih>>5u,l=ih&31u,is=l>>4u;"
        "uint bo=p.w_byte_offset+(o*p.blocks_per_row+br)*210u;"
        "uint "
        "qlb=bo+hi*64u+l+((stream&1u)?32u:0u),qhb=bo+128u+hi*32u+l,scb=bo+192u+hi*8u+is+2u*stream;"
        "uint qsh=(stream>=2u)?4u:0u,hsh=2u*stream;"
        "float ds=load_f16(w,bo+208u)*float(load_i8(w,scb));"
        "for(uint j=0u;j<8u;j++){uint "
        "qu=((uint(w[qlb+j])>>qsh)&15u)|(((uint(w[qhb+j])>>hsh)&3u)<<4u);bs[j*32u+(lid-64u)]=half("
        "ds*float(int(qu)-32));}}"
        "threadgroup_barrier(mem_flags::mem_threadgroup);"
        "simdgroup_half8x8 ma,mb;simdgroup_load(mb,bs+8u*uint(sgitg),32);"
        "for(uint "
        "rj=0u;rj<8u;rj++){simdgroup_load(ma,as+64u*rj,8);simdgroup_multiply_accumulate(acc[rj],ma,"
        "mb,acc[rj]);}"
        "threadgroup_barrier(mem_flags::mem_threadgroup);}"
        "uint co=p.y_offset+o0+8u*uint(sgitg);"
        "for(uint "
        "rj=0u;rj<8u;rj++){simdgroup_store(acc[rj],y+co+(b0+8u*rj)*p.y_row_stride,p.y_row_stride,0,"
        "false);}"
        "}\n";

/* llama.cpp kernel_mul_mv_q6_K_f32 structure: 2 simdgroups/threadgroup, 2 rows
 * per simdgroup (grid.x*4 rows/tg), block-coherent loads — each
 * thread handles 16 elems of a 256-block via 4 contiguous ql/qh bytes and
 * int8 scales, one f16 d per (row,block). */
static const char metal_mv_q6k_body[] =
        "static inline float q6h(device const uchar*w,uint o){ushort "
        "b=ushort(w[o])|(ushort(w[o+1])<<8);return float(as_type<half>(b));}\n"
        "static inline int q6i8(device const uchar*w,uint o){uint u=uint(w[o]);return "
        "u<128u?int(u):int(u)-256;}\n"
        "static inline void mv_q6k(device const float*x,device const uchar*w,device "
        "float*y,constant P&p,uint tx,uint b,ushort ti,ushort sg){"
        "uint fr=(tx*2u+uint(sg))*2u;if(fr>=p.no||b>=p.rows)return;"
        "uint nb=p.ni>>8u;"
        "ushort tid=ti/2u,ix=ti%2u,ip=tid/8u,il=tid%8u,l0=4u*il,is=8u*ip+l0/16u;"
        "uint yoff=128u*uint(ip)+uint(l0),qlo=64u*uint(ip)+uint(l0),qho=32u*uint(ip)+uint(l0);"
        "device const float*yy=x+p.xo+b*p.xs+yoff;"
        "float yl[16];float sf0=0.0f,sf1=0.0f;"
        "for(uint i=uint(ix);i<nb;i+=2u){"
        "device const float*yv=yy+i*256u;"
        "for(ushort "
        "l=0;l<4;l++){yl[4*l+0]=yv[l];yl[4*l+1]=yv[l+32];yl[4*l+2]=yv[l+64];yl[4*l+3]=yv[l+96];}"
        "for(uint rr=0u;rr<2u;rr++){uint row=fr+rr;if(row>=p.no)break;"
        "uint bo=p.wo+(row*p.bpr+i)*210u;"
        "device const uchar*q1=w+bo+qlo;device const uchar*q2=q1+32u;device const "
        "uchar*qh=w+bo+128u+qho;"
        "uint sco=bo+192u+uint(is);"
        "float4 s=float4(0.0f);"
        "for(ushort l=0;l<4;l++){uint hb=uint(qh[l]);"
        "s[0]+=yl[4*l+0]*float(int((uint(q1[l])&15u)|((hb&3u)<<4u))-32);"
        "s[1]+=yl[4*l+1]*float(int((uint(q2[l])&15u)|(((hb>>2u)&3u)<<4u))-32);"
        "s[2]+=yl[4*l+2]*float(int((uint(q1[l])>>4u)|(((hb>>4u)&3u)<<4u))-32);"
        "s[3]+=yl[4*l+3]*float(int((uint(q2[l])>>4u)|(((hb>>6u)&3u)<<4u))-32);}"
        "float "
        "acc=q6h(w,bo+208u)*(s[0]*float(q6i8(w,sco))+s[1]*float(q6i8(w,sco+2u))+s[2]*float(q6i8(w,"
        "sco+4u))+"
        "s[3]*float(q6i8(w,sco+6u)));"
        "if(rr==0u)sf0+=acc;else sf1+=acc;}}"
        "float a0=simd_sum(sf0),a1=simd_sum(sf1);"
        "if(ti==0){uint o=p.yo+b*p.ys+fr;y[o]=a0;if(fr+1u<p.no)y[o+1u]=a1;}}\n";

static const char metal_q6k_n4_kernel[] =
        "kernel void matvec_q6k_n4(device const float*x[[buffer(0)]],device const "
        "uchar*w[[buffer(1)]],device float*y[[buffer(2)]],constant P&p[[buffer(3)]],uint3 "
        "tg[[threadgroup_position_in_grid]],ushort ti[[thread_index_in_simdgroup]],ushort "
        "sg[[simdgroup_index_in_threadgroup]]){mv_q6k(x,w,y,p,tg.x,tg.y,ti,sg);}\n";

/* fused->hadamard_rotate (see struct geist_hadamard_args): one
 * threadgroup per (block, row) stages the block in threadgroup memory —
 * gathered through the grouped-value permutation and sign-multiplied on
 * the forward pass — runs the log2(block) butterfly passes, and writes
 * the orthonormal result, sign-multiplied on the inverse pass. A block is
 * read completely before any of it is written, so y may alias x whenever
 * there is no permutation. */
static const char metal_hadamard_source[] =
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "struct H{uint width,block,hd,nk,rep,inv,sgn,xo,xs,yo,ys,so;};\n"
        "kernel void hadamard_rows(device const float*x[[buffer(0)]],device "
        "float*y[[buffer(1)]],device const float*s[[buffer(2)]],constant H&p[[buffer(3)]],"
        "threadgroup float*sh[[threadgroup(0)]],uint3 tg[[threadgroup_position_in_grid]],uint "
        "ti[[thread_index_in_threadgroup]],uint3 ntg[[threads_per_threadgroup]]){"
        "uint row=tg.y,base=tg.x*p.block,nt=ntg.x;"
        "device const float*xr=x+p.xo+row*p.xs;device float*yr=y+p.yo+row*p.ys;"
        "for(uint j=ti;j<p.block;j+=nt){uint g=base+j,src=g;"
        "if(p.rep>1u){uint d=g%p.hd,q=g/p.hd,r=q%p.rep,k=q/p.rep;src=d+p.hd*(k+p.nk*r);}"
        "float v=xr[src];if(p.sgn!=0u&&p.inv==0u)v*=s[p.so+g];sh[j]=v;}"
        "threadgroup_barrier(mem_flags::mem_threadgroup);"
        "for(uint len=1u;len<p.block;len<<=1u){"
        "for(uint i=ti;i<p.block/2u;i+=nt){uint j=(i/len)*2u*len+(i%len);"
        "float a=sh[j],b=sh[j+len];sh[j]=a+b;sh[j+len]=a-b;}"
        "threadgroup_barrier(mem_flags::mem_threadgroup);}"
        "float sc=rsqrt(float(p.block));"
        "for(uint j=ti;j<p.block;j+=nt){float v=sh[j]*sc;"
        "if(p.sgn!=0u&&p.inv!=0u)v*=s[p.so+base+j];yr[base+j]=v;}}\n"
        /* hadamard_rows4: the same transform for blocks of 128..4096 with
         * four elements per thread (element e*nt+ti), nt = block/4. The
         * butterflies run in the same stage order and pair the same values
         * as hadamard_rows, so the result is bit-identical: strides below
         * 32 through simd shuffles, up to nt through threadgroup memory,
         * nt and 2*nt in registers. */
        "kernel void hadamard_rows4(device const float*x[[buffer(0)]],device "
        "float*y[[buffer(1)]],device const float*s[[buffer(2)]],constant "
        "H&p[[buffer(3)]],threadgroup float*sh[[threadgroup(0)]],uint3 "
        "tg[[threadgroup_position_in_grid]],uint ti[[thread_index_in_threadgroup]],uint3 "
        "ntg[[threads_per_threadgroup]]){"
        "uint row=tg.y,base=tg.x*p.block,nt=ntg.x;"
        "device const float*xr=x+p.xo+row*p.xs;device float*yr=y+p.yo+row*p.ys;float v[4];"
        "for(uint e=0u;e<4u;e++){uint g=base+e*nt+ti,src=g;"
        "if(p.rep>1u){uint d=g%p.hd,q=g/p.hd,r=q%p.rep,k=q/p.rep;src=d+p.hd*(k+p.nk*r);}"
        "float a=xr[src];if(p.sgn!=0u&&p.inv==0u)a*=s[p.so+g];v[e]=a;}"
        "for(uint len=1u;len<32u;len<<=1u){for(uint e=0u;e<4u;e++){float "
        "o=simd_shuffle_xor(v[e],ushort(len));v[e]=(ti&len)==0u?v[e]+o:o-v[e];}}"
        "for(uint len=32u;len<nt;len<<=1u){for(uint e=0u;e<4u;e++)sh[e*nt+ti]=v[e];"
        "threadgroup_barrier(mem_flags::mem_threadgroup);"
        "for(uint e=0u;e<4u;e++){float o=sh[e*nt+(ti^len)];v[e]=(ti&len)==0u?v[e]+o:o-v[e];}"
        "threadgroup_barrier(mem_flags::mem_threadgroup);}"
        "float a0=v[0]+v[1],a1=v[0]-v[1],a2=v[2]+v[3],a3=v[2]-v[3];"
        "v[0]=a0+a2;v[2]=a0-a2;v[1]=a1+a3;v[3]=a1-a3;"
        "float sc=rsqrt(float(p.block));"
        "for(uint e=0u;e<4u;e++){uint j=base+e*nt+ti;float "
        "o=v[e]*sc;if(p.sgn!=0u&&p.inv!=0u)o*=s[p.so+j];yr[j]=o;}}\n";

static const char metal_elem_source[] =
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "struct Bin{uint "
        "rows,cols,a_offset,b_offset,y_offset,a_row_stride,b_row_stride,y_row_stride;};\n"
        "struct Sc{uint rows,cols,x_offset,y_offset,x_row_stride,y_row_stride;float scale;};\n"
        "static inline float gelu_tanh_f(float x){return "
        "0.5f*x*(1.0f+tanh(clamp(0.7978845608028654f*(x+0.044715f*x*x*x),-10.0f,10.0f)));}\n"
        "kernel void gelu_rows(device const float*x[[buffer(0)]],device "
        "float*y[[buffer(1)]],constant Sc&p[[buffer(2)]],uint gid[[thread_position_in_grid]]){\n"
        "    uint total=p.rows*p.cols;if(gid>=total){return;}uint "
        "r=gid/"
        "p.cols,c=gid-r*p.cols;y[p.y_offset+r*p.y_row_stride+c]=gelu_tanh_f(x[p.x_offset+r*p.x_row_"
        "stride+c]);\n"
        "}\n"
        "kernel void mul_rows(device const float*a[[buffer(0)]],device const "
        "float*b[[buffer(1)]],device float*y[[buffer(2)]],constant Bin&p[[buffer(3)]],uint "
        "gid[[thread_position_in_grid]]){\n"
        "    uint total=p.rows*p.cols;if(gid>=total){return;}uint "
        "r=gid/"
        "p.cols,c=gid-r*p.cols;y[p.y_offset+r*p.y_row_stride+c]=a[p.a_offset+r*p.a_row_stride+c]*b["
        "p.b_offset+r*p.b_row_stride+c];\n"
        "}\n"
        "kernel void gelu_mul_rows(device const float*a[[buffer(0)]],device const "
        "float*b[[buffer(1)]],device float*y[[buffer(2)]],constant Bin&p[[buffer(3)]],uint "
        "gid[[thread_position_in_grid]]){\n"
        "    uint total=p.rows*p.cols;if(gid>=total){return;}uint "
        "r=gid/p.cols,c=gid-r*p.cols;float av=a[p.a_offset+r*p.a_row_stride+c];float "
        "bv=b[p.b_offset+r*p.b_row_stride+c];y[p.y_offset+r*p.y_row_stride+c]=gelu_tanh_f(av)*bv;\n"
        "}\n"
        "kernel void add_rows(device const float*a[[buffer(0)]],device const "
        "float*b[[buffer(1)]],device float*y[[buffer(2)]],constant Bin&p[[buffer(3)]],uint "
        "gid[[thread_position_in_grid]]){\n"
        "    uint total=p.rows*p.cols;if(gid>=total){return;}uint "
        "r=gid/"
        "p.cols,c=gid-r*p.cols;y[p.y_offset+r*p.y_row_stride+c]=a[p.a_offset+r*p.a_row_stride+c]+b["
        "p.b_offset+r*p.b_row_stride+c];\n"
        "}\n"
        "kernel void scale_rows(device const float*x[[buffer(0)]],device "
        "float*y[[buffer(1)]],constant Sc&p[[buffer(2)]],uint gid[[thread_position_in_grid]]){\n"
        "    uint total=p.rows*p.cols;if(gid>=total){return;}uint "
        "r=gid/"
        "p.cols,c=gid-r*p.cols;y[p.y_offset+r*p.y_row_stride+c]=x[p.x_offset+r*p.x_row_stride+c]*p."
        "scale;\n"
        "}\n";

static const char metal_qgate_source[] =
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "struct QG{uint rows,heads,hd,jo,qo,go,js,qs,gs;};\n"
        "struct Bin{uint "
        "rows,cols,a_offset,b_offset,y_offset,a_row_stride,b_row_stride,y_row_stride;};\n"
        "kernel void qgate_split(device const float*jv[[buffer(0)]],device float*q[[buffer(1)]],"
        "device float*g[[buffer(2)]],constant QG&p[[buffer(3)]],uint i[[thread_position_in_grid]]){"
        "uint total=p.rows*p.heads*p.hd;if(i>=total)return;uint "
        "d=i%p.hd,x=i/p.hd,h=x%p.heads,r=x/p.heads;"
        "uint src=p.jo+r*p.js+h*2u*p.hd+d;q[p.qo+r*p.qs+h*p.hd+d]=jv[src];"
        "g[p.go+r*p.gs+h*p.hd+d]=jv[src+p.hd];}\n"
        "kernel void sigmoid_mul_rows(device const float*x[[buffer(0)]],device const "
        "float*g[[buffer(1)]],device float*y[[buffer(2)]],constant Bin&p[[buffer(3)]],uint "
        "i[[thread_position_in_grid]]){uint total=p.rows*p.cols;if(i>=total)return;uint "
        "r=i/p.cols,c=i-r*p.cols;float v=g[p.b_offset+r*p.b_row_stride+c];"
        "y[p.y_offset+r*p.y_row_stride+c]=x[p.a_offset+r*p.a_row_stride+c]/(1.0f+exp(-v));}\n";

static const char metal_silu_source[] =
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "struct Sc{uint rows,cols,x_offset,y_offset,x_row_stride,y_row_stride;float scale;};\n"
        "kernel void silu_rows(device const float*x[[buffer(0)]],device "
        "float*y[[buffer(1)]],constant Sc&p[[buffer(2)]],uint gid[[thread_position_in_grid]]){uint "
        "total=p.rows*p.cols;if(gid>=total)return;uint r=gid/p.cols,c=gid-r*p.cols;float "
        "v=x[p.x_offset+r*p.x_row_stride+c];y[p.y_offset+r*p.y_row_stride+c]=v/(1.0f+exp(-v));}\n"
        /* BitNet b1.58 2B-4T FFN activation, max(x, 0)^2 — relu_squared_fp32. */
        "kernel void relu2_rows(device const float*x[[buffer(0)]],device "
        "float*y[[buffer(1)]],constant Sc&p[[buffer(2)]],uint gid[[thread_position_in_grid]]){uint "
        "total=p.rows*p.cols;if(gid>=total)return;uint r=gid/p.cols,c=gid-r*p.cols;float "
        "v=max(x[p.x_offset+r*p.x_row_stride+c],0.0f);y[p.y_offset+r*p.y_row_stride+c]=v*v;}\n"
        /* Fused SwiGLU epilogue: silu(a)*b in one pass,
         * exact same silu formula as silu_rows above so the fused path
         * is bit-identical to silu+mul. Field layout matches struct Bin
         * (metal_binary_rows_params). */
        "struct BinS{uint rows,cols,a_offset,b_offset,y_offset,a_row_stride,b_row_stride,y_row_"
        "stride;};\n"
        "kernel void silu_mul_rows(device const float*a[[buffer(0)]],device const "
        "float*b[[buffer(1)]],device float*y[[buffer(2)]],constant BinS&p[[buffer(3)]],uint "
        "gid[[thread_position_in_grid]]){uint total=p.rows*p.cols;if(gid>=total)return;uint "
        "r=gid/p.cols,c=gid-r*p.cols;float av=a[p.a_offset+r*p.a_row_stride+c];float "
        "bv=b[p.b_offset+r*p.b_row_stride+c];y[p.y_offset+r*p.y_row_stride+c]=(av/(1.0f+exp(-av)))"
        "*bv;}\n";

static const char metal_elem_simd_source[] =
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "struct Rows{uint rows,cols,x_offset,w_offset,y_offset,x_row_stride,y_row_stride;float "
        "eps;};\n"
        "struct Post{uint "
        "rows,cols,residual_offset,x_offset,w_offset,y_offset,residual_row_stride,x_row_stride,y_"
        "row_stride;float eps;};\n"
        /* A single row (decode) of up to 4 x threads elements, see
         * metal_rows_threads, keeps x, w (and res) in registers: each thread
         * issues all of its loads before the reduction instead of loading x
         * once to reduce and again to scale (#17). Several rows (prefill)
         * keep the loop below and its bits. The partials need no zeroing:
         * every simdgroup writes its own before the barrier. */
        "#define RN_HEAD(XO,XS) threadgroup float partial[32];if(row>=p.rows)return;"
        "const uint xb=p.XO+row*p.XS;\n"
        "#define RN_SUM float sum=simd_sum(ss);if(sl==0u)partial[sg]=sum;"
        "threadgroup_barrier(mem_flags::mem_threadgroup);float total=0.0f;"
        "for(uint i=0u;i<nt/32u;i++)total+=partial[i];"
        "float inv=rsqrt(total/float(p.cols)+p.eps);\n"
        "kernel void rmsnorm_rows_simd(device const float*x[[buffer(0)]],device const "
        "float*w[[buffer(1)]],device float*y[[buffer(2)]],constant Rows&p[[buffer(3)]],uint "
        "row[[threadgroup_position_in_grid]],uint lid[[thread_index_in_threadgroup]],uint "
        "nt[[threads_per_threadgroup]],uint "
        "sg[[simdgroup_index_in_threadgroup]],uint sl[[thread_index_in_simdgroup]]){\n"
        "RN_HEAD(x_offset,x_row_stride)const uint yb=p.y_offset+row*p.y_row_stride;"
        "if(p.rows==1u&&p.cols<=4u*nt){float xv[4],wv[4];for(uint k=0u;k<4u;k++){const uint "
        "c=lid+k*nt;"
        "xv[k]=c<p.cols?x[xb+c]:0.0f;wv[k]=c<p.cols?w[p.w_offset+c]:0.0f;}float ss=0.0f;"
        "for(uint k=0u;k<4u;k++){if(lid+k*nt<p.cols)ss+=xv[k]*xv[k];}RN_SUM "
        "for(uint k=0u;k<4u;k++){const uint c=lid+k*nt;if(c<p.cols)y[yb+c]=xv[k]*inv*wv[k];}"
        "return;}"
        "float ss=0.0f;for(uint c=lid;c<p.cols;c+=nt){float v=x[xb+c];ss+=v*v;}RN_SUM "
        "for(uint c=lid;c<p.cols;c+=nt)y[yb+c]=x[xb+c]*inv*w[p.w_offset+c];}\n"
        "kernel void rmsnorm_add_rows_simd(device const float*res[[buffer(0)]],device const "
        "float*x[[buffer(1)]],device const float*w[[buffer(2)]],device "
        "float*y[[buffer(3)]],constant Post&p[[buffer(4)]],uint "
        "row[[threadgroup_position_in_grid]],uint lid[[thread_index_in_threadgroup]],uint "
        "nt[[threads_per_threadgroup]],uint "
        "sg[[simdgroup_index_in_threadgroup]],uint sl[[thread_index_in_simdgroup]]){\n"
        "RN_HEAD(x_offset,x_row_stride)const uint yb=p.y_offset+row*p.y_row_stride,"
        "rb=p.residual_offset+row*p.residual_row_stride;"
        "if(p.rows==1u&&p.cols<=4u*nt){float xv[4],wv[4],rv[4];for(uint k=0u;k<4u;k++){const uint "
        "c=lid+k*nt;"
        "xv[k]=c<p.cols?x[xb+c]:0.0f;wv[k]=c<p.cols?w[p.w_offset+c]:0.0f;"
        "rv[k]=c<p.cols?res[rb+c]:0.0f;}float ss=0.0f;"
        "for(uint k=0u;k<4u;k++){if(lid+k*nt<p.cols)ss+=xv[k]*xv[k];}RN_SUM "
        "for(uint k=0u;k<4u;k++){const uint c=lid+k*nt;if(c<p.cols){float "
        "n=xv[k]*inv*wv[k];y[yb+c]=rv[k]+n;}}return;}"
        "float ss=0.0f;for(uint c=lid;c<p.cols;c+=nt){float v=x[xb+c];ss+=v*v;}RN_SUM "
        "for(uint c=lid;c<p.cols;c+=nt){float n=x[xb+c]*inv*w[p.w_offset+c];"
        "y[yb+c]=res[rb+c]+n;}}\n";

/* Compiled after metal_elem_simd_source, into the same library. */
static const char metal_elem_simd_add_norm_source[] =
        /* rmsnorm_add (add without w), scale and the next rmsnorm in one
         * pass (#17): the same arithmetic as those kernels in the same
         * order, at the same thread count and element-to-thread map, so the
         * same bits; y is computed and stored once and normalised from
         * registers (one row) or from the thread's own stores (several).
         * The barrier before the second reduction keeps its partials from
         * overwriting the first's while they are being read. */
        "struct Post2{uint rows,cols,residual_offset,x_offset,w_offset,zw_offset,y_offset,"
        "z_offset,residual_row_stride,x_row_stride,y_row_stride,z_row_stride,has_w;float "
        "eps,scale;};\n"
        "static inline float rn_inv(float ss,threadgroup float*partial,uint sg,uint sl,uint "
        "nt,uint cols,float eps){float sum=simd_sum(ss);if(sl==0u)partial[sg]=sum;"
        "threadgroup_barrier(mem_flags::mem_threadgroup);float total=0.0f;"
        "for(uint i=0u;i<nt/32u;i++)total+=partial[i];return rsqrt(total/float(cols)+eps);}\n"
        "kernel void rmsnorm_add_norm_rows_simd(device const float*res[[buffer(0)]],device "
        "const float*x[[buffer(1)]],device const float*w[[buffer(2)]],device const "
        "float*zw[[buffer(3)]],device float*y[[buffer(4)]],device float*z[[buffer(5)]],"
        "constant Post2&p[[buffer(6)]],uint row[[threadgroup_position_in_grid]],uint "
        "lid[[thread_index_in_threadgroup]],uint nt[[threads_per_threadgroup]],uint "
        "sg[[simdgroup_index_in_threadgroup]],uint sl[[thread_index_in_simdgroup]]){\n"
        "threadgroup float partial[32];if(row>=p.rows)return;"
        "const uint xb=p.x_offset+row*p.x_row_stride,rb=p.residual_offset+row*p.residual_"
        "row_stride,yb=p.y_offset+row*p.y_row_stride,zb=p.z_offset+row*p.z_row_stride;"
        "if(p.rows==1u&&p.cols<=4u*nt){float yv[4],zv[4];"
        "if(p.has_w!=0u){float xv[4],wv[4],rv[4];for(uint k=0u;k<4u;k++){const uint "
        "c=lid+k*nt;xv[k]=c<p.cols?x[xb+c]:0.0f;wv[k]=c<p.cols?w[p.w_offset+c]:0.0f;"
        "rv[k]=c<p.cols?res[rb+c]:0.0f;}float ss=0.0f;"
        "for(uint k=0u;k<4u;k++){if(lid+k*nt<p.cols)ss+=xv[k]*xv[k];}"
        "const float inv=rn_inv(ss,partial,sg,sl,nt,p.cols,p.eps);"
        "for(uint k=0u;k<4u;k++){float n=xv[k]*inv*wv[k];yv[k]=(rv[k]+n)*p.scale;}"
        "threadgroup_barrier(mem_flags::mem_threadgroup);}"
        "else{for(uint k=0u;k<4u;k++){const uint c=lid+k*nt;"
        "yv[k]=c<p.cols?(res[rb+c]+x[xb+c])*p.scale:0.0f;}}"
        "for(uint k=0u;k<4u;k++){const uint c=lid+k*nt;zv[k]=c<p.cols?zw[p.zw_offset+c]:0.0f;"
        "if(c<p.cols)y[yb+c]=yv[k];}float s2=0.0f;"
        "for(uint k=0u;k<4u;k++){if(lid+k*nt<p.cols)s2+=yv[k]*yv[k];}"
        "const float inv2=rn_inv(s2,partial,sg,sl,nt,p.cols,p.eps);"
        "for(uint k=0u;k<4u;k++){const uint c=lid+k*nt;if(c<p.cols)z[zb+c]=yv[k]*inv2*zv[k];}"
        "return;}"
        "float inv=1.0f;if(p.has_w!=0u){float ss=0.0f;for(uint c=lid;c<p.cols;c+=nt){float "
        "v=x[xb+c];ss+=v*v;}inv=rn_inv(ss,partial,sg,sl,nt,p.cols,p.eps);"
        "threadgroup_barrier(mem_flags::mem_threadgroup);}"
        "float s2=0.0f;for(uint c=lid;c<p.cols;c+=nt){float v;if(p.has_w!=0u){float "
        "n=x[xb+c]*inv*w[p.w_offset+c];v=res[rb+c]+n;}else{v=res[rb+c]+x[xb+c];}v=v*p.scale;"
        "y[yb+c]=v;s2+=v*v;}"
        "const float inv2=rn_inv(s2,partial,sg,sl,nt,p.cols,p.eps);"
        "for(uint c=lid;c<p.cols;c+=nt)z[zb+c]=y[yb+c]*inv2*zw[p.zw_offset+c];}\n";

static const char metal_embed_source[] =
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "struct E{uint n,dtype,bpr,wbo,yo,token;float scale;};\n"
        "static inline float h(device const uchar*w,uint o){ushort "
        "b=ushort(w[o])|(ushort(w[o+1])<<8);return float(as_type<half>(b));}\n"
        "static inline float bf(device const uchar*w,uint o){return "
        "as_type<float>((uint(w[o])|(uint(w[o+1])<<8))<<16);}\n"
        "static inline float f32(device const uchar*w,uint o){return "
        "as_type<float>(uint(w[o])|(uint(w[o+1])<<8)|(uint(w[o+2])<<16)|(uint(w[o+3])<<24));}\n"
        "static inline int i8(device const uchar*w,uint o){uint u=uint(w[o]);return "
        "u<128u?int(u):int(u)-256;}\n"
        "static inline float q40(device const uchar*w,constant E&p,uint row,uint k){uint "
        "br=k/32u,j=k&31u,bo=p.wbo+(row*p.bpr+br)*18u;uint qb=uint(w[bo+2u+(j&15u)]);int "
        "q=int(j<16u?(qb&15u):(qb>>4u))-8;return h(w,bo)*float(q);}\n"
        "static inline float q80(device const uchar*w,constant E&p,uint row,uint k){uint "
        "br=k/32u,j=k&31u,bo=p.wbo+(row*p.bpr+br)*34u;return h(w,bo)*float(i8(w,bo+2u+j));}\n"
        "static inline void sm(uint j,device const uchar*q,thread uint&s,thread "
        "uint&m){if(j<4){s=uint(q[j])&63u;m=uint(q[j+4])&63u;}else{uint "
        "a=uint(q[j+4]),b=uint(q[j-4]),c=uint(q[j]);s=(a&15u)|((b>>6u)<<4u);m=(a>>4u)|((c>>6u)<<4u)"
        ";}}\n"
        "static inline float q4(device const uchar*w,constant E&p,uint row,uint k){uint "
        "br=k/256u,ib=k-br*256u,sub=ib/32u,idx=ib-sub*32u;uint bo=p.wbo+(row*p.bpr+br)*144u;float "
        "d=h(w,bo),dm=h(w,bo+2u);uint s,m;sm(sub,w+bo+4u,s,m);uint "
        "qb=uint(w[bo+16u+(sub/2u)*32u+idx]);uint q=(sub&1u)==0u?(qb&15u):(qb>>4u);return "
        "d*float(s)*float(q)-dm*float(m);}\n"
        "static inline float q5(device const uchar*w,constant E&p,uint row,uint k){uint "
        "br=k/256u,ib=k-br*256u,g=ib/64u,ih=ib-g*64u,ha=ih/32u,l=ih-ha*32u;uint "
        "bo=p.wbo+(row*p.bpr+br)*176u;float d=h(w,bo),dm=h(w,bo+2u);uint "
        "s,m;sm(g*2u+ha,w+bo+4u,s,m);uint qb=uint(w[bo+48u+g*32u+l]);uint "
        "q=(ha==0u?(qb&15u):(qb>>4u));uint "
        "bit=1u<<(g*2u+ha);q+=(uint(w[bo+16u+l])&bit)?16u:0u;return "
        "d*float(s)*float(q)-dm*float(m);}\n"
        "static inline float q6(device const uchar*w,constant E&p,uint row,uint k){uint "
        "br=k/256u,ib=k-br*256u,hi=ib/128u,ih=ib-hi*128u,st=ih/32u,l=ih-st*32u,is=l/16u;uint "
        "bo=p.wbo+(row*p.bpr+br)*210u;uint qlb=bo+hi*64u,qhb=bo+128u+hi*32u,scb=bo+192u+hi*8u;uint "
        "ql0=uint(w[qlb+l]),ql1=uint(w[qlb+l+32u]),qh=uint(w[qhb+l]);uint "
        "qu,si;if(st==0u){qu=(ql0&15u)|(((qh>>0u)&3u)<<4u);si=is;}else "
        "if(st==1u){qu=(ql1&15u)|(((qh>>2u)&3u)<<4u);si=is+2u;}else "
        "if(st==2u){qu=(ql0>>4u)|(((qh>>4u)&3u)<<4u);si=is+4u;}else{qu=(ql1>>4u)|(((qh>>6u)&3u)<<"
        "4u);si=is+6u;}return h(w,bo+208u)*float(i8(w,scb+si))*float(int(qu)-32);}\n"
        "static inline float pq2(device const uchar*w,constant E&p,uint row,uint k){uint "
        "br=k>>7u,j=k&127u,bo=p.wbo+(row*p.bpr+br)*34u;uint "
        "c=(uint(w[bo+2u+(j>>2u)])>>(2u*(j&3u)))&3u;return h(w,bo)*float(int(c)-1);}\n"
        "kernel void embed_lookup_scaled(device const uchar*w[[buffer(0)]],device "
        "float*y[[buffer(1)]],constant E&p[[buffer(2)]],uint "
        "gid[[thread_position_in_grid]]){if(gid>=p.n)return;float "
        "v=p.dtype==0u?f32(w,p.wbo+(p.token*p.n+gid)*4u):(p.dtype==1u?h(w,p.wbo+(p.token*p.n+gid)*"
        "2u):(p.dtype==2u?bf(w,p.wbo+(p.token*p.n+gid)*2u):(p.dtype==5u?q40(w,p,p.token,gid):(p."
        "dtype==7u?q80(w,p,p.token,gid):(p.dtype==9u?q4(w,p,p.token,gid):(p.dtype==10u?q5(w,p,p."
        "token,gid):(p.dtype==22u?pq2(w,p,p.token,gid):q6(w,p,p.token,gid))))))));"
        "y[p.yo+gid]=v*p.scale;}\n";

/* Batched variant: ids arrive via a small constant buffer,
 * p.token carries the row count, one dispatch embeds the whole prefill
 * chunk. Same per-element math as the single-token kernel; separate
 * array (4095-char literal limit), concatenated at library build. */
static const char metal_embed_rows_source[] =
        "kernel void embed_lookup_scaled_rows(device const uchar*w[[buffer(0)]],device "
        "float*y[[buffer(1)]],constant E&p[[buffer(2)]],constant uint*ids[[buffer(3)]],uint2 "
        "gid[[thread_position_in_grid]]){if(gid.x>=p.n||gid.y>=p.token)return;uint "
        "row=ids[gid.y];float "
        "v=p.dtype==0u?f32(w,p.wbo+(row*p.n+gid.x)*4u):(p.dtype==1u?h(w,p.wbo+(row*p.n+gid.x)*"
        "2u):(p.dtype==2u?bf(w,p.wbo+(row*p.n+gid.x)*2u):(p.dtype==5u?q40(w,p,row,gid.x):(p."
        "dtype==7u?q80(w,p,row,gid.x):(p.dtype==9u?q4(w,p,row,gid.x):(p.dtype==10u?q5(w,p,row,"
        "gid.x):(p.dtype==22u?pq2(w,p,row,gid.x):q6(w,p,row,gid.x))))))));"
        "y[p.yo+gid.y*p.n+gid.x]=v*p.scale;}\n";

/* matmul_<NAME> (one row per threadgroup) and matmul_<NAME>_sg (8x8
 * simdgroup tiles), instantiated for F32 weights and for F16 / BF16 ones
 * converted to float as they are loaded (#564). F16 and BF16 are exact in
 * float, so the result is bit-identical to running the F32 kernels on the
 * matrix widened to F32, without the doubled bytes. NAME is the suffix, WT the
 * stored element type, CV the conversion of `v`.
 *
 * matmul_<NAME>_sg stages 64 k (eight 8x8 tiles of x and of w) per barrier
 * pair, with float4 / WT4 loads where the tile is aligned and whole in k,
 * and then runs the same eight simdgroup_multiply_accumulate steps in the
 * same k order as one 8-k step per barrier pair did: bit-identical, but
 * the per-tile chain of dependent global loads is 8x shorter. That chain
 * is the kernel's time on the narrow prefill shapes (qwen35 alpha/beta:
 * 5120 -> 48 at 320 rows, 240 tiles), 0.57 -> 0.21 ms on an M1 Max (#77).
 * Tiles are stored 72 floats apart so the staging stores do not collide in
 * threadgroup memory; w is staged transposed and loaded with transpose. */
#define GEIST_METAL_HALF_W_KERNELS(NAME, WT, CV)                                               \
    "kernel void matmul_" NAME "(device const float*x[[buffer(0)]],device const " WT           \
    "*w[[buffer(1)]],device float*y[[buffer(2)]],constant P&p[[buffer(3)]],uint3 "             \
    "tg[[threadgroup_position_in_grid]],uint lid[[thread_index_in_threadgroup]],uint3 "        \
    "nt3[[threads_per_threadgroup]],uint sg[[simdgroup_index_in_threadgroup]],uint "           \
    "sl[[thread_index_in_simdgroup]]){threadgroup "                                            \
    "float part[32];uint nt=nt3.x;uint row=tg.x,b=tg.y;if(row>=p.no||b>=p.rows)return;"        \
    "if(lid<32u)part[lid]=0.0f;threadgroup_barrier(mem_flags::mem_threadgroup);"               \
    "float s=0.0f;for(uint "                                                                   \
    "k=lid;k<p.ni;k+=nt){" WT " v=w[p.wo+row*p.ni+k];s+=x[p.xo+b*p.xs+k]*(" CV ");}"           \
    "float r=simd_sum(s);if(sl==0u)part[sg]=r;threadgroup_"                                    \
    "barrier(mem_flags::mem_threadgroup);"                                                     \
    "if(lid==0u){float t=0.0f;for(uint i=0u;i<nt/32u;i++)t+=part[i];y[p.yo+b*p.ys+row]=t;}}\n" \
    "kernel void matmul_" NAME "_sg(device const float*x[[buffer(0)]],device const " WT        \
    "*w[[buffer(1)]],device float*y[[buffer(2)]],constant P&p[[buffer(3)]],uint3 "             \
    "tg[[threadgroup_position_in_grid]],uint lid[[thread_index_in_threadgroup]]){"             \
    "threadgroup float as[576];threadgroup float bs[576];threadgroup float cs[64];"            \
    "uint b0=tg.y*8u,o0=tg.x*8u;"                                                              \
    "simdgroup_float8x8 acc=make_filled_simdgroup_matrix<float,8>(0.0f);"                      \
    "bool va=((p.xo|p.xs|p.wo|p.ni)&3u)==0u;"                                                  \
    "for(uint kb=0u;kb<p.ni;kb+=64u){"                                                         \
    "if(va&&kb+64u<=p.ni&&b0+8u<=p.rows&&o0+8u<=p.no){"                                        \
    "for(uint q=0u;q<4u;q++){uint i=q*32u+lid,r=i/16u,k=(i%16u)*4u;"                           \
    "float4 xv=*(device const float4*)(x+p.xo+(b0+r)*p.xs+kb+k);"                              \
    "*(threadgroup float4*)(as+(k/8u)*72u+r*8u+k%8u)=xv;"                                      \
    "" WT "4 v4=*(device const " WT "4*)(w+p.wo+(o0+r)*p.ni+kb+k);"                            \
    "threadgroup float*bp=bs+(k/8u)*72u+r*8u+k%8u;"                                            \
    "for(uint j=0u;j<4u;j++){" WT " v=v4[j];bp[j]=" CV ";}}"                                   \
    "}else if(va&&kb+64u<=p.ni){"                                                              \
    "for(uint q=0u;q<4u;q++){uint i=q*32u+lid,r=i/16u,k=(i%16u)*4u;"                           \
    "float4 xv=b0+r<p.rows?*(device const float4*)(x+p.xo+(b0+r)*p.xs+kb+k):float4(0.0f);"     \
    "*(threadgroup float4*)(as+(k/8u)*72u+r*8u+k%8u)=xv;"                                      \
    "" WT "4 v4=o0+r<p.no?*(device const " WT "4*)(w+p.wo+(o0+r)*p.ni+kb+k):" WT "4(0);"       \
    "threadgroup float*bp=bs+(k/8u)*72u+r*8u+k%8u;"                                            \
    "for(uint j=0u;j<4u;j++){" WT " v=v4[j];bp[j]=" CV ";}}"                                   \
    "}else{"                                                                                   \
    "for(uint i=lid;i<512u;i+=32u){uint r=i/64u,k=i%64u;uint d=(k/8u)*72u+r*8u+k%8u;"          \
    "as[d]=(b0+r<p.rows&&kb+k<p.ni)?x[p.xo+(b0+r)*p.xs+kb+k]:0.0f;"                            \
    "float wv=0.0f;if(o0+r<p.no&&kb+k<p.ni){" WT " v=w[p.wo+(o0+r)*p.ni+kb+k];wv=" CV          \
    ";}bs[d]=wv;}"                                                                             \
    "}"                                                                                        \
    "threadgroup_barrier(mem_flags::mem_threadgroup);"                                         \
    "uint ns=(min(64u,p.ni-kb)+7u)/8u;"                                                        \
    "for(uint t=0u;t<ns;t++){simdgroup_float8x8 "                                              \
    "ma,mb;simdgroup_load(ma,as+t*72u,8);simdgroup_load(mb,bs+t*72u,8,ulong2(0,0),true);"      \
    "simdgroup_multiply_accumulate(acc,ma,mb,acc);}"                                           \
    "threadgroup_barrier(mem_flags::mem_threadgroup);}"                                        \
    "simdgroup_store(acc,cs,8);threadgroup_barrier(mem_flags::mem_threadgroup);"               \
    "for(uint i=lid;i<64u;i+=32u){uint "                                                       \
    "r=i/8u,c=i%8u;if(b0+r<p.rows&&o0+c<p.no)y[p.yo+(b0+r)*p.ys+o0+c]=cs[i];}}\n"

static const char metal_f32_source[] =
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "#define FOR_UNROLL(x) _Pragma(\"clang loop unroll(full)\") for(x)\n"
        "struct P{uint ni,no,rows,xo,wo,yo,xs,ys;};\n"
        "struct GP{uint ni,no,rows,xo,wo,po,yo,xs,ps,ys;};\n"
        "struct PN{uint ni,no,rows,xo,wo,ro,nwo,yo,xs,rs,ys;float eps;};\n"
        "static inline float gelu(float x){return "
        "0.5f*x*(1.0f+tanh(clamp(0.7978845608028654f*(x+0.044715f*x*x*x),-10.0f,10.0f)));}"
        "\n" GEIST_METAL_HALF_W_KERNELS("f32", "float", "v");

/* The PLE kernels of the F32 library: a separate literal, next to the one
 * above, to stay within the 4095-char literal limit. */
static const char metal_f32_ple_source[] =
        "kernel void ple_gate_f32(device const float*x[[buffer(0)]],device const "
        "float*w[[buffer(1)]],device const float*ple[[buffer(2)]],device "
        "float*y[[buffer(3)]],constant GP&p[[buffer(4)]],uint3 "
        "tg[[threadgroup_position_in_grid]],uint "
        "lid[[thread_index_in_threadgroup]]){threadgroup "
        "float part[256];uint row=tg.x,b=tg.y;if(row>=p.no||b>=p.rows)return;float "
        "s=0.0f;for(uint "
        "k=lid;k<p.ni;k+=256u){s+=x[p.xo+b*p.xs+k]*w[p.wo+row*p.ni+k];}part[lid]=s;"
        "threadgroup_"
        "barrier(mem_flags::mem_threadgroup);for(uint "
        "st=128u;st>0u;st>>=1u){if(lid<st){part[lid]+=part[lid+st];}threadgroup_"
        "barrier(mem_flags::"
        "mem_threadgroup);}if(lid==0u){uint "
        "po=p.po+b*p.ps+row;y[p.yo+b*p.ys+row]=gelu(part[0])*ple[po];}}\n"
        "kernel void ple_proj_norm_f32(device const float*x[[buffer(0)]],device const "
        "float*w[[buffer(1)]],device const float*res[[buffer(2)]],device const "
        "float*nw[[buffer(3)]],device float*y[[buffer(4)]],constant "
        "PN&p[[buffer(5)]],uint "
        "b[[threadgroup_position_in_grid]],uint "
        "lid[[thread_index_in_threadgroup]]){threadgroup "
        "float part[256];if(b>=p.rows)return;float ss=0.0f;for(uint "
        "c=lid;c<p.no;c+=256u){float "
        "s=0.0f;for(uint "
        "k=0u;k<p.ni;k++){s+=x[p.xo+b*p.xs+k]*w[p.wo+c*p.ni+k];}y[p.yo+b*p.ys+c]=s;ss+"
        "=s*s;}part["
        "lid]=ss;threadgroup_barrier(mem_flags::mem_threadgroup);for(uint "
        "st=128u;st>0u;st>>=1u){if(lid<st){part[lid]+=part[lid+st];}threadgroup_"
        "barrier(mem_flags::"
        "mem_threadgroup);}float inv=rsqrt(part[0]/float(p.no)+p.eps);for(uint "
        "c=lid;c<p.no;c+=256u){float "
        "v=y[p.yo+b*p.ys+c]*inv*nw[p.nwo+c];y[p.yo+b*p.ys+c]=res[p.ro+b*p.rs+c]+v;}}"
        "\n";

static const char metal_f16_w_source[] = GEIST_METAL_HALF_W_KERNELS("f16w", "half", "float(v)");
static const char metal_bf16_w_source[] =
        GEIST_METAL_HALF_W_KERNELS("bf16w", "ushort", "as_type<float>(uint(v)<<16u)");

/* llama-mm_sg-structured f32 GEMM (64-out x 32-row tile, 4 simdgroups,
 * f32 threadgroup staging so the per-element summation order matches
 * matmul_f32_sg = bit-identical results). Interior fast path only:
 * dispatched when rows%32==0, n_out%64==0, n_in%32==0 and x/w aligned.
 * Own array for the C99 4095-char literal limit; concatenated with
 * metal_f32_source at library init (FOR_UNROLL/struct P live there). */
static const char metal_f32_mm_source[] =
        "kernel void matmul_f32_mm_sg(device const float*x[[buffer(0)]],device const "
        "float*w[[buffer(1)]],device float*y[[buffer(2)]],constant P&p[[buffer(3)]],threadgroup "
        "char*shmem[[threadgroup(0)]],uint3 tg[[threadgroup_position_in_grid]],ushort "
        "ti[[thread_index_in_threadgroup]],ushort sg[[simdgroup_index_in_threadgroup]]){\n"
        "threadgroup float*sa=(threadgroup float*)shmem;threadgroup float*sb=(threadgroup "
        "float*)(shmem+8192);"
        "constexpr short NK=32;constexpr short NL0=2;constexpr short NL1=4;"
        "uint r0=tg.y*64u;uint r1=tg.x*32u;"
        "short lr0=short(ti/NL0);short lr1=short(ti/NL1);short il0=short(ti%NL0);"
        "device const float*wpf=w+p.wo+(r0+uint(lr0))*p.ni+uint(16*il0);"
        "short iy=short(8*(ti%NL1));"
        "device const float*xp=x+p.xo+min(r1+uint(lr1),p.rows-1u)*p.xs+uint(iy);"
        "simdgroup_float8x8 ma[4];simdgroup_float8x8 mb[2];simdgroup_float8x8 mc[8];"
        "FOR_UNROLL(short i=0;i<8;i++){mc[i]=make_filled_simdgroup_matrix<float,8>(0.0f);}"
        "short ay=short((ti/NL0)/8);short lx=short((ti/NL0)%8);"
        "short sx=short(ti%NL1);short sy=short((ti/NL1)/8);short lyb=short((ti/NL1)%8);short "
        "ibb=short(4*sx+sy);"
        "for(uint loop_k=0u;loop_k<p.ni;loop_k+=uint(NK)){"
        "float4 f0=*(device const float4*)(wpf);float4 f1=*(device const float4*)(wpf+4);float4 "
        "f2=*(device const float4*)(wpf+8);float4 f3=*(device const float4*)(wpf+12);"
        "threadgroup_barrier(mem_flags::mem_threadgroup);"
        "FOR_UNROLL(short i=0;i<16;i++){short ax=short(2*il0+i/8);short ly=short(i%8);float "
        "v=i<4?f0[i]:(i<8?f1[i-4]:(i<12?f2[i-8]:f3[i-12]));*(sa+64*(8*ax+ay)+8*ly+lx)=v;}"
        "*(threadgroup float2x4*)(sb+64*ibb+8*lyb)=*((device const float2x4*)xp);"
        "wpf+=NK;xp+=NK;"
        "threadgroup_barrier(mem_flags::mem_threadgroup);"
        "threadgroup const float*lsma=sa+4*64*(sg%2);threadgroup const float*lsmb=sb+2*64*(sg/2);"
        "FOR_UNROLL(short ik=0;ik<NK/8;ik++){"
        "simdgroup_barrier(mem_flags::mem_none);"
        "FOR_UNROLL(short i=0;i<4;i++){simdgroup_load(ma[i],lsma+64*i,8,0,false);}"
        "simdgroup_barrier(mem_flags::mem_none);"
        "FOR_UNROLL(short i=0;i<2;i++){simdgroup_load(mb[i],lsmb+64*i,8,0,false);}"
        "simdgroup_barrier(mem_flags::mem_none);"
        "FOR_UNROLL(short i=0;i<8;i++){simdgroup_multiply_accumulate(mc[i],mb[i/4],ma[i%4],mc[i]);}"
        "lsma+=8*64;lsmb+=4*64;}}"
        "if(r1+32u<=p.rows){device float*C=y+p.yo+(r1+16u*uint(sg>>1))*p.ys+r0+32u*uint(sg&1);"
        "FOR_UNROLL(short "
        "i=0;i<8;i++){simdgroup_store(mc[i],C+8*(i%4)+8*p.ys*(i/4),p.ys,0,false);}}"
        "else{threadgroup_barrier(mem_flags::mem_threadgroup);threadgroup float*T=(threadgroup "
        "float*)shmem;"
        "FOR_UNROLL(short "
        "i=0;i<8;i++){simdgroup_store(mc[i],T+32*(sg&1)+1024*(sg>>1)+8*(i%4)+512*(i/"
        "4),64,0,false);}"
        "threadgroup_barrier(mem_flags::mem_threadgroup);for(uint "
        "j=uint(ti);j<(p.rows-r1)*64u;j+=128u)"
        "{y[p.yo+(r1+j/64u)*p.ys+r0+j%64u]=T[j];}}"
        "}\n";

static const char metal_q_norm_rope_source[] =
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "struct NR{uint "
        "rows,heads,head_dim,x_offset,w_offset,cos_offset,sin_offset,x_row_stride,rope_row_stride,"
        "rope_row_offset;float eps;};\n"
        "kernel void q_norm_rope_rows(device float*x[[buffer(0)]],device const "
        "float*w[[buffer(1)]],device const float*c[[buffer(2)]],device const "
        "float*s[[buffer(3)]],constant NR&p[[buffer(4)]],uint2 "
        "tg[[threadgroup_position_in_grid]],uint lid[[thread_index_in_threadgroup]]){\n"
        " threadgroup float part[256];uint r=tg.x,h=tg.y;if(r>=p.rows||h>=p.heads){return;}uint "
        "base=p.x_offset+r*p.x_row_stride+h*p.head_dim;float ss=0.0f;for(uint "
        "i=lid;i<p.head_dim;i+=256u){float "
        "v=x[base+i];ss+=v*v;}part[lid]=ss;threadgroup_barrier(mem_flags::mem_threadgroup);for("
        "uint "
        "st=128u;st>0u;st>>=1u){if(lid<st){part[lid]+=part[lid+st];}threadgroup_barrier(mem_flags::"
        "mem_threadgroup);}float inv=rsqrt(part[0]/float(p.head_dim)+p.eps);"
        /* Norm covers the whole head; rotation covers the rotated block
           only, so the two are separate passes with a barrier between —
           a thread's rotation partner is another thread's normed write. */
        "for(uint i=lid;i<p.head_dim;i+=256u){x[base+i]=x[base+i]*inv*w[p.w_offset+i];}"
        "threadgroup_barrier(mem_flags::mem_device);uint "
        "hd2=p.rope_row_stride/2u,ro=p.rope_row_offset+r;for(uint i=lid;i<hd2;i+=256u){float "
        "x0=x[base+i],x1=x[base+i+hd2];float "
        "co=c[p.cos_offset+ro*p.rope_row_stride+i],si=s[p.sin_offset+ro*p.rope_row_stride+i];x["
        "base+i]=x0*co-x1*si;x[base+i+hd2]=x0*si+x1*co;}\n"
        "}\n";

static const char metal_k_norm_rope_append_source[] =
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "struct KRA{uint "
        "rows,heads,head_dim,x_offset,w_offset,cos_offset,sin_offset,cache_offset,x_row_stride,"
        "rope_row_stride,rope_row_offset,q_position;float eps;};\n"
        "kernel void k_norm_rope_append_rows(device float*x[[buffer(0)]],device const "
        "float*w[[buffer(1)]],device const float*c[[buffer(2)]],device const "
        "float*s[[buffer(3)]],device float*cache[[buffer(4)]],constant KRA&p[[buffer(5)]],uint2 "
        "tg[[threadgroup_position_in_grid]],uint lid[[thread_index_in_threadgroup]]){\n"
        " threadgroup float part[256];uint r=tg.x,h=tg.y;if(r>=p.rows||h>=p.heads){return;}uint "
        "base=p.x_offset+r*p.x_row_stride+h*p.head_dim;float ss=0.0f;for(uint "
        "i=lid;i<p.head_dim;i+=256u){float "
        "v=x[base+i];ss+=v*v;}part[lid]=ss;threadgroup_barrier(mem_flags::mem_threadgroup);for("
        "uint "
        "st=128u;st>0u;st>>=1u){if(lid<st){part[lid]+=part[lid+st];}threadgroup_barrier(mem_flags::"
        "mem_threadgroup);}float inv=rsqrt(part[0]/float(p.head_dim)+p.eps);uint "
        "hd2=p.head_dim/"
        "2u,ro=p.rope_row_offset+r,cb=p.cache_offset+(p.q_position+r)*p.heads*p.head_dim+h*p.head_"
        "dim;for(uint i=lid;i<hd2;i+=256u){float "
        "x0=x[base+i]*inv*w[p.w_offset+i],x1=x[base+i+hd2]*inv*w[p.w_offset+i+hd2];float "
        "co=c[p.cos_offset+ro*p.rope_row_stride+i],si=s[p.sin_offset+ro*p.rope_row_stride+i];float "
        "y0=x0*co-x1*si,y1=x0*si+x1*co;x[base+i]=y0;x[base+i+hd2]=y1;cache[cb+i]=y0;cache[cb+i+hd2]"
        "=y1;}\n"
        "}\n"
        "kernel void k_norm_rope_append_rows_f16(device float*x[[buffer(0)]],device const "
        "float*w[[buffer(1)]],device const float*c[[buffer(2)]],device const "
        "float*s[[buffer(3)]],device half*cache[[buffer(4)]],constant KRA&p[[buffer(5)]],uint2 "
        "tg[[threadgroup_position_in_grid]],uint lid[[thread_index_in_threadgroup]]){\n"
        " threadgroup float part[256];uint r=tg.x,h=tg.y;if(r>=p.rows||h>=p.heads){return;}uint "
        "base=p.x_offset+r*p.x_row_stride+h*p.head_dim;float ss=0.0f;for(uint "
        "i=lid;i<p.head_dim;i+=256u){float "
        "v=x[base+i];ss+=v*v;}part[lid]=ss;threadgroup_barrier(mem_flags::mem_threadgroup);for("
        "uint "
        "st=128u;st>0u;st>>=1u){if(lid<st){part[lid]+=part[lid+st];}threadgroup_barrier(mem_flags::"
        "mem_threadgroup);}float inv=rsqrt(part[0]/float(p.head_dim)+p.eps);uint "
        "hd2=p.head_dim/"
        "2u,ro=p.rope_row_offset+r,cb=p.cache_offset+(p.q_position+r)*p.heads*p.head_dim+h*p.head_"
        "dim;for(uint i=lid;i<hd2;i+=256u){float "
        "x0=x[base+i]*inv*w[p.w_offset+i],x1=x[base+i+hd2]*inv*w[p.w_offset+i+hd2];float "
        "co=c[p.cos_offset+ro*p.rope_row_stride+i],si=s[p.sin_offset+ro*p.rope_row_stride+i];float "
        "y0=x0*co-x1*si,y1=x0*si+x1*co;x[base+i]=y0;x[base+i+hd2]=y1;cache[cb+i]=half(y0);cache[cb+"
        "i+hd2]=half(y1);}\n"
        "}\n";

static const char metal_v_norm_append_source[] =
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "struct VA{uint "
        "rows,heads,head_dim,x_offset,w_offset,cache_offset,x_row_stride,q_position;float eps;};\n"
        "kernel void v_norm_append_rows(device float*x[[buffer(0)]],device const "
        "float*w[[buffer(1)]],device float*cache[[buffer(2)]],constant VA&p[[buffer(3)]],uint2 "
        "tg[[threadgroup_position_in_grid]],uint lid[[thread_index_in_threadgroup]]){\n"
        " threadgroup float part[256];uint r=tg.x,h=tg.y;if(r>=p.rows||h>=p.heads){return;}uint "
        "base=p.x_offset+r*p.x_row_stride+h*p.head_dim;float ss=0.0f;for(uint "
        "i=lid;i<p.head_dim;i+=256u){float "
        "v=x[base+i];ss+=v*v;}part[lid]=ss;threadgroup_barrier(mem_flags::mem_threadgroup);for("
        "uint "
        "st=128u;st>0u;st>>=1u){if(lid<st){part[lid]+=part[lid+st];}threadgroup_barrier(mem_flags::"
        "mem_threadgroup);}float inv=rsqrt(part[0]/float(p.head_dim)+p.eps);uint "
        "cb=p.cache_offset+(p.q_position+r)*p.heads*p.head_dim+h*p.head_dim;for(uint "
        "i=lid;i<p.head_dim;i+=256u){float "
        "y=x[base+i]*inv*w[p.w_offset+i];x[base+i]=y;cache[cb+i]=y;}\n"
        "}\n"
        "kernel void v_norm_append_rows_f16(device float*x[[buffer(0)]],device const "
        "float*w[[buffer(1)]],device half*cache[[buffer(2)]],constant VA&p[[buffer(3)]],uint2 "
        "tg[[threadgroup_position_in_grid]],uint lid[[thread_index_in_threadgroup]]){\n"
        " threadgroup float part[256];uint r=tg.x,h=tg.y;if(r>=p.rows||h>=p.heads){return;}uint "
        "base=p.x_offset+r*p.x_row_stride+h*p.head_dim;float ss=0.0f;for(uint "
        "i=lid;i<p.head_dim;i+=256u){float "
        "v=x[base+i];ss+=v*v;}part[lid]=ss;threadgroup_barrier(mem_flags::mem_threadgroup);for("
        "uint "
        "st=128u;st>0u;st>>=1u){if(lid<st){part[lid]+=part[lid+st];}threadgroup_barrier(mem_flags::"
        "mem_threadgroup);}float inv=rsqrt(part[0]/float(p.head_dim)+p.eps);uint "
        "cb=p.cache_offset+(p.q_position+r)*p.heads*p.head_dim+h*p.head_dim;for(uint "
        "i=lid;i<p.head_dim;i+=256u){float "
        "y=x[base+i]*inv*w[p.w_offset+i];x[base+i]=y;cache[cb+i]=half(y);}\n"
        "}\n";

static const char metal_kv_norm_append_source[] =
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "struct K{uint rows,heads,hd,xo,wo,co,so,cao,xs,rs,ro,qp;float eps;};\n"
        "struct V{uint rows,heads,hd,xo,wo,cao,xs,qp;float eps;};\n"
        "kernel void kv_norm_append_rows(device float*k[[buffer(0)]],device "
        "float*v[[buffer(1)]],device const float*kw[[buffer(2)]],device const "
        "float*vw[[buffer(3)]],device const float*c[[buffer(4)]],device const "
        "float*s[[buffer(5)]],device float*kc[[buffer(6)]],device float*vc[[buffer(7)]],constant "
        "K&kp[[buffer(8)]],constant V&vp[[buffer(9)]],uint2 "
        "tg[[threadgroup_position_in_grid]],uint lid[[thread_index_in_threadgroup]]){\n"
        " threadgroup float kr[256];threadgroup float vr[256];uint "
        "r=tg.x,h=tg.y;if(r>=kp.rows||h>=kp.heads){return;}uint "
        "kb=kp.xo+r*kp.xs+h*kp.hd,vb=vp.xo+r*vp.xs+h*vp.hd;float ks=0.0f,vs=0.0f;for(uint "
        "i=lid;i<kp.hd;i+=256u){float x=k[kb+i];ks+=x*x;}for(uint i=lid;i<vp.hd;i+=256u){float "
        "x=v[vb+i];vs+=x*x;}kr[lid]=ks;vr[lid]=vs;threadgroup_barrier(mem_flags::mem_threadgroup);"
        "for(uint "
        "st=128u;st>0u;st>>=1u){if(lid<st){kr[lid]+=kr[lid+st];vr[lid]+=vr[lid+st];}threadgroup_"
        "barrier(mem_flags::mem_threadgroup);}float "
        "kinv=rsqrt(kr[0]/float(kp.hd)+kp.eps),vinv=rsqrt(vr[0]/float(vp.hd)+vp.eps);uint "
        "hd2=kp.hd/"
        "2u,ro=kp.ro+r,kdst=kp.cao+(kp.qp+r)*kp.heads*kp.hd+h*kp.hd,vdst=vp.cao+(vp.qp+r)*vp.heads*"
        "vp.hd+h*vp.hd;for(uint i=lid;i<hd2;i+=256u){float "
        "x0=k[kb+i]*kinv*kw[kp.wo+i],x1=k[kb+i+hd2]*kinv*kw[kp.wo+i+hd2];float "
        "co=c[kp.co+ro*kp.rs+i],si=s[kp.so+ro*kp.rs+i];float "
        "y0=x0*co-x1*si,y1=x0*si+x1*co;k[kb+i]=y0;k[kb+i+hd2]=y1;kc[kdst+i]=y0;kc[kdst+i+hd2]=y1;}"
        "for(uint i=lid;i<vp.hd;i+=256u){float "
        "y=v[vb+i]*vinv*vw[vp.wo+i];v[vb+i]=y;vc[vdst+i]=y;}\n"
        "}\n";

static const char metal_kv_norm_append_f16_source[] =
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "struct K{uint rows,heads,hd,xo,wo,co,so,cao,xs,rs,ro,qp;float eps;};\n"
        "struct V{uint rows,heads,hd,xo,wo,cao,xs,qp;float eps;};\n"
        "kernel void kv_norm_append_rows_f16(device float*k[[buffer(0)]],device "
        "float*v[[buffer(1)]],device const float*kw[[buffer(2)]],device const "
        "float*vw[[buffer(3)]],device const float*c[[buffer(4)]],device const "
        "float*s[[buffer(5)]],device half*kc[[buffer(6)]],device half*vc[[buffer(7)]],constant "
        "K&kp[[buffer(8)]],constant V&vp[[buffer(9)]],uint2 "
        "tg[[threadgroup_position_in_grid]],uint lid[[thread_index_in_threadgroup]]){\n"
        " threadgroup float kr[256];threadgroup float vr[256];uint "
        "r=tg.x,h=tg.y;if(r>=kp.rows||h>=kp.heads){return;}uint "
        "kb=kp.xo+r*kp.xs+h*kp.hd,vb=vp.xo+r*vp.xs+h*vp.hd;float ks=0.0f,vs=0.0f;for(uint "
        "i=lid;i<kp.hd;i+=256u){float x=k[kb+i];ks+=x*x;}for(uint i=lid;i<vp.hd;i+=256u){float "
        "x=v[vb+i];vs+=x*x;}kr[lid]=ks;vr[lid]=vs;threadgroup_barrier(mem_flags::mem_threadgroup);"
        "for(uint "
        "st=128u;st>0u;st>>=1u){if(lid<st){kr[lid]+=kr[lid+st];vr[lid]+=vr[lid+st];}threadgroup_"
        "barrier(mem_flags::mem_threadgroup);}float "
        "kinv=rsqrt(kr[0]/float(kp.hd)+kp.eps),vinv=rsqrt(vr[0]/float(vp.hd)+vp.eps);uint "
        "hd2=kp.hd/"
        "2u,ro=kp.ro+r,kdst=kp.cao+(kp.qp+r)*kp.heads*kp.hd+h*kp.hd,vdst=vp.cao+(vp.qp+r)*vp.heads*"
        "vp.hd+h*vp.hd;for(uint i=lid;i<hd2;i+=256u){float "
        "x0=k[kb+i]*kinv*kw[kp.wo+i],x1=k[kb+i+hd2]*kinv*kw[kp.wo+i+hd2];float "
        "co=c[kp.co+ro*kp.rs+i],si=s[kp.so+ro*kp.rs+i];float "
        "y0=x0*co-x1*si,y1=x0*si+x1*co;k[kb+i]=y0;k[kb+i+hd2]=y1;kc[kdst+i]=half(y0);kc[kdst+i+hd2]"
        "=half(y1);}for(uint i=lid;i<vp.hd;i+=256u){float "
        "y=v[vb+i]*vinv*vw[vp.wo+i];v[vb+i]=y;vc[vdst+i]=half(y);}\n"
        "}\n";

/* Gemma attention prep in one dispatch (#196): q_norm_rope_rows over the
 * q heads, then kv_norm_append_rows over the k/v heads, one threadgroup
 * per (row, head) as the two kernels have, each branch their code
 * unchanged, so the same bits. Appended to the F32- and F16-cache K/V
 * libraries after a CT define (the cache element type). */
static const char metal_qkv_norm_append_source[] =
        "struct NR{uint "
        "rows,heads,head_dim,x_offset,w_offset,cos_offset,sin_offset,x_row_stride,rope_row_stride,"
        "rope_row_offset;float eps;};\n"
        "kernel void qkv_norm_append_rows(device float*x[[buffer(0)]],device const "
        "float*w[[buffer(1)]],device const float*c[[buffer(2)]],device const "
        "float*s[[buffer(3)]],constant NR&p[[buffer(4)]],device float*k[[buffer(5)]],device "
        "float*v[[buffer(6)]],device const float*kw[[buffer(7)]],device const "
        "float*vw[[buffer(8)]],device CT*kc[[buffer(9)]],device CT*vc[[buffer(10)]],constant "
        "K&kp[[buffer(11)]],constant V&vp[[buffer(12)]],uint2 "
        "tg[[threadgroup_position_in_grid]],uint lid[[thread_index_in_threadgroup]]){\n"
        " threadgroup float part[256];threadgroup float kr[256];threadgroup float vr[256];"
        "if(tg.y<p.heads){uint r=tg.x,h=tg.y;if(r>=p.rows){return;}uint "
        "base=p.x_offset+r*p.x_row_stride+h*p.head_dim;float ss=0.0f;for(uint "
        "i=lid;i<p.head_dim;i+=256u){float "
        "v=x[base+i];ss+=v*v;}part[lid]=ss;threadgroup_barrier(mem_flags::mem_threadgroup);for("
        "uint "
        "st=128u;st>0u;st>>=1u){if(lid<st){part[lid]+=part[lid+st];}threadgroup_barrier(mem_flags::"
        "mem_threadgroup);}float inv=rsqrt(part[0]/float(p.head_dim)+p.eps);"
        "for(uint i=lid;i<p.head_dim;i+=256u){x[base+i]=x[base+i]*inv*w[p.w_offset+i];}"
        "threadgroup_barrier(mem_flags::mem_device);uint "
        "hd2=p.rope_row_stride/2u,ro=p.rope_row_offset+r;for(uint i=lid;i<hd2;i+=256u){float "
        "x0=x[base+i],x1=x[base+i+hd2];float "
        "co=c[p.cos_offset+ro*p.rope_row_stride+i],si=s[p.sin_offset+ro*p.rope_row_stride+i];x["
        "base+i]=x0*co-x1*si;x[base+i+hd2]=x0*si+x1*co;}return;}\n"
        "uint r=tg.x,h=tg.y-p.heads;if(r>=kp.rows||h>=kp.heads){return;}uint "
        "kb=kp.xo+r*kp.xs+h*kp.hd,vb=vp.xo+r*vp.xs+h*vp.hd;float ks=0.0f,vs=0.0f;for(uint "
        "i=lid;i<kp.hd;i+=256u){float x=k[kb+i];ks+=x*x;}for(uint i=lid;i<vp.hd;i+=256u){float "
        "x=v[vb+i];vs+=x*x;}kr[lid]=ks;vr[lid]=vs;threadgroup_barrier(mem_flags::mem_threadgroup);"
        "for(uint "
        "st=128u;st>0u;st>>=1u){if(lid<st){kr[lid]+=kr[lid+st];vr[lid]+=vr[lid+st];}threadgroup_"
        "barrier(mem_flags::mem_threadgroup);}float "
        "kinv=rsqrt(kr[0]/float(kp.hd)+kp.eps),vinv=rsqrt(vr[0]/float(vp.hd)+vp.eps);uint "
        "hd2=kp.hd/"
        "2u,ro=kp.ro+r,kdst=kp.cao+(kp.qp+r)*kp.heads*kp.hd+h*kp.hd,vdst=vp.cao+(vp.qp+r)*vp.heads*"
        "vp.hd+h*vp.hd;for(uint i=lid;i<hd2;i+=256u){float "
        "x0=k[kb+i]*kinv*kw[kp.wo+i],x1=k[kb+i+hd2]*kinv*kw[kp.wo+i+hd2];float "
        "co=c[kp.co+ro*kp.rs+i],si=s[kp.so+ro*kp.rs+i];float "
        "y0=x0*co-x1*si,y1=x0*si+x1*co;k[kb+i]=y0;k[kb+i+hd2]=y1;kc[kdst+i]=CT(y0);kc[kdst+i+hd2]"
        "=CT(y1);}"
        "for(uint i=lid;i<vp.hd;i+=256u){float "
        "y=v[vb+i]*vinv*vw[vp.wo+i];v[vb+i]=y;vc[vdst+i]=CT(y);}\n"
        "}\n";
static const char metal_cache_f32_define[] = "#define CT float\n";
static const char metal_cache_f16_define[] = "#define CT half\n";

/* Llama / Qwen3 attention prep with the KV append (#17), the twin of
 * q_norm_rope + kv_norm_append: one threadgroup per (row, head) over the
 * q heads, then the k heads. A head is optionally RMS-normed (Qwen3's
 * q/k norms), RoPE-rotated over its first nrot elements (half split), q
 * scaled when qsc is set, written back in place; a k head and its v head
 * are also stored at row qp of the caches. The norm is rmsnorm_rows_simd's
 * multi-row loop at 256 threads, the rotation rope_rows', the scale
 * scale_rows' and the stores kv_append_rows', each step reading the
 * previous one's result back from memory as the separate kernels do, so the
 * output is theirs bit for bit. */
#define GEIST_METAL_ROPE_APPEND(NAME, T)                                                         \
    "kernel void " NAME "(device float*q[[buffer(0)]],device float*k[[buffer(1)]],device const " \
    "float*v[[buffer(2)]],device const float*qw[[buffer(3)]],device const "                      \
    "float*kw[[buffer(4)]],device const float*c[[buffer(5)]],device const "                      \
    "float*s[[buffer(6)]],device " T "*kc[[buffer(7)]],device " T "*vc[[buffer(8)]],constant "   \
    "RA&p[[buffer(9)]],uint2 tg[[threadgroup_position_in_grid]],uint "                           \
    "lid[[thread_index_in_threadgroup]],uint2 ntg[[threads_per_threadgroup]],uint "              \
    "sg[[simdgroup_index_in_threadgroup]],uint sl[[thread_index_in_simdgroup]]){"                \
    "threadgroup float partial[32];threadgroup float hv[512];uint nt=ntg.x;"                     \
    "uint r=tg.x,h=tg.y;if(r>=p.rows||h>=p.qh+p.kvh)return;"                                     \
    "bool isq=h<p.qh;uint hh=isq?h:h-p.qh;device float*x=isq?q:k;"                               \
    "uint base=isq?p.qo+r*p.qh*p.hd+hh*p.hd:p.ko+r*p.kvh*p.hd+hh*p.hd;"                          \
    "if(isq?p.qn!=0u:p.kn!=0u){device const float*w=isq?qw+p.qwo:kw+p.kwo;float ss=0.0f;"        \
    "for(uint i=lid;i<p.hd;i+=nt){float a=x[base+i];ss+=a*a;}"                                   \
    "float sum=simd_sum(ss);if(sl==0u)partial[sg]=sum;"                                          \
    "threadgroup_barrier(mem_flags::mem_threadgroup);float total=0.0f;"                          \
    "for(uint i=0u;i<nt/32u;i++)total+=partial[i];float inv=rsqrt(total/float(p.hd)+p.eps);"     \
    "for(uint i=lid;i<p.hd;i+=nt)hv[i]=x[base+i]*inv*w[i];}"                                     \
    "else{for(uint i=lid;i<p.hd;i+=nt)hv[i]=x[base+i];}"                                         \
    "threadgroup_barrier(mem_flags::mem_threadgroup);uint hd2=p.nrot/2u;"                        \
    "for(uint i=lid;i<hd2;i+=nt){float x0=hv[i],x1=hv[i+hd2];float "                             \
    "co=c[p.co+r*p.nrot+i],si=s[p.so+r*p.nrot+i];"                                               \
    "x[base+i]=x0*co-x1*si;x[base+i+hd2]=x0*si+x1*co;}"                                          \
    "for(uint i=p.nrot+lid;i<p.hd;i+=nt)x[base+i]=hv[i];"                                        \
    "threadgroup_barrier(mem_flags::mem_device);"                                                \
    "if(isq){if(p.qsc!=0u){for(uint i=lid;i<p.hd;i+=nt)x[base+i]=x[base+i]*p.qs;}return;}"       \
    "uint cb=(p.qp+r)*p.kvh*p.hd+hh*p.hd,vb=p.vo+r*p.kvh*p.hd+hh*p.hd;"                          \
    "for(uint i=lid;i<p.hd;i+=nt){kc[p.kco+cb+i]=" T "(x[base+i]);vc[p.vco+cb+i]=" T             \
    "(v[vb+i]);}}\n"

static const char metal_rope_append_source[] =
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "struct RA{uint rows,qh,kvh,hd,nrot,qo,ko,vo,qwo,kwo,co,so,kco,vco,qp,qn,kn,qsc;float "
        "eps,qs;};\n" GEIST_METAL_ROPE_APPEND("rope_append_rows", "float")
                GEIST_METAL_ROPE_APPEND("rope_append_rows_f16", "half");

static const char metal_attn_source[] =
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "struct Rope{uint "
        "rows,heads,head_dim,x_offset,cos_offset,sin_offset,x_row_stride,rope_row_stride,rope_row_"
        "offset;};\n"
        "struct Append{uint "
        "elems,kv_out,k_offset,v_offset,k_cache_offset,v_cache_offset,q_position;};\n"
        "struct Attn{uint "
        "rows,kv_len,q_heads,kv_heads,head_dim,q_position,sliding_window,q_offset,k_cache_offset,v_"
        "cache_offset,y_offset;};\n"
        "kernel void rope_rows(device float*x[[buffer(0)]],device const "
        "float*c[[buffer(1)]],device const float*s[[buffer(2)]],constant Rope&p[[buffer(3)]],uint "
        "gid[[thread_position_in_grid]]){\n"
        " uint hd2=p.rope_row_stride/2u,total=p.rows*p.heads*hd2;if(gid>=total){return;}uint "
        "i=gid%hd2,h=(gid/hd2)%p.heads,r=gid/(hd2*p.heads);uint "
        "base=p.x_offset+r*p.x_row_stride+h*p.head_dim;uint ro=p.rope_row_offset+r;float "
        "x0=x[base+i],x1=x[base+i+hd2];float "
        "co=c[p.cos_offset+ro*p.rope_row_stride+i],si=s[p.sin_offset+ro*p.rope_row_stride+i];x["
        "base+i]=x0*co-x1*si;x[base+i+hd2]=x0*si+x1*co;\n"
        "}\n"
        "kernel void kv_append_rows(device const float*k[[buffer(0)]],device const "
        "float*v[[buffer(1)]],device float*kc[[buffer(2)]],device float*vc[[buffer(3)]],constant "
        "Append&p[[buffer(4)]],uint gid[[thread_position_in_grid]]){\n"
        " if(gid>=p.elems){return;}uint r=gid/p.kv_out,c=gid-r*p.kv_out;uint "
        "dst=(p.q_position+r)*p.kv_out+c;kc[p.k_cache_offset+dst]=k[p.k_offset+gid];vc[p.v_cache_"
        "offset+dst]=v[p.v_offset+gid];\n"
        "}\n"
        "struct CP{uint so,dof,n;};\n"
        "kernel void copy_u32(device const uint*src[[buffer(0)]],device "
        "uint*dst[[buffer(1)]],constant CP&p[[buffer(2)]],uint "
        "gid[[thread_position_in_grid]]){if(gid>=p.n){return;}dst[p.dof+gid]=src[p.so+gid];}\n"
        "kernel void attention_rows(device const float*q[[buffer(0)]],device const "
        "float*kc[[buffer(1)]],device const float*vc[[buffer(2)]],device "
        "float*y[[buffer(3)]],constant Attn&p[[buffer(4)]],uint2 "
        "tg[[threadgroup_position_in_grid]],uint lid[[thread_index_in_threadgroup]]){\n"
        " threadgroup float qv[512];threadgroup float pt[256];threadgroup float "
        "red[256];threadgroup float mls[2];uint "
        "r=tg.x,h=tg.y,hd=p.head_dim;if(r>=p.rows||h>=p.q_heads||hd>512u){return;}uint "
        "kvh=h/(p.q_heads/"
        "p.kv_heads),qpos=p.q_position+r,qb=p.q_offset+r*p.q_heads*hd+h*hd;for(uint "
        "i=lid;i<hd;i+=256u)qv[i]=q[qb+i];uint "
        "slo=(p.sliding_window>0u&&qpos+1u>p.sliding_window)?qpos+1u-p.sliding_window:0u;uint "
        "shi=qpos<p.kv_len?qpos:p.kv_len-1u;float "
        "a0=0.0f,a1=0.0f;if(lid==0u){mls[0]=-3.402823466e+38f;mls[1]=0.0f;}threadgroup_barrier(mem_"
        "flags::mem_threadgroup);for(uint tb=slo;tb<=shi;tb+=256u){uint t=tb+lid;float "
        "sc=-3.402823466e+38f;if(t<=shi){float d=0.0f;for(uint "
        "i=0u;i<hd;i++)d+=qv[i]*kc[p.k_cache_offset+t*p.kv_heads*hd+kvh*hd+i];sc=d;}red[lid]=sc;"
        "threadgroup_barrier(mem_flags::mem_threadgroup);for(uint "
        "st=128u;st>0u;st>>=1u){if(lid<st)red[lid]=red[lid]>red[lid+st]?red[lid]:red[lid+st];"
        "threadgroup_barrier(mem_flags::mem_threadgroup);}float "
        "mold=mls[0],mnew=mold>red[0]?mold:red[0],corr=exp(mold-mnew);threadgroup_barrier(mem_"
        "flags::mem_threadgroup);float "
        "e=t<=shi?exp(sc-mnew):0.0f;pt[lid]=e;red[lid]=e;threadgroup_barrier(mem_flags::mem_"
        "threadgroup);for(uint "
        "st=128u;st>0u;st>>=1u){if(lid<st)red[lid]+=red[lid+st];threadgroup_barrier(mem_flags::mem_"
        "threadgroup);}float tsum=red[0];uint "
        "tw=shi-tb+1u<256u?shi-tb+1u:256u;a0*=corr;a1*=corr;for(uint j=0u;j<tw;j++){float "
        "pv=pt[j];uint "
        "vb=p.v_cache_offset+(tb+j)*p.kv_heads*hd+kvh*hd;if(lid<hd)a0+=pv*vc[vb+lid];if(lid+256u<"
        "hd)a1+=pv*vc[vb+lid+256u];}threadgroup_barrier(mem_flags::mem_threadgroup);if(lid==0u){"
        "mls[0]=mnew;mls[1]=mls[1]*corr+tsum;}threadgroup_barrier(mem_flags::mem_threadgroup);}"
        "float inv=1.0f/mls[1];uint "
        "yb=p.y_offset+r*p.q_heads*hd+h*hd;if(lid<hd)y[yb+lid]=a0*inv;if(lid+256u<hd)y[yb+lid+256u]"
        "=a1*inv;\n"
        "}\n";

static const char metal_q4k_gate_up_n4_source[] =
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "struct P{uint n_in,n_out,rows,bpr,xo,gw,uw,gy,uy,xs,ys;};\n"
        "static inline float gelu(float x){return "
        "0.5f*x*(1.0f+tanh(clamp(0.7978845608028654f*(x+0.044715f*x*x*x),-10.0f,10.0f)));}\n"
        "static inline float dotb(device const uchar*w,uint off,uint bpr,uint row,uint ib,uint "
        "iq,uint ir,thread const float*yl,thread const float*yh,float4 sumy){constexpr ushort "
        "km1=0x3f3f,km2=0x0f0f,km3=0xc0c0;device const uchar*wb=w+off+(row*bpr+ib)*144u;device "
        "const ushort*sc=(device const ushort*)(wb+4u)+iq;device const ushort*q1=(device const "
        "ushort*)(wb+16u)+16u*iq+4u*ir;device const half*dh=(device const half*)wb;ushort "
        "sc16[4];thread const uchar*sc8=(thread const "
        "uchar*)sc16;sc16[0]=sc[0]&km1;sc16[1]=sc[2]&km1;sc16[2]=((sc[4]>>0)&km2)|((sc[0]&km3)>>2);"
        "sc16[3]=((sc[4]>>4)&km2)|((sc[2]&km3)>>2);device const ushort*q2=q1+32u;float4 "
        "a1=float4(0.0f),a2=float4(0.0f);for(uint i=0u;i<4u;i++){ushort "
        "q=q1[i];a1[0]+=yl[2u*i+0u]*float(q&0x000f);a1[1]+=yl[2u*i+1u]*float(q&0x0f00);a1[2]+=yl["
        "2u*i+8u]*float(q&0x00f0);a1[3]+=yl[2u*i+9u]*float(q&0xf000);q=q2[i];a2[0]+=yh[2u*i+0u]*"
        "float(q&0x000f);a2[1]+=yh[2u*i+1u]*float(q&0x0f00);a2[2]+=yh[2u*i+8u]*float(q&0x00f0);a2["
        "3]+=yh[2u*i+9u]*float(q&0xf000);}return "
        "float(dh[0])*((a1[0]+(1.0f/256.0f)*a1[1])*float(sc8[0])+(a1[2]+(1.0f/"
        "256.0f)*a1[3])*float(sc8[1])*(1.0f/16.0f)+(a2[0]+(1.0f/"
        "256.0f)*a2[1])*float(sc8[4])+(a2[2]+(1.0f/256.0f)*a2[3])*float(sc8[5])*(1.0f/"
        "16.0f))-float(dh[1])*(sumy[0]*float(sc8[2])+sumy[1]*float(sc8[3])+sumy[2]*float(sc8[6])+"
        "sumy[3]*float(sc8[7]));}\n"
        "static inline float silu(float x){return x/(1.0f+exp(-x));}\n";

/* The gate/up decode kernels: both weights against one activation pass,
 * ACT(gate) * up written directly. ACT is gelu (GeGLU) or silu (SwiGLU);
 * silu is silu_mul_rows' formula, so the fused SwiGLU front is
 * bit-identical to two matvecs and silu_mul. The Q4_K kernel takes 2 rows
 * per simdgroup like matvec_q4k_n4, the Q8_0 one 4 like matvec_q80_n4,
 * each output accumulated in the order of its matvec kernel (#17). */
#define GEIST_METAL_GATE_UP_Q4K(NAME, ACT)                                                       \
    "kernel void " NAME "(device const float*x[[buffer(0)]],device const "                       \
    "uchar*wg[[buffer(1)]],device const uchar*wu[[buffer(2)]],device "                           \
    "float*gy[[buffer(3)]],device float*uy[[buffer(4)]],constant P&p[[buffer(5)]],uint3 "        \
    "tg[[threadgroup_position_in_grid]],ushort ti[[thread_index_in_simdgroup]],ushort "          \
    "sg[[simdgroup_index_in_threadgroup]]){(void)uy;uint "                                       \
    "ix=uint(ti)/"                                                                               \
    "8u,it=uint(ti)&7u,iq=it>>2u,ir=it&3u,b=tg.y,fr=(tg.x*2u+uint(sg))*2u;if(fr>=p.n_out||b>=p." \
    "rows)return;uint nb=p.n_in>>8u;float g0=0.0f,g1=0.0f,u0=0.0f,u1=0.0f;device const "         \
    "float*y4=x+p.xo+b*p.xs+ix*256u+64u*iq+8u*ir;for(uint ib=ix;ib<nb;ib+=4u){float "            \
    "yl[16],yh[16];float4 sumy=float4(0.0f);for(uint "                                           \
    "i=0u;i<8u;i++){yl[i]=y4[i];sumy[0]+=yl[i];yl[i+8u]=y4[i+32u];sumy[1]+=yl[i+8u];yh[i]=y4[i+" \
    "128u];sumy[2]+=yh[i];yh[i+8u]=y4[i+160u];sumy[3]+=yh[i+8u];}if(fr<p.n_out){g0+=dotb(wg,p."  \
    "gw,p.bpr,fr,ib,iq,ir,yl,yh,sumy);u0+=dotb(wu,p.uw,p.bpr,fr,ib,iq,ir,yl,yh,sumy);}if(fr+1u<" \
    "p.n_out){g1+=dotb(wg,p.gw,p.bpr,fr+1u,ib,iq,ir,yl,yh,sumy);u1+=dotb(wu,p.uw,p.bpr,fr+1u,"   \
    "ib,iq,ir,yl,yh,sumy);}y4+=1024u;}float "                                                    \
    "ag0=simd_sum(g0),au0=simd_sum(u0),ag1=simd_sum(g1),au1=simd_sum(u1);if(ti==0){uint "        \
    "go=p.gy+b*p.ys+fr;gy[go]=" ACT "(ag0)*au0;if(fr+1u<p.n_out)gy[go+1u]=" ACT "(ag1)*au1;}}\n"

#define GEIST_METAL_GATE_UP_Q80(NAME, ACT)                                                     \
    "kernel void " NAME "(device const float*x[[buffer(0)]],device const "                     \
    "uchar*wg[[buffer(1)]],device const uchar*wu[[buffer(2)]],device "                         \
    "float*gy[[buffer(3)]],device float*uy[[buffer(4)]],constant P&p[[buffer(5)]],uint3 "      \
    "tg[[threadgroup_position_in_grid]],ushort ti[[thread_index_in_simdgroup]],ushort "        \
    "sg[[simdgroup_index_in_threadgroup]]){(void)uy;"                                          \
    "uint b=tg.y,fr=(tg.x*2u+uint(sg))*4u;if(fr>=p.n_out||b>=p.rows)return;"                   \
    "uint nb=p.n_in>>5u,ix=uint(ti)>>1u,il=(uint(ti)&1u)*16u;"                                 \
    "uint nr=min(p.n_out-fr,4u);"                                                              \
    "device const float*yb=x+p.xo+b*p.xs+ix*32u+il;"                                           \
    "float sgt[4]={0.0f,0.0f,0.0f,0.0f},sup[4]={0.0f,0.0f,0.0f,0.0f};"                         \
    "for(uint ib=ix;ib<nb;ib+=16u){"                                                           \
    "float yl[16];for(uint i=0u;i<16u;i++)yl[i]=yb[i];"                                        \
    "for(uint rr=0u;rr<nr;rr++){"                                                              \
    "uint go=p.gw+((fr+rr)*p.bpr+ib)*34u,uo=p.uw+((fr+rr)*p.bpr+ib)*34u;"                      \
    "device const char*qg=(device const char*)(wg+go+2u+il);"                                  \
    "device const char*qu=(device const char*)(wu+uo+2u+il);"                                  \
    "float ag=0.0f,au=0.0f;for(uint i=0u;i<16u;i++){ag+=yl[i]*float(qg[i]);"                   \
    "au+=yl[i]*float(qu[i]);}"                                                                 \
    "float dg=float(*((device const half*)(wg+go))),du=float(*((device const half*)(wu+uo)));" \
    "sgt[rr]+=dg*ag;sup[rr]+=du*au;}"                                                          \
    "yb+=512u;}"                                                                               \
    "for(uint rr=0u;rr<4u;rr++){float g=simd_sum(sgt[rr]),u=simd_sum(sup[rr]);"                \
    "if(ti==0&&fr+rr<p.n_out)gy[p.gy+b*p.ys+fr+rr]=" ACT "(g)*u;}}\n"

static const char metal_gate_up_q4k_gelu_kernel[] =
        GEIST_METAL_GATE_UP_Q4K("gate_up_q4k_n4", "gelu");
/* The SwiGLU Q4_K kernel runs matvec_q4k_n4's loop over four rows, two
 * of gate and two of up, each accumulated by the same statement as in the
 * matvec, so the result is that of two matvecs and silu_mul bit for bit.
 * (dotb above contracts differently; the GeGLU kernel keeps it and its
 * bits.) */
static const char metal_gate_up_q4k_silu_kernel[] =
        "kernel void gate_up_silu_q4k_n4(device const float*x[[buffer(0)]],device const "
        "uchar*wg[[buffer(1)]],device const uchar*wu[[buffer(2)]],device "
        "float*gy[[buffer(3)]],device float*uy[[buffer(4)]],constant P&p[[buffer(5)]],uint3 "
        "tg[[threadgroup_position_in_grid]],ushort ti[[thread_index_in_simdgroup]],ushort "
        "sg[[simdgroup_index_in_threadgroup]]){(void)uy;constexpr ushort "
        "km1=0x3f3f,km2=0x0f0f,km3=0xc0c0;uint ix=uint(ti)/8u,it=uint(ti)&7u,iq=it>>2u,ir=it&3u,"
        "b=tg.y,fr=(tg.x*2u+uint(sg))*2u;if(fr>=p.n_out||b>=p.rows)return;uint "
        "nb=p.n_in>>8u;float sum0=0.0f,sum1=0.0f,sum2=0.0f,sum3=0.0f;device const "
        "float*y4=x+p.xo+b*p.xs+ix*256u+64u*iq+8u*ir;ushort sc16[4];thread const uchar*sc8=(thread "
        "const uchar*)sc16;for(uint ib=ix;ib<nb;ib+=4u){float yl[16],yh[16];float4 "
        "sumy=float4(0.0f);for(uint "
        "i=0u;i<8u;i++){yl[i]=y4[i];sumy[0]+=yl[i];yl[i+8u]=y4[i+32u];sumy[1]+=yl[i+8u];yh[i]=y4[i+"
        "128u];sumy[2]+=yh[i];yh[i+8u]=y4[i+160u];sumy[3]+=yh[i+8u];}for(uint "
        "rr=0u;rr<4u;rr++){uint row=fr+(rr&1u);if(row>=p.n_out)continue;device const "
        "uchar*wb=(rr<2u?wg+p.gw:wu+p.uw)+(row*p.bpr+ib)*144u;device const ushort*sc=(device const "
        "ushort*)(wb+4u)+iq;device const ushort*q1=(device const "
        "ushort*)(wb+16u)+16u*iq+4u*ir;device const half*dh=(device const "
        "half*)wb;sc16[0]=sc[0]&km1;sc16[1]=sc[2]&km1;sc16[2]=((sc[4]>>0)&km2)|((sc[0]&km3)>>2);"
        "sc16[3]=((sc[4]>>4)&km2)|((sc[2]&km3)>>2);device const ushort*q2=q1+32u;float4 "
        "a1=float4(0.0f),a2=float4(0.0f);for(uint i=0u;i<4u;i++){ushort "
        "q=q1[i];a1[0]+=yl[2u*i+0u]*float(q&0x000f);a1[1]+=yl[2u*i+1u]*float(q&0x0f00);a1[2]+=yl["
        "2u*i+8u]*float(q&0x00f0);a1[3]+=yl[2u*i+9u]*float(q&0xf000);q=q2[i];a2[0]+=yh[2u*i+0u]*"
        "float(q&0x000f);a2[1]+=yh[2u*i+1u]*float(q&0x0f00);a2[2]+=yh[2u*i+8u]*float(q&0x00f0);a2["
        "3]+=yh[2u*i+9u]*float(q&0xf000);}float "
        "acc=float(dh[0])*((a1[0]+(1.0f/256.0f)*a1[1])*float(sc8[0])+(a1[2]+(1.0f/"
        "256.0f)*a1[3])*float(sc8[1])*(1.0f/16.0f)+(a2[0]+(1.0f/"
        "256.0f)*a2[1])*float(sc8[4])+(a2[2]+(1.0f/256.0f)*a2[3])*float(sc8[5])*(1.0f/"
        "16.0f))-float(dh[1])*(sumy[0]*float(sc8[2])+sumy[1]*float(sc8[3])+sumy[2]*float(sc8[6])+"
        "sumy[3]*float(sc8[7]));if(rr==0u)sum0+=acc;else if(rr==1u)sum1+=acc;else "
        "if(rr==2u)sum2+=acc;else sum3+=acc;}y4+=1024u;}float "
        "g0=simd_sum(sum0),g1=simd_sum(sum1),u0=simd_sum(sum2),u1=simd_sum(sum3);if(ti==0){uint "
        "o=p.gy+b*p.ys+fr;gy[o]=silu(g0)*u0;if(fr+1u<p.n_out)gy[o+1u]=silu(g1)*u1;}}\n";
static const char metal_gate_up_q80_gelu_kernel[] =
        GEIST_METAL_GATE_UP_Q80("gate_up_q80_n4", "gelu");
static const char metal_gate_up_q80_silu_kernel[] =
        GEIST_METAL_GATE_UP_Q80("gate_up_silu_q80_n4", "silu");

/* q, k and v of a decode step in one dispatch (#17): the grid is the
 * three matvecs' threadgroups side by side, and each threadgroup runs the
 * body of its weight's matvec kernel (mv_q4k, mv_q6k, mv_q80 above) on its
 * own range, so every output is the matvec kernel's bits. One kernel per
 * dtype triple keeps each body's register use its own. */
static const char metal_mv3_head[] = "struct T3{uint g1,g2;P p0,p1,p2;};\n";
#define GEIST_METAL_MV3(A, B, C)                                                                   \
    "kernel void mv3_" A "_" B "_" C "(device const float*x[[buffer(0)]],device const "            \
    "uchar*w0[[buffer(1)]],device const uchar*w1[[buffer(2)]],device const "                       \
    "uchar*w2[[buffer(3)]],device float*y0[[buffer(4)]],device float*y1[[buffer(5)]],device "      \
    "float*y2[[buffer(6)]],constant T3&t[[buffer(7)]],uint3 "                                      \
    "tg[[threadgroup_position_in_grid]],ushort ti[[thread_index_in_simdgroup]],ushort "            \
    "sg[[simdgroup_index_in_threadgroup]]){if(tg.x<t.g1){mv_" A "(x,w0,y0,t.p0,tg.x,tg.y,ti,sg);}" \
    "else if(tg.x<t.g2){mv_" B "(x,w1,y1,t.p1,tg.x-t.g1,tg.y,ti,sg);}"                             \
    "else{mv_" C "(x,w2,y2,t.p2,tg.x-t.g2,tg.y,ti,sg);}}\n"
static const char metal_mv3_kernels[] = GEIST_METAL_MV3("q4k", "q4k", "q4k")
        GEIST_METAL_MV3("q4k", "q4k", "q6k") GEIST_METAL_MV3("q80", "q80", "q80");

static const char metal_q4k_pair_n4_source[] =
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "struct P{uint n_in,n_out,rows,bpr,xo,gw,uw,gy,uy,xs,ys;};\n"
        "static inline float gelu(float x){return "
        "0.5f*x*(1.0f+tanh(clamp(0.7978845608028654f*(x+0.044715f*x*x*x),-10.0f,10.0f)));}\n"
        "static inline float dotb(device const uchar*w,uint off,uint bpr,uint row,uint ib,uint "
        "iq,uint ir,thread const float*yl,thread const float*yh,float4 sumy){constexpr ushort "
        "km1=0x3f3f,km2=0x0f0f,km3=0xc0c0;device const uchar*wb=w+off+(row*bpr+ib)*144u;device "
        "const ushort*sc=(device const ushort*)(wb+4u)+iq;device const ushort*q1=(device const "
        "ushort*)(wb+16u)+16u*iq+4u*ir;device const half*dh=(device const half*)wb;ushort "
        "sc16[4];thread const uchar*sc8=(thread const "
        "uchar*)sc16;sc16[0]=sc[0]&km1;sc16[1]=sc[2]&km1;sc16[2]=((sc[4]>>0)&km2)|((sc[0]&km3)>>2);"
        "sc16[3]=((sc[4]>>4)&km2)|((sc[2]&km3)>>2);device const ushort*q2=q1+32u;float4 "
        "a1=float4(0.0f),a2=float4(0.0f);for(uint i=0u;i<4u;i++){ushort "
        "q=q1[i];a1[0]+=yl[2u*i+0u]*float(q&0x000f);a1[1]+=yl[2u*i+1u]*float(q&0x0f00);a1[2]+=yl["
        "2u*i+8u]*float(q&0x00f0);a1[3]+=yl[2u*i+9u]*float(q&0xf000);q=q2[i];a2[0]+=yh[2u*i+0u]*"
        "float(q&0x000f);a2[1]+=yh[2u*i+1u]*float(q&0x0f00);a2[2]+=yh[2u*i+8u]*float(q&0x00f0);a2["
        "3]+=yh[2u*i+9u]*float(q&0xf000);}return "
        "float(dh[0])*((a1[0]+(1.0f/256.0f)*a1[1])*float(sc8[0])+(a1[2]+(1.0f/"
        "256.0f)*a1[3])*float(sc8[1])*(1.0f/16.0f)+(a2[0]+(1.0f/"
        "256.0f)*a2[1])*float(sc8[4])+(a2[2]+(1.0f/256.0f)*a2[3])*float(sc8[5])*(1.0f/"
        "16.0f))-float(dh[1])*(sumy[0]*float(sc8[2])+sumy[1]*float(sc8[3])+sumy[2]*float(sc8[6])+"
        "sumy[3]*float(sc8[7]));}\n"
        "kernel void pair_q4k_n4(device const float*x[[buffer(0)]],device const "
        "uchar*wg[[buffer(1)]],device const uchar*wu[[buffer(2)]],device "
        "float*gy[[buffer(3)]],device float*uy[[buffer(4)]],constant P&p[[buffer(5)]],uint3 "
        "tg[[threadgroup_position_in_grid]],ushort ti[[thread_index_in_simdgroup]],ushort "
        "sg[[simdgroup_index_in_threadgroup]]){uint "
        "ix=uint(ti)/"
        "8u,it=uint(ti)&7u,iq=it>>2u,ir=it&3u,b=tg.y,fr=(tg.x*2u+uint(sg))*2u;if(fr>=p.n_out||b>=p."
        "rows)return;uint nb=p.n_in>>8u;float g0=0.0f,g1=0.0f,u0=0.0f,u1=0.0f;device const "
        "float*y4=x+p.xo+b*p.xs+ix*256u+64u*iq+8u*ir;for(uint ib=ix;ib<nb;ib+=4u){float "
        "yl[16],yh[16];float4 sumy=float4(0.0f);for(uint "
        "i=0u;i<8u;i++){yl[i]=y4[i];sumy[0]+=yl[i];yl[i+8u]=y4[i+32u];sumy[1]+=yl[i+8u];yh[i]=y4[i+"
        "128u];sumy[2]+=yh[i];yh[i+8u]=y4[i+160u];sumy[3]+=yh[i+8u];}if(fr<p.n_out){g0+=dotb(wg,p."
        "gw,p.bpr,fr,ib,iq,ir,yl,yh,sumy);u0+=dotb(wu,p.uw,p.bpr,fr,ib,iq,ir,yl,yh,sumy);}if(fr+1u<"
        "p.n_out){g1+=dotb(wg,p.gw,p.bpr,fr+1u,ib,iq,ir,yl,yh,sumy);u1+=dotb(wu,p.uw,p.bpr,fr+1u,"
        "ib,iq,ir,yl,yh,sumy);}y4+=1024u;}float "
        "ag0=simd_sum(g0),au0=simd_sum(u0),ag1=simd_sum(g1),au1=simd_sum(u1);if(ti==0){uint "
        "go=p.gy+b*p.ys+fr,uo=p.uy+b*p.ys+fr;gy[go]=ag0;uy[uo]=au0;if(fr+1u<p.n_out){gy[go+1u]=ag1;"
        "uy[uo+1u]=au1;}}}\n";

static const char metal_argmax_source[] =
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "struct P{uint n,xo;};\n"
        "struct PB{uint rows,n,xo,xs,oo;};\n"
        "kernel void argmax_f32(device const float*x[[buffer(0)]],device "
        "uint*out[[buffer(1)]],constant P&p[[buffer(2)]],uint "
        "lid[[thread_index_in_threadgroup]]){threadgroup float bv[256];threadgroup uint "
        "bi[256];float best=-3.402823466e+38f;uint idx=0u;for(uint i=lid;i<p.n;i+=256u){float "
        "v=x[p.xo+i];if(v>best||(v==best&&i<idx)){best=v;idx=i;}}bv[lid]=best;bi[lid]=idx;"
        "threadgroup_barrier(mem_flags::mem_threadgroup);for(uint "
        "st=128u;st>0u;st>>=1u){if(lid<st){float ov=bv[lid+st];uint "
        "oi=bi[lid+st];if(ov>bv[lid]||(ov==bv[lid]&&oi<bi[lid])){bv[lid]=ov;bi[lid]=oi;}}"
        "threadgroup_barrier(mem_flags::mem_threadgroup);}if(lid==0u){out[0]=bi[0];}}\n"
        "kernel void argmax_f32_batch(device const float*x[[buffer(0)]],device "
        "uint*out[[buffer(1)]],constant PB&p[[buffer(2)]],uint "
        "row[[threadgroup_position_in_grid]],uint lid[[thread_index_in_threadgroup]]){threadgroup "
        "float bv[256];threadgroup uint bi[256];if(row>=p.rows)return;uint "
        "base=p.xo+row*p.xs;float best=-3.402823466e+38f;uint idx=0u;for(uint "
        "i=lid;i<p.n;i+=256u){float "
        "v=x[base+i];if(v>best||(v==best&&i<idx)){best=v;idx=i;}}bv[lid]=best;bi[lid]=idx;"
        "threadgroup_barrier(mem_flags::mem_threadgroup);for(uint "
        "st=128u;st>0u;st>>=1u){if(lid<st){float ov=bv[lid+st];uint "
        "oi=bi[lid+st];if(ov>bv[lid]||(ov==bv[lid]&&oi<bi[lid])){bv[lid]=ov;bi[lid]=oi;}}"
        "threadgroup_barrier(mem_flags::mem_threadgroup);}if(lid==0u){out[p.oo+row]=bi[0];}}\n";

static const char metal_attn_f16_source[] =
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "struct Append{uint "
        "elems,kv_out,k_offset,v_offset,k_cache_offset,v_cache_offset,q_position;};\n"
        "struct Attn{uint "
        "rows,kv_len,q_heads,kv_heads,head_dim,q_position,sliding_window,q_offset,k_cache_offset,v_"
        "cache_offset,y_offset;};\n"
        "kernel void kv_append_rows_f16(device const float*k[[buffer(0)]],device const "
        "float*v[[buffer(1)]],device half*kc[[buffer(2)]],device half*vc[[buffer(3)]],constant "
        "Append&p[[buffer(4)]],uint gid[[thread_position_in_grid]]){\n"
        " if(gid>=p.elems){return;}uint r=gid/p.kv_out,c=gid-r*p.kv_out;uint "
        "dst=(p.q_position+r)*p.kv_out+c;kc[p.k_cache_offset+dst]=half(k[p.k_offset+gid]);vc[p.v_"
        "cache_offset+dst]=half(v[p.v_offset+gid]);\n"
        "}\n"
        "kernel void attention_rows_f16(device const float*q[[buffer(0)]],device const "
        "half*kc[[buffer(1)]],device const half*vc[[buffer(2)]],device "
        "float*y[[buffer(3)]],constant Attn&p[[buffer(4)]],uint2 "
        "tg[[threadgroup_position_in_grid]],uint lid[[thread_index_in_threadgroup]]){\n"
        " threadgroup float qv[512];threadgroup float pt[256];threadgroup float "
        "red[256];threadgroup float mls[2];uint "
        "r=tg.x,h=tg.y,hd=p.head_dim;if(r>=p.rows||h>=p.q_heads||hd>512u){return;}uint "
        "kvh=h/(p.q_heads/"
        "p.kv_heads),qpos=p.q_position+r,qb=p.q_offset+r*p.q_heads*hd+h*hd;for(uint "
        "i=lid;i<hd;i+=256u)qv[i]=q[qb+i];uint "
        "slo=(p.sliding_window>0u&&qpos+1u>p.sliding_window)?qpos+1u-p.sliding_window:0u;uint "
        "shi=qpos<p.kv_len?qpos:p.kv_len-1u;float "
        "a0=0.0f,a1=0.0f;if(lid==0u){mls[0]=-3.402823466e+38f;mls[1]=0.0f;}threadgroup_barrier(mem_"
        "flags::mem_threadgroup);for(uint tb=slo;tb<=shi;tb+=256u){uint t=tb+lid;float "
        "sc=-3.402823466e+38f;if(t<=shi){float d=0.0f;for(uint "
        "i=0u;i<hd;i++)d+=qv[i]*float(kc[p.k_cache_offset+t*p.kv_heads*hd+kvh*hd+i]);sc=d;}red[lid]"
        "=sc;threadgroup_barrier(mem_flags::mem_threadgroup);for(uint "
        "st=128u;st>0u;st>>=1u){if(lid<st)red[lid]=red[lid]>red[lid+st]?red[lid]:red[lid+st];"
        "threadgroup_barrier(mem_flags::mem_threadgroup);}float "
        "mold=mls[0],mnew=mold>red[0]?mold:red[0],corr=exp(mold-mnew);threadgroup_barrier(mem_"
        "flags::mem_threadgroup);float "
        "e=t<=shi?exp(sc-mnew):0.0f;pt[lid]=e;red[lid]=e;threadgroup_barrier(mem_flags::mem_"
        "threadgroup);for(uint "
        "st=128u;st>0u;st>>=1u){if(lid<st)red[lid]+=red[lid+st];threadgroup_barrier(mem_flags::mem_"
        "threadgroup);}float tsum=red[0];uint "
        "tw=shi-tb+1u<256u?shi-tb+1u:256u;a0*=corr;a1*=corr;for(uint j=0u;j<tw;j++){float "
        "pv=pt[j];uint "
        "vb=p.v_cache_offset+(tb+j)*p.kv_heads*hd+kvh*hd;if(lid<hd)a0+=pv*float(vc[vb+lid]);if(lid+"
        "256u<hd)a1+=pv*float(vc[vb+lid+256u]);}threadgroup_barrier(mem_flags::mem_threadgroup);if("
        "lid==0u){mls[0]=mnew;mls[1]=mls[1]*corr+tsum;}threadgroup_barrier(mem_flags::mem_"
        "threadgroup);}float inv=1.0f/mls[1];uint "
        "yb=p.y_offset+r*p.q_heads*hd+h*hd;if(lid<hd)y[yb+lid]=a0*inv;if(lid+256u<hd)y[yb+lid+256u]"
        "=a1*inv;\n"
        "}\n";

/* Decode-specialized fused qnorm+rope+attention (rows==1, f16 KV): tokens are
 * split across the 8 simdgroups with per-simdgroup online softmax kept in
 * registers (simd_sum reductions, no tree-reduce barriers, no serial PV
 * loop); one threadgroup-memory merge at the end. */
/* The plain split-KV decode kernels compile twice: for an F16 KV cache
 * and for an F32 one. A defs part ahead of the shared source picks the K/V
 * element type KVT, KVF(x) (the float a K/V element is computed with) and
 * the kernel name ATTN(n). F32 KV is rounded to half as it is read: the
 * round-to-nearest the f32->f16 staging copy applied to the same element,
 * so the result is bit-identical to staging first, without the pass over
 * every live row per token (#64). */
static const char metal_attn_kv_header[] =
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "struct A{uint rows,kv_len,qh,kvh,hd,qpos,sw,qo,kco,vco,yo;};\n"
        "struct SP{uint ns;};\n";
static const char metal_attn_kv_f16_defs[] = "\n#define KVT half\n"
                                             "#define KVF(x) float(x)\n"
                                             "#define ATTN(n) attention_##n##_f16\n";
static const char metal_attn_kv_f32_defs[] = "\n#define KVT float\n"
                                             "#define KVF(x) float(half(x))\n"
                                             "#define ATTN(n) attention_##n##_f32kv\n";

static const char metal_attn_qnorm_dec_f16_source[] =
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "struct NR{uint rows,heads,hd,xo,wo,co,so,xs,rs,ro;float eps;};\n"
        "struct A{uint rows,kv_len,qh,kvh,hd,qpos,sw,qo,kco,vco,yo;};\n"
        "struct SP{uint ns;};\n"
        "kernel void attention_qnorm_dec_f16(device const float*q[[buffer(0)]],device const "
        "float*qw[[buffer(1)]],device const float*c[[buffer(2)]],device const "
        "float*s[[buffer(3)]],device const half*kc[[buffer(4)]],device const "
        "half*vc[[buffer(5)]],device float*pb[[buffer(6)]],constant NR&nr[[buffer(7)]],constant "
        "A&p[[buffer(8)]],constant SP&sp[[buffer(9)]],uint2 "
        "tg[[threadgroup_position_in_grid]],uint lid[[thread_index_in_threadgroup]],uint "
        "sg[[simdgroup_index_in_threadgroup]],uint ln[[thread_index_in_simdgroup]]){"
        "threadgroup float qv[256];threadgroup float red[8];threadgroup float tm[8];threadgroup "
        "float tl[8];threadgroup float ta[2048];"
        "uint h=tg.y,ss=tg.x,hd=p.hd;if(h>=p.qh||hd>256u)return;"
        "uint base=nr.xo+h*nr.hd;"
        "float ssq=0.0f;for(uint i=lid;i<hd;i+=256u){float v=q[base+i];ssq+=v*v;}"
        "ssq=simd_sum(ssq);if(ln==0u)red[sg]=ssq;threadgroup_barrier(mem_flags::mem_threadgroup);"
        "float tot=red[0]+red[1]+red[2]+red[3]+red[4]+red[5]+red[6]+red[7];"
        "float invn=rsqrt(tot/float(hd)+nr.eps);"
        "uint hd2=hd/2u,rr=nr.ro;"
        "for(uint i=lid;i<hd2;i+=256u){float "
        "x0=q[base+i]*invn*qw[nr.wo+i],x1=q[base+i+hd2]*invn*qw[nr.wo+i+hd2];float "
        "co=c[nr.co+rr*nr.rs+i],si=s[nr.so+rr*nr.rs+i];qv[i]=x0*co-x1*si;qv[i+hd2]=x0*si+x1*co;}"
        "threadgroup_barrier(mem_flags::mem_threadgroup);";

/* Split-KV decode body shared by the qnorm and the no-norm variants (the
 * array above ends mid-kernel; this body closes it and is appended a second
 * time after the plain head below at library init). */
static const char metal_attn_dec_f16_body[] =
        "uint kvh=h/(p.qh/p.kvh),qp=p.qpos;"
        "uint slo=(p.sw>0u&&qp+1u>p.sw)?qp+1u-p.sw:0u;uint shi=qp<p.kv_len?qp:p.kv_len-1u;"
        /* fixed 8-chunk unroll (hd<=256) keeps a[] in registers — a runtime
         * bound would spill the accumulators to thread-private device memory */
        "float m=-3.402823466e+38f,l=0.0f,a0=0,a1=0,a2=0,a3=0,a4=0,a5=0,a6=0,a7=0;"
        "bool "
        "g1=ln+32u<hd,g2=ln+64u<hd,g3=ln+96u<hd,g4=ln+128u<hd,g5=ln+160u<hd,g6=ln+192u<hd,g7=ln+"
        "224u<hd;"
        "for(uint t=slo+ss*8u+sg;t<=shi;t+=sp.ns*8u){"
        "uint kb=p.kco+t*p.kvh*hd+kvh*hd;"
        "float d=qv[ln]*KVF(kc[kb+ln]);"
        "if(g1)d+=qv[ln+32u]*KVF(kc[kb+ln+32u]);"
        "if(g2)d+=qv[ln+64u]*KVF(kc[kb+ln+64u]);"
        "if(g3)d+=qv[ln+96u]*KVF(kc[kb+ln+96u]);"
        "if(g4)d+=qv[ln+128u]*KVF(kc[kb+ln+128u]);"
        "if(g5)d+=qv[ln+160u]*KVF(kc[kb+ln+160u]);"
        "if(g6)d+=qv[ln+192u]*KVF(kc[kb+ln+192u]);"
        "if(g7)d+=qv[ln+224u]*KVF(kc[kb+ln+224u]);"
        "d=simd_sum(d);"
        "float mn=max(m,d),corr=exp(m-mn),e=exp(d-mn);"
        "l=l*corr+e;"
        "uint vb=p.vco+t*p.kvh*hd+kvh*hd;"
        "a0=a0*corr+e*KVF(vc[vb+ln]);"
        "if(g1)a1=a1*corr+e*KVF(vc[vb+ln+32u]);"
        "if(g2)a2=a2*corr+e*KVF(vc[vb+ln+64u]);"
        "if(g3)a3=a3*corr+e*KVF(vc[vb+ln+96u]);"
        "if(g4)a4=a4*corr+e*KVF(vc[vb+ln+128u]);"
        "if(g5)a5=a5*corr+e*KVF(vc[vb+ln+160u]);"
        "if(g6)a6=a6*corr+e*KVF(vc[vb+ln+192u]);"
        "if(g7)a7=a7*corr+e*KVF(vc[vb+ln+224u]);"
        "m=mn;}"
        "if(ln==0u){tm[sg]=m;tl[sg]=l;}"
        "uint tb=sg*hd+ln;"
        "ta[tb]=a0;"
        "if(g1)ta[tb+32u]=a1;if(g2)ta[tb+64u]=a2;if(g3)ta[tb+96u]=a3;"
        "if(g4)ta[tb+128u]=a4;if(g5)ta[tb+160u]=a5;if(g6)ta[tb+192u]=a6;if(g7)ta[tb+224u]=a7;"
        "threadgroup_barrier(mem_flags::mem_threadgroup);"
        "float M=tm[0];for(uint j=1u;j<8u;j++)M=max(M,tm[j]);"
        "float L=0.0f;for(uint j=0u;j<8u;j++)L+=tl[j]*exp(tm[j]-M);"
        "uint pbb=(h*32u+ss)*(hd+2u);"
        "if(lid==0u){pb[pbb]=M;pb[pbb+1u]=L;}"
        "for(uint i=lid;i<hd;i+=256u){float acc=0.0f;for(uint "
        "j=0u;j<8u;j++)acc+=ta[j*hd+i]*exp(tm[j]-M);pb[pbb+2u+i]=acc;}}\n";

/* Prefill flash attention (#91): one template over head_dim D (a
 * multiple of 32 up to 256 on 4 simdgroups, of 64 up to 512 on 8) after
 * the structure of llama.cpp's kernel_flash_attn_ext. A threadgroup owns
 * R = 16 query rows of one head (two 8-row tiles) and walks their live
 * keys in 64-key blocks:
 *   - Q^T K: each simdgroup computes whole 8x8 score tiles over the full
 *     head_dim, K loaded transposed straight from the cache and reused for
 *     both row tiles; no cross-simdgroup reduction.
 *   - online softmax on all lanes (two scores per lane and row, simd_max /
 *     simd_sum), causal and window masks computed in place; the weights go
 *     to threadgroup memory as half.
 *   - P.V: each simdgroup keeps its 8-column slices of O for both row tiles
 *     in registers, rescaled by a diagonal multiply.
 * Two barriers per block. Operands f16; scores, softmax and O in f32.
 * The tail block re-covers [thi-63, thi] with the keys already processed
 * masked, so kv_len >= 64 keeps every cache read in bounds. Measured on
 * the M1 Max (#91): 16 rows beat 8 (+25-30 % on Bonsai 2 27B shapes),
 * 24/32 rows spill the O registers at head_dim 256, 32-key blocks and O
 * in threadgroup memory were slower. Two literals for the ISO 4095-char
 * limit. */
static const char metal_attn_flash_source_a[] =
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "struct A{uint rows,kv_len,qh,kvh,hd,qpos,sw,qo,kco,vco,yo;};\n"
        "template<short D,short NSG>\n"
        "kernel void attention_flash(device const float*q[[buffer(0)]],device const "
        "half*kc[[buffer(1)]],device const half*vc[[buffer(2)]],device float*y[[buffer(3)]],"
        "constant A&p[[buffer(4)]],uint2 tg[[threadgroup_position_in_grid]],ushort "
        "sg[[simdgroup_index_in_threadgroup]],ushort ln[[thread_index_in_simdgroup]]){"
        "constexpr short C=64,T=2,R=8*T,NQ=R/NSG,NO=D/8/NSG,NC=C/8/NSG;"
        "threadgroup half sq[R*D];threadgroup float ss[R*C];threadgroup half sp[R*C];"
        "threadgroup float sd[T*64];"
        "const uint h=tg.y,q0=tg.x*R,kvh=h/(p.qh/p.kvh),kstr=p.kvh*D;"
        "device const half*kp=kc+p.kco+kvh*D;device const half*vp=vc+p.vco+kvh*D;"
        /* row j of the tile is jj*NSG+sg; a ragged last tile repeats the
         * last row and stores only the valid ones (#35) */
        "uint qp[NQ];"
        "for(short jj=0;jj<NQ;jj++){const short j=jj*NSG+sg;"
        "const uint row=min(q0+j,p.rows-1u);qp[jj]=p.qpos+row;"
        "device const float*qr=q+p.qo+row*p.qh*D+h*D;"
        "for(short i=ln;i<D;i+=32)sq[j*D+i]=half(qr[i]);}"
        "for(short i=sg*32+ln;i<T*64;i+=NSG*32)sd[i]=0.0f;"
        "threadgroup_barrier(mem_flags::mem_threadgroup);"
        "simdgroup_float8x8 lo[T][NO];"
        "for(short t=0;t<T;t++)for(short i=0;i<NO;i++)"
        "lo[t][i]=make_filled_simdgroup_matrix<float,8>(0.0f);"
        "float M[NQ],S[NQ];for(short jj=0;jj<NQ;jj++){M[jj]=-FLT_MAX/2;S[jj]=0.0f;}"
        "const uint qf=p.qpos+q0,ql=p.qpos+min(q0+R-1u,p.rows-1u);"
        "const uint thi=min(ql,p.kv_len-1u);"
        "const uint tlo=(p.sw>0u&&qf+1u>p.sw)?qf+1u-p.sw:0u;"
        "const uint span=thi+1u-tlo,nfull=span/C,nit=nfull+(span%C>0u?1u:0u);"
        "for(uint it=0u;it<nit;it++){"
        "const uint t0=it<nfull?tlo+it*C:(thi+1u>=C?thi+1u-C:0u);"
        "const uint skipb=it<nfull?0u:tlo+nfull*C;"
        "for(short cc=0;cc<NC;cc++){const short kb=cc*NSG+sg;"
        "simdgroup_float8x8 mqk[T];"
        "for(short t=0;t<T;t++)mqk[t]=make_filled_simdgroup_matrix<float,8>(0.0f);"
        "device const half*pk=kp+(t0+8u*kb)*kstr;"
        "for(short i=0;i<D/8;i++){simdgroup_half8x8 mk;simdgroup_load(mk,pk+8*i,kstr,0,true);"
        "for(short t=0;t<T;t++){simdgroup_half8x8 mq;simdgroup_load(mq,sq+t*8*D+8*i,D);"
        "simdgroup_multiply_accumulate(mqk[t],mq,mk,mqk[t]);}}"
        "for(short t=0;t<T;t++)simdgroup_store(mqk[t],ss+t*8*C+8*kb,C);}"
        "threadgroup_barrier(mem_flags::mem_threadgroup);";

static const char metal_attn_flash_source_b[] =
        "for(short jj=0;jj<NQ;jj++){const short j=jj*NSG+sg;"
        "const uint slr=(p.sw>0u&&qp[jj]+1u>p.sw)?qp[jj]+1u-p.sw:0u;"
        "float2 s=*(threadgroup float2*)(ss+j*C+2*ln);"
        "const uint t=t0+2u*ln;"
        "if(t<skipb||t>qp[jj]||t<slr)s.x=-FLT_MAX;"
        "if(t+1u<skipb||t+1u>qp[jj]||t+1u<slr)s.y=-FLT_MAX;"
        "const float m=M[jj];M[jj]=simd_max(max(m,max(s.x,s.y)));"
        "const float ms=exp(m-M[jj]);const float2 vs=exp(s-M[jj]);"
        "S[jj]=S[jj]*ms+simd_sum(vs.x+vs.y);"
        "*(threadgroup half2*)(sp+j*C+2*ln)=half2(vs);"
        "if(ln==0)sd[(j/8)*64+(j%8)*9]=ms;}"
        "threadgroup_barrier(mem_flags::mem_threadgroup);"
        "for(short t=0;t<T;t++){simdgroup_float8x8 md;simdgroup_load(md,sd+t*64,8);"
        "for(short ii=0;ii<NO;ii++)simdgroup_multiply(lo[t][ii],md,lo[t][ii]);}"
        "for(short cc=0;cc<C/8;cc++){simdgroup_half8x8 mp[T];"
        "for(short t=0;t<T;t++)simdgroup_load(mp[t],sp+t*8*C+8*cc,C);"
        "device const half*pv=vp+(t0+8u*cc)*kstr+8*sg;"
        "for(short ii=0;ii<NO;ii++){simdgroup_half8x8 mv;simdgroup_load(mv,pv+8*NSG*ii,kstr);"
        "for(short t=0;t<T;t++)simdgroup_multiply_accumulate(lo[t][ii],mp[t],mv,lo[t][ii]);}}}"
        "threadgroup_barrier(mem_flags::mem_threadgroup);"
        "for(short jj=0;jj<NQ;jj++){const short j=jj*NSG+sg;"
        "if(ln==0)sd[(j/8)*64+(j%8)*9]=1.0f/S[jj];}"
        "threadgroup_barrier(mem_flags::mem_threadgroup);"
        "for(short t=0;t<T;t++){simdgroup_float8x8 md;simdgroup_load(md,sd+t*64,8);"
        "const uint r0=q0+8u*t;device float*yo=y+p.yo+r0*p.qh*D+h*D+8*sg;"
        "for(short ii=0;ii<NO;ii++){simdgroup_float8x8 o;simdgroup_multiply(o,md,lo[t][ii]);"
        "if(r0+8u<=p.rows){simdgroup_store(o,yo+8*NSG*ii,p.qh*D);}"
        "else{simdgroup_store(o,ss+sg*64,8);simdgroup_barrier(mem_flags::mem_threadgroup);"
        "for(short e=ln;e<64;e+=32){if(r0+e/8<p.rows)yo[(e/8)*p.qh*D+8*NSG*ii+e%8]=ss[sg*64+e];}"
        "simdgroup_barrier(mem_flags::mem_threadgroup);}}}}\n"
        "typedef decltype(attention_flash<32,4>) attention_flash_t;\n"
        "template [[host_name(\"attention_flash_d32\")]] kernel attention_flash_t "
        "attention_flash<32,4>;\n"
        "template [[host_name(\"attention_flash_d64\")]] kernel attention_flash_t "
        "attention_flash<64,4>;\n"
        "template [[host_name(\"attention_flash_d96\")]] kernel attention_flash_t "
        "attention_flash<96,4>;\n"
        "template [[host_name(\"attention_flash_d128\")]] kernel attention_flash_t "
        "attention_flash<128,4>;\n"
        "template [[host_name(\"attention_flash_d160\")]] kernel attention_flash_t "
        "attention_flash<160,4>;\n"
        "template [[host_name(\"attention_flash_d192\")]] kernel attention_flash_t "
        "attention_flash<192,4>;\n"
        "template [[host_name(\"attention_flash_d224\")]] kernel attention_flash_t "
        "attention_flash<224,4>;\n"
        "template [[host_name(\"attention_flash_d256\")]] kernel attention_flash_t "
        "attention_flash<256,4>;\n"
        "template [[host_name(\"attention_flash_d320\")]] kernel attention_flash_t "
        "attention_flash<320,8>;\n"
        "template [[host_name(\"attention_flash_d384\")]] kernel attention_flash_t "
        "attention_flash<384,8>;\n"
        "template [[host_name(\"attention_flash_d448\")]] kernel attention_flash_t "
        "attention_flash<448,8>;\n"
        "template [[host_name(\"attention_flash_d512\")]] kernel attention_flash_t "
        "attention_flash<512,8>;\n";

/* Split-KV decode flash for 256 < head_dim <= 512 (the gemma-3n
 * full-attention layers): 16-chunk QK/PV lane unroll (a0..a15),
 * qv[512]/ta[4096]; same split/combine contract as attention_dec_f16. */
static const char metal_attn_dec512_f16_source_a[] =
        "kernel void ATTN(dec512)(device const float*q[[buffer(0)]],device const "
        "KVT*kc[[buffer(1)]],device const KVT*vc[[buffer(2)]],device "
        "float*pb[[buffer(3)]],constant A&p[[buffer(4)]],constant SP&sp[[buffer(5)]],uint2 "
        "tg[[threadgroup_position_in_grid]],uint lid[[thread_index_in_threadgroup]],uint "
        "sg[[simdgroup_index_in_threadgroup]],uint ln[[thread_index_in_simdgroup]]){threadgroup "
        "float qv[512];threadgroup float red[8];threadgroup float tm[8];threadgroup float "
        "tl[8];threadgroup float ta[4096];uint "
        "h=tg.y,ss=tg.x,hd=p.hd;if(h>=p.qh||hd>512u)return;for(uint "
        "i=lid;i<hd;i+=256u){qv[i]=q[p.qo+h*hd+i];}threadgroup_barrier(mem_flags::mem_threadgroup);"
        "uint kvh=h/(p.qh/p.kvh),qp=p.qpos;uint slo=(p.sw>0u&&qp+1u>p.sw)?qp+1u-p.sw:0u;uint "
        "shi=qp<p.kv_len?qp:p.kv_len-1u;float "
        "m=-3.402823466e+38f,l=0.0f,a0=0,a1=0,a2=0,a3=0,a4=0,a5=0,a6=0,a7=0,a8=0,a9=0,a10=0,a11=0,"
        "a12=0,a13=0,a14=0,a15=0;bool "
        "g1=ln+32u<hd,g2=ln+64u<hd,g3=ln+96u<hd,g4=ln+128u<hd,g5=ln+160u<hd,g6=ln+192u<hd,g7=ln+"
        "224u<hd,g8=ln+256u<hd,g9=ln+288u<hd,g10=ln+320u<hd,g11=ln+352u<hd,g12=ln+384u<hd,g13=ln+"
        "416u<hd,g14=ln+448u<hd,g15=ln+480u<hd;for(uint t=slo+ss*8u+sg;t<=shi;t+=sp.ns*8u){uint "
        "kb=p.kco+t*p.kvh*hd+kvh*hd;float "
        "d=qv[ln]*KVF(kc[kb+ln]);if(g1)d+=qv[ln+32u]*KVF(kc[kb+ln+32u]);if(g2)d+=qv[ln+64u]*"
        "KVF(kc[kb+ln+64u]);if(g3)d+=qv[ln+96u]*KVF(kc[kb+ln+96u]);if(g4)d+=qv[ln+128u]*KVF("
        "kc[kb+ln+128u]);if(g5)d+=qv[ln+160u]*KVF(kc[kb+ln+160u]);if(g6)d+=qv[ln+192u]*KVF(kc["
        "kb+ln+192u]);if(g7)d+=qv[ln+224u]*KVF(kc[kb+ln+224u]);if(g8)d+=qv[ln+256u]*KVF(kc[kb+"
        "ln+256u]);if(g9)d+=qv[ln+288u]*KVF(kc[kb+ln+288u]);if(g10)d+=qv[ln+320u]*KVF(kc[kb+ln+"
        "320u]);";
static const char metal_attn_dec512_f16_source_b[] =
        "if(g11)d+=qv[ln+352u]*KVF(kc[kb+ln+352u]);if(g12)d+=qv[ln+384u]*KVF(kc[kb+ln+384u]);"
        "if(g13)d+=qv[ln+416u]*KVF(kc[kb+ln+416u]);if(g14)d+=qv[ln+448u]*KVF(kc[kb+ln+448u]);"
        "if(g15)d+=qv[ln+480u]*KVF(kc[kb+ln+480u]);d=simd_sum(d);float "
        "mn=max(m,d),corr=exp(m-mn),e=exp(d-mn);l=l*corr+e;uint "
        "vb=p.vco+t*p.kvh*hd+kvh*hd;a0=a0*corr+e*KVF(vc[vb+ln]);if(g1)a1=a1*corr+e*KVF(vc[vb+"
        "ln+32u]);if(g2)a2=a2*corr+e*KVF(vc[vb+ln+64u]);if(g3)a3=a3*corr+e*KVF(vc[vb+ln+96u]);"
        "if(g4)a4=a4*corr+e*KVF(vc[vb+ln+128u]);if(g5)a5=a5*corr+e*KVF(vc[vb+ln+160u]);if(g6)"
        "a6=a6*corr+e*KVF(vc[vb+ln+192u]);if(g7)a7=a7*corr+e*KVF(vc[vb+ln+224u]);if(g8)a8=a8*"
        "corr+e*KVF(vc[vb+ln+256u]);if(g9)a9=a9*corr+e*KVF(vc[vb+ln+288u]);if(g10)a10=a10*corr+"
        "e*KVF(vc[vb+ln+320u]);if(g11)a11=a11*corr+e*KVF(vc[vb+ln+352u]);if(g12)a12=a12*corr+e*"
        "KVF(vc[vb+ln+384u]);if(g13)a13=a13*corr+e*KVF(vc[vb+ln+416u]);if(g14)a14=a14*corr+e*"
        "KVF(vc[vb+ln+448u]);if(g15)a15=a15*corr+e*KVF(vc[vb+ln+480u]);m=mn;}if(ln==0u){tm[sg]="
        "m;tl[sg]=l;}uint "
        "tb=sg*hd+ln;ta[tb]=a0;if(g1)ta[tb+32u]=a1;if(g2)ta[tb+64u]=a2;if(g3)ta[tb+96u]=a3;if(g4)"
        "ta[tb+128u]=a4;if(g5)ta[tb+160u]=a5;if(g6)ta[tb+192u]=a6;if(g7)ta[tb+224u]=a7;if(g8)ta[tb+"
        "256u]=a8;if(g9)ta[tb+288u]=a9;if(g10)ta[tb+320u]=a10;if(g11)ta[tb+352u]=a11;if(g12)ta[tb+"
        "384u]=a12;if(g13)ta[tb+416u]=a13;if(g14)ta[tb+448u]=a14;if(g15)ta[tb+480u]=a15;"
        "threadgroup_barrier(mem_flags::mem_threadgroup);float M=tm[0];for(uint "
        "j=1u;j<8u;j++)M=max(M,tm[j]);float L=0.0f;for(uint "
        "j=0u;j<8u;j++)L+=tl[j]*exp(tm[j]-M);uint "
        "pbb=(h*32u+ss)*(hd+2u);if(lid==0u){pb[pbb]=M;pb[pbb+1u]=L;}for(uint "
        "i=lid;i<hd;i+=256u){float acc=0.0f;for(uint "
        "j=0u;j<8u;j++)acc+=ta[j*hd+i]*exp(tm[j]-M);pb[pbb+2u+i]=acc;}}";

/* No-norm head (the engine applies q-norm and rope as
 * separate ops): plain Q load into qv, then the shared split-KV body. */
static const char metal_attn_dec_f16_plain_head[] =
        "kernel void ATTN(dec)(device const float*q[[buffer(0)]],device const "
        "KVT*kc[[buffer(1)]],device const KVT*vc[[buffer(2)]],device "
        "float*pb[[buffer(3)]],constant A&p[[buffer(4)]],constant SP&sp[[buffer(5)]],uint2 "
        "tg[[threadgroup_position_in_grid]],uint lid[[thread_index_in_threadgroup]],uint "
        "sg[[simdgroup_index_in_threadgroup]],uint ln[[thread_index_in_simdgroup]]){"
        "threadgroup float qv[256];threadgroup float red[8];threadgroup float tm[8];threadgroup "
        "float tl[8];threadgroup float ta[2048];"
        "uint h=tg.y,ss=tg.x,hd=p.hd;if(h>=p.qh||hd>256u)return;"
        "for(uint i=lid;i<hd;i+=256u){qv[i]=q[p.qo+h*hd+i];}"
        "threadgroup_barrier(mem_flags::mem_threadgroup);";

/* Pass 2: merge the per-split partials (m, l, unnormalized acc) into y.
 * A threadgroup of 64 threads owns 64 elements of one head; each
 * simdgroup first turns the splits' (m, l) into their weights
 * exp(m_s - M) / L, lane s for split s, then every thread sums its element
 * over the splits, four independent loads per step. One thread looping
 * over the splits per element (16-32 dependent load and exp steps) cost
 * 18-37 us per call on the M1 Max, as much as the attention itself (#17). */
static const char metal_attn_dec_combine_source[] =
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "struct CB{uint qh,hd,ns,yo;};\n"
        "kernel void attention_dec_combine(device const float*pb[[buffer(0)]],device "
        "float*y[[buffer(1)]],constant CB&p[[buffer(2)]],uint2 "
        "tg[[threadgroup_position_in_grid]],ushort lid[[thread_index_in_threadgroup]],ushort "
        "sg[[simdgroup_index_in_threadgroup]],ushort ln[[thread_index_in_simdgroup]]){"
        "threadgroup float sf[64];const uint h=tg.y,hd=p.hd,st=hd+2u;if(h>=p.qh)return;"
        "device const float*b=pb+h*32u*st;"
        "const float m=ln<p.ns?b[ln*st]:-FLT_MAX;const float M=simd_max(m);"
        "const float e=ln<p.ns?exp(m-M):0.0f;"
        "const float L=simd_sum(ln<p.ns?e*b[ln*st+1u]:0.0f);"
        "sf[sg*32u+ln]=e/L;simdgroup_barrier(mem_flags::mem_threadgroup);"
        "const uint i=tg.x*64u+lid;if(i>=hd)return;"
        "threadgroup const float*f=sf+sg*32u;device const float*bi=b+2u+i;"
        "float a0=0.0f,a1=0.0f,a2=0.0f,a3=0.0f;uint s=0u;"
        "for(;s+4u<=p.ns;s+=4u){a0+=bi[s*st]*f[s];a1+=bi[(s+1u)*st]*f[s+1u];"
        "a2+=bi[(s+2u)*st]*f[s+2u];a3+=bi[(s+3u)*st]*f[s+3u];}"
        "for(;s<p.ns;s++)a0+=bi[s*st]*f[s];"
        "y[p.yo+h*hd+i]=(a0+a1)+(a2+a3);}\n";

/* Split-KV decode attention in 32-key blocks (#17), after the structure of
 * llama.cpp's kernel_flash_attn_ext_vec. One simdgroup per threadgroup
 * owns one query head and every ns-th block of its live keys:
 *   - Q.K: the 32 lanes form NE groups of 32/NE; a group takes one key
 *     at a time, each lane D*NE/128 float4 chunks of it, and the 32/NE
 *     keys a group walks per block are independent loads in flight
 *     together, not the one-key-per-iteration chain of the kernels above,
 *     whose memory latency set their time (43/93 us per call at 512 keys,
 *     head_dim 256/512, against 17/32 us for llama.cpp's). NE 4, and 2 at
 *     head_dim 256, measured best on the M1 Max (24-38 us at 512 keys).
 *   - one online-softmax step per block: lane j holds key j's score.
 *   - P.V in the Q.K layout, O in registers, the 4 groups summed by two
 *     shuffles at the end.
 * Q and O in f32, K/V read as KT (F32 rounded to half, as the kernels
 * above read it). Blocks touched are those overlapping the window, so
 * every block holds a live key; keys outside it are masked and their
 * loads clamped into it. Partials and their combine as above. */
static const char metal_attn_decv_source_a[] =
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "struct A{uint rows,kv_len,qh,kvh,hd,qpos,sw,qo,kco,vco,yo;};\n"
        "struct SP{uint ns,fc,yo;};\n"
        "static inline float4 kv4(half4 x){return float4(x);}\n"
        "static inline float4 kv4(float4 x){return float4(half4(x));}\n"
        "template<short D,short NE,typename KT>\n"
        "kernel void attention_decv(device const float*q[[buffer(0)]],device const "
        "KT*kc[[buffer(1)]],device const KT*vc[[buffer(2)]],coherent(device) device "
        "float*pb[[buffer(3)]],"
        "constant A&p[[buffer(4)]],constant SP&sp[[buffer(5)]],device atomic_uint*cnt[[buffer(6)]],"
        "device float*y[[buffer(7)]],uint2 "
        "tg[[threadgroup_position_in_grid]],ushort ln[[thread_index_in_simdgroup]]){"
        "constexpr short NL=32/NE,C=32,D4=D/4,NI=D4/NL;typedef vec<KT,4> KT4;"
        "threadgroup float4 sq[D4];threadgroup float ss[C];"
        "const uint h=tg.y,w=tg.x;if(h>=p.qh)return;"
        "const uint kvh=h/(p.qh/p.kvh),qp=p.qpos;"
        "const uint slo=(p.sw>0u&&qp+1u>p.sw)?qp+1u-p.sw:0u,shi=qp<p.kv_len?qp:p.kv_len-1u;"
        "device const float4*q4=(device const float4*)(q+p.qo+h*D);"
        "for(short i=ln;i<D4;i+=32)sq[i]=q4[i];"
        "simdgroup_barrier(mem_flags::mem_threadgroup);"
        "const short tx=ln%NL,ty=ln/NL;const uint rs=p.kvh*D;"
        "device const KT*kb=kc+p.kco+kvh*D;device const KT*vb=vc+p.vco+kvh*D;"
        "float M=-FLT_MAX/2,S=0.0f;float4 lo[NI];for(short i=0;i<NI;i++)lo[i]=0.0f;"
        "for(uint b=slo/C+w;b<=shi/C;b+=sp.ns){const uint t0=b*C;float mqk[C/NE];"
        "for(short cc=0;cc<C/NE;cc++){const uint t=clamp(t0+uint(cc*NE+ty),slo,shi);"
        "device const KT4*pk=(device const KT4*)(kb+t*rs)+tx;float a=0.0f;"
        "for(short ii=0;ii<NI;ii++)a+=dot(kv4(pk[ii*NL]),sq[ii*NL+tx]);"
        "for(short o=NL/2;o>0;o/=2)a+=simd_shuffle_down(a,ushort(o));"
        "mqk[cc]=simd_shuffle(a,ushort(NL*ty));}"
        "{const uint t=t0+uint(NE*tx+ty);ss[NE*tx+ty]=(t<slo||t>shi)?-FLT_MAX/2:mqk[tx];}"
        "simdgroup_barrier(mem_flags::mem_threadgroup);"
        "const float s=ss[ln],Mn=max(M,simd_max(s)),ms=exp(M-Mn),vs=exp(s-Mn);"
        "S=S*ms+simd_sum(vs);M=Mn;ss[ln]=vs;"
        "simdgroup_barrier(mem_flags::mem_threadgroup);"
        "for(short i=0;i<NI;i++)lo[i]*=ms;"
        "for(short cc=0;cc<C/NE;cc++){const uint t=clamp(t0+uint(cc*NE+ty),slo,shi);"
        "const float wv=ss[cc*NE+ty];device const KT4*pv=(device const KT4*)(vb+t*rs)+tx;"
        "for(short ii=0;ii<NI;ii++)lo[ii]+=kv4(pv[ii*NL])*wv;}"
        "simdgroup_barrier(mem_flags::mem_threadgroup);}"
        "for(short o=16;o>=NL;o/=2)for(short "
        "i=0;i<NI;i++)lo[i]+=simd_shuffle_down(lo[i],ushort(o));"
        "const uint pbb=(h*32u+w)*(D+2u);if(ln==0){pb[pbb]=M;pb[pbb+1u]=S;}"
        "if(ty==0){for(short i=0;i<NI;i++){const uint o=pbb+2u+4u*uint(i*NL+tx);"
        "pb[o]=lo[i].x;pb[o+1u]=lo[i].y;pb[o+2u]=lo[i].z;pb[o+3u]=lo[i].w;}}";
/* Continued: one literal stays within the 4095-char limit. */
static const char metal_attn_decv_combine_source[] =
        /* Fused combine (#196): the head's last split to finish merges the
         * partials as attention_dec_combine does, the same arithmetic in
         * the same order. pb is coherent(device): without it the stores of
         * other threadgroups need not be visible however fenced. Each
         * split's partial stores are ordered before its count by the
         * device-scope fence; the split that brings the
         * count to ns fences again before reading the others' partials and
         * resets the count for the next dispatch, which the serial encoder
         * starts only after this one has finished. */
        "if(sp.fc==0u)return;"
        "atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);"
        "uint done=0u;if(ln==0)done=atomic_fetch_add_explicit(cnt+h,1u,memory_order_relaxed);"
        "done=simd_broadcast(done,ushort(0));if(done+1u!=sp.ns)return;"
        "atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);"
        "if(ln==0)atomic_store_explicit(cnt+h,0u,memory_order_relaxed);"
        "const uint st=D+2u;coherent(device) device const float*b=pb+h*32u*st;"
        "const float m=ln<sp.ns?b[ln*st]:-FLT_MAX;const float Mx=simd_max(m);"
        "const float e=ln<sp.ns?exp(m-Mx):0.0f;"
        "const float L=simd_sum(ln<sp.ns?e*b[ln*st+1u]:0.0f);"
        "ss[ln]=e/L;simdgroup_barrier(mem_flags::mem_threadgroup);"
        /* Lane ln owns elements ln + 32 j. Each element sums its splits in
         * attention_dec_combine's order; the elements are independent, so
         * a step over four splits issues all of the lane's loads at once
         * rather than one element's chain after another's. */
        "constexpr short E=D/32;float a0[E],a1[E],a2[E],a3[E];"
        "for(short j=0;j<E;j++){a0[j]=0.0f;a1[j]=0.0f;a2[j]=0.0f;a3[j]=0.0f;}"
        "coherent(device) device const float*bl=b+2u+ln;uint s=0u;"
        "for(;s+4u<=sp.ns;s+=4u){const float f0=ss[s],f1=ss[s+1u],f2=ss[s+2u],f3=ss[s+3u];"
        "for(short j=0;j<E;j++){const uint i=uint(32*j);a0[j]+=bl[s*st+i]*f0;"
        "a1[j]+=bl[(s+1u)*st+i]*f1;a2[j]+=bl[(s+2u)*st+i]*f2;a3[j]+=bl[(s+3u)*st+i]*f3;}}"
        "for(;s<sp.ns;s++){const float f0=ss[s];"
        "for(short j=0;j<E;j++)a0[j]+=bl[s*st+uint(32*j)]*f0;}"
        "for(short j=0;j<E;j++)y[sp.yo+h*D+ln+uint(32*j)]=(a0[j]+a1[j])+(a2[j]+a3[j]);}\n"
        "typedef decltype(attention_decv<64,4,half>) attention_decv_t;\n"
        "typedef decltype(attention_decv<64,4,float>) attention_decv32_t;\n"
        "#define DV(D,NE) template [[host_name(\"attention_decv_d\" #D \"_f16\")]] kernel "
        "attention_decv_t attention_decv<D,NE,half>;template [[host_name(\"attention_decv_d\" #D "
        "\"_f32kv\")]] kernel attention_decv32_t attention_decv<D,NE,float>;\n"
        "DV(64,4)\nDV(128,4)\nDV(256,2)\nDV(512,4)\n";

/* Correctness-first Gated-DeltaNet mixer kernel. One workgroup owns one
 * head and advances its convolution channels and delta state serially over
 * the sequence; 256 lanes parallelize the head dimensions. */
static const char metal_deltanet_source[] =
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "struct P{uint seq,nkh,nvh,dk,dv,K,qo,zo,bo,ao,cwo,awo,dto,no,cso,so;float eps;};\n"
        "inline float silu1(float x){float e=exp(-fabs(x));return "
        "x>=0.0f?x/(1.0f+e):x*e/(1.0f+e);}\n"
        "kernel void deltanet_mix(device float*qkv[[buffer(0)]],device float*z[[buffer(1)]],"
        "device const float*beta[[buffer(2)]],device const float*alpha[[buffer(3)]],"
        "device const float*cw[[buffer(4)]],device const float*aw[[buffer(5)]],"
        "device const float*dt[[buffer(6)]],device const float*nw[[buffer(7)]],"
        "device float*cs[[buffer(8)]],device float*S[[buffer(9)]],constant P&p[[buffer(10)]],"
        "uint hk[[threadgroup_position_in_grid]],uint j[[thread_index_in_threadgroup]]){"
        /* tq/tk stage the normed q/k rows in threadgroup memory: the state
         * loop reads them cross-thread, and mem_threadgroup barriers only
         * fence threadgroup-address-space data, so device reads here would
         * be unfenced. */
        "threadgroup float red[256];threadgroup float tq[256];threadgroup float tk[256];"
        "uint keyd=p.nkh*p.dk,vd=p.nvh*p.dv,cd=2u*keyd+vd;"
        "if(hk>=p.nkh)return;float qs=rsqrt(float(p.dk));"
        "for(uint t=0;t<p.seq;t++){uint qb=p.qo+t*cd,kb=qb+keyd,vb=kb+keyd;"
        "float qr=0.0f,kr=0.0f,qc=0.0f,kc=0.0f;"
        "if(j<p.dk){uint cq=hk*p.dk+j,ck=keyd+cq;qr=qkv[qb+cq];kr=qkv[qb+ck];"
        "for(uint r=0;r<p.K;r++){float qx=r+1u<p.K?cs[p.cso+r*cd+cq]:qr;"
        "float kx=r+1u<p.K?cs[p.cso+r*cd+ck]:kr;qc+=cw[p.cwo+cq*p.K+r]*qx;"
        "kc+=cw[p.cwo+ck*p.K+r]*kx;}for(uint r=0;r+2u<p.K;r++){"
        "cs[p.cso+r*cd+cq]=cs[p.cso+(r+1u)*cd+cq];"
        "cs[p.cso+r*cd+ck]=cs[p.cso+(r+1u)*cd+ck];}"
        "if(p.K>=2u){cs[p.cso+(p.K-2u)*cd+cq]=qr;cs[p.cso+(p.K-2u)*cd+ck]=kr;}"
        "qkv[qb+cq]=silu1(qc);qkv[qb+ck]=silu1(kc);}"
        "threadgroup_barrier(mem_flags::mem_threadgroup);"
        "red[j]=j<p.dk?qkv[qb+hk*p.dk+j]*qkv[qb+hk*p.dk+j]:0.0f;"
        "threadgroup_barrier(mem_flags::mem_threadgroup);for(uint s=128u;s>0u;s>>=1u){"
        "if(j<s)red[j]+=red[j+s];threadgroup_barrier(mem_flags::mem_threadgroup);}"
        "float qi=rsqrt(red[0]+p.eps);if(j<p.dk){float v=qkv[qb+hk*p.dk+j]*qi*qs;"
        "qkv[qb+hk*p.dk+j]=v;tq[j]=v;}"
        "threadgroup_barrier(mem_flags::mem_threadgroup);"
        "red[j]=j<p.dk?qkv[kb+hk*p.dk+j]*qkv[kb+hk*p.dk+j]:0.0f;"
        "threadgroup_barrier(mem_flags::mem_threadgroup);for(uint s=128u;s>0u;s>>=1u){"
        "if(j<s)red[j]+=red[j+s];threadgroup_barrier(mem_flags::mem_threadgroup);}"
        "float ki=rsqrt(red[0]+p.eps);if(j<p.dk){float v=qkv[kb+hk*p.dk+j]*ki;"
        "qkv[kb+hk*p.dk+j]=v;tk[j]=v;}"
        "threadgroup_barrier(mem_flags::mem_threadgroup);for(uint h=hk;h<p.nvh;h+=p.nkh){"
        "float vr=0.0f,vc=0.0f;if(j<p.dv){uint cv=2u*keyd+h*p.dv+j;vr=qkv[qb+cv];"
        "for(uint r=0;r<p.K;r++){float vx=r+1u<p.K?cs[p.cso+r*cd+cv]:vr;"
        "vc+=cw[p.cwo+cv*p.K+r]*vx;}for(uint r=0;r+2u<p.K;r++)"
        "cs[p.cso+r*cd+cv]=cs[p.cso+(r+1u)*cd+cv];"
        "if(p.K>=2u)cs[p.cso+(p.K-2u)*cd+cv]=vr;qkv[qb+cv]=silu1(vc);}"
        "threadgroup_barrier(mem_flags::mem_threadgroup);float out=0.0f,gate=0.0f;"
        "if(j<p.dv){float be=1.0f/(1.0f+exp(-beta[p.bo+t*p.nvh+h]));"
        "float sp=log(1.0f+exp(alpha[p.ao+t*p.nvh+h]+dt[p.dto+h]));"
        "float decay=exp(aw[p.awo+h]*sp),mem=0.0f;uint sb=p.so+h*p.dk*p.dv;"
        /* decay folds into both passes: one read-only mem pass, one fused
         * decay+update pass — 2R+1W over S instead of 2R+2W. */
        "for(uint i=0;i<p.dk;i++){mem+=S[sb+i*p.dv+j]*tk[i];}mem*=decay;"
        "float d=(qkv[vb+h*p.dv+j]-mem)*be;"
        "for(uint i=0;i<p.dk;i++){uint si=sb+i*p.dv+j;float sv=S[si]*decay+tk[i]*d;"
        "S[si]=sv;out+=sv*tq[i];}gate=z[p.zo+t*vd+h*p.dv+j];}"
        "red[j]=j<p.dv?out*out:0.0f;threadgroup_barrier(mem_flags::mem_threadgroup);"
        "for(uint s=128u;s>0u;s>>=1u){if(j<s)red[j]+=red[j+s];"
        "threadgroup_barrier(mem_flags::mem_threadgroup);}if(j<p.dv){"
        "float inv=rsqrt(red[0]/float(p.dv)+p.eps);z[p.zo+t*vd+h*p.dv+j]="
        "out*inv*nw[p.no+j]*silu1(gate);}threadgroup_barrier(mem_flags::mem_threadgroup);}}}\n";

/* Single-token (decode) DeltaNet, split so the state update spreads over
 * the GPU instead of 16 serial threadgroups. dn_dec_qk: one threadgroup per
 * k-head runs the q/k conv + silu, rolls their conv state and L2-normalizes
 * q (scaled) and k in place — the first half of deltanet_mix at seq 1.
 * dn_dec_v: one threadgroup per v-head, dv columns x 4 row groups
 * (ceil(dk/4) rows each, the last one short); conv +
 * silu of v, then S is read once for mem, updated and read for out with the
 * dk rows split four ways and the partial sums reduced in threadgroup
 * memory, then the gated RMS norm. Same math as deltanet_mix. */
static const char metal_dn_decode_source[] =
        "kernel void dn_dec_qk(device float*qkv[[buffer(0)]],device float*z[[buffer(1)]],"
        "device const float*beta[[buffer(2)]],device const float*alpha[[buffer(3)]],"
        "device const float*cw[[buffer(4)]],device const float*aw[[buffer(5)]],"
        "device const float*dt[[buffer(6)]],device const float*nw[[buffer(7)]],"
        "device float*cs[[buffer(8)]],device float*S[[buffer(9)]],constant P&p[[buffer(10)]],"
        "uint hk[[threadgroup_position_in_grid]],uint j[[thread_index_in_threadgroup]]){"
        "threadgroup float red[256];uint keyd=p.nkh*p.dk,vd=p.nvh*p.dv,cd=2u*keyd+vd;"
        "if(hk>=p.nkh)return;float qs=rsqrt(float(p.dk));uint qb=p.qo,kb=qb+keyd;"
        "float qr=0.0f,kr=0.0f,qc=0.0f,kc=0.0f;"
        "if(j<p.dk){uint cq=hk*p.dk+j,ck=keyd+cq;qr=qkv[qb+cq];kr=qkv[qb+ck];"
        "for(uint r=0;r<p.K;r++){float qx=r+1u<p.K?cs[p.cso+r*cd+cq]:qr;"
        "float kx=r+1u<p.K?cs[p.cso+r*cd+ck]:kr;qc+=cw[p.cwo+cq*p.K+r]*qx;"
        "kc+=cw[p.cwo+ck*p.K+r]*kx;}for(uint r=0;r+2u<p.K;r++){"
        "cs[p.cso+r*cd+cq]=cs[p.cso+(r+1u)*cd+cq];cs[p.cso+r*cd+ck]=cs[p.cso+(r+1u)*cd+ck];}"
        "if(p.K>=2u){cs[p.cso+(p.K-2u)*cd+cq]=qr;cs[p.cso+(p.K-2u)*cd+ck]=kr;}"
        "qc=silu1(qc);kc=silu1(kc);}"
        "red[j]=j<p.dk?qc*qc:0.0f;threadgroup_barrier(mem_flags::mem_threadgroup);"
        "for(uint s=128u;s>0u;s>>=1u){if(j<s)red[j]+=red[j+s];"
        "threadgroup_barrier(mem_flags::mem_threadgroup);}float qi=rsqrt(red[0]+p.eps);"
        "threadgroup_barrier(mem_flags::mem_threadgroup);"
        "red[j]=j<p.dk?kc*kc:0.0f;threadgroup_barrier(mem_flags::mem_threadgroup);"
        "for(uint s=128u;s>0u;s>>=1u){if(j<s)red[j]+=red[j+s];"
        "threadgroup_barrier(mem_flags::mem_threadgroup);}float ki=rsqrt(red[0]+p.eps);"
        "if(j<p.dk){qkv[qb+hk*p.dk+j]=qc*qi*qs;qkv[kb+hk*p.dk+j]=kc*ki;}}\n"
        "kernel void dn_dec_v(device float*qkv[[buffer(0)]],device float*z[[buffer(1)]],"
        "device const float*beta[[buffer(2)]],device const float*alpha[[buffer(3)]],"
        "device const float*cw[[buffer(4)]],device const float*aw[[buffer(5)]],"
        "device const float*dt[[buffer(6)]],device const float*nw[[buffer(7)]],"
        "device float*cs[[buffer(8)]],device float*S[[buffer(9)]],constant P&p[[buffer(10)]],"
        "uint h[[threadgroup_position_in_grid]],uint lid[[thread_index_in_threadgroup]]){"
        "threadgroup float tq[256];threadgroup float tk[256];threadgroup float vv[256];"
        "threadgroup float part[1024];"
        "uint keyd=p.nkh*p.dk,vd=p.nvh*p.dv,cd=2u*keyd+vd;if(h>=p.nvh)return;"
        "uint hk=h%p.nkh,j=lid%p.dv,g=lid/p.dv,nt=4u*p.dv;uint qb=p.qo,kb=qb+keyd;"
        "for(uint i=lid;i<p.dk;i+=nt){tq[i]=qkv[qb+hk*p.dk+i];tk[i]=qkv[kb+hk*p.dk+i];}"
        "if(g==0u){uint cv=2u*keyd+h*p.dv+j;float vr=qkv[qb+cv],vc=0.0f;"
        "for(uint r=0;r<p.K;r++){float vx=r+1u<p.K?cs[p.cso+r*cd+cv]:vr;"
        "vc+=cw[p.cwo+cv*p.K+r]*vx;}for(uint r=0;r+2u<p.K;r++)"
        "cs[p.cso+r*cd+cv]=cs[p.cso+(r+1u)*cd+cv];"
        "if(p.K>=2u)cs[p.cso+(p.K-2u)*cd+cv]=vr;vc=silu1(vc);qkv[qb+cv]=vc;vv[j]=vc;}"
        "threadgroup_barrier(mem_flags::mem_threadgroup);"
        "float be=1.0f/(1.0f+exp(-beta[p.bo+h]));"
        "float sp=log(1.0f+exp(alpha[p.ao+h]+dt[p.dto+h]));float decay=exp(aw[p.awo+h]*sp);"
        "uint rq=(p.dk+3u)/4u,i0=g*rq,i1=min(i0+rq,p.dk),sb=p.so+h*p.dk*p.dv;float mp=0.0f;"
        "for(uint i=i0;i<i1;i++)mp+=S[sb+i*p.dv+j]*tk[i];"
        "part[g*p.dv+j]=mp;threadgroup_barrier(mem_flags::mem_threadgroup);"
        "float mem=(part[j]+part[p.dv+j]+part[2u*p.dv+j]+part[3u*p.dv+j])*decay;"
        "float d=(vv[j]-mem)*be;float op=0.0f;"
        "for(uint i=i0;i<i1;i++){uint si=sb+i*p.dv+j;float sv=S[si]*decay+tk[i]*d;"
        "S[si]=sv;op+=sv*tq[i];}"
        "threadgroup_barrier(mem_flags::mem_threadgroup);part[g*p.dv+j]=op;"
        "threadgroup_barrier(mem_flags::mem_threadgroup);"
        "float out=part[j]+part[p.dv+j]+part[2u*p.dv+j]+part[3u*p.dv+j];"
        "threadgroup_barrier(mem_flags::mem_threadgroup);"
        "if(g==0u)part[j]=out*out;threadgroup_barrier(mem_flags::mem_threadgroup);"
        "if(g==0u){float ss=0.0f;for(uint c=0u;c<p.dv;c++)ss+=part[c];"
        "float inv=rsqrt(ss/float(p.dv)+p.eps);float gate=z[p.zo+h*p.dv+j];"
        "z[p.zo+h*p.dv+j]=out*inv*nw[p.no+j]*silu1(gate);}}\n";

/* Chunked DeltaNet prefill on simdgroup matrices (#99): the chunk recipe of
 * transformer_dn_head_chunk / dn_run_prefill_chunked (64-token chunks, F32
 * state and accumulation), arranged so that only the state recurrence runs
 * in order and every product is an 8x8 simdgroup MMA. For conv_kernel 4,
 * head_k a multiple of 32 up to 128 and head_v a multiple of 32 up to 256;
 * other shapes take the serial deltanet_mix. Seven dispatches per call of up
 * to 512 tokens:
 *   dnf_cst_copy  snapshot of the conv state (dnf_roll reads it).
 *   dnf_conv      conv + SiLU, one thread per channel over 16 tokens with
 *                 the window in registers. Rows past seq, up to the 64-token
 *                 padding, are written as zeros, so a short last chunk needs
 *                 no masking downstream.
 *   dnf_norm      q/k L2 norms (q also scaled by head_k^-1/2) and the
 *                 beta / log-decay scalars, one threadgroup per token.
 *   dnf_roll      the conv state advanced by seq rows.
 *   dnf_chunk     per (chunk, v-head): decay cumsum gamma; the strictly
 *                 lower A = -beta_t e^(gamma_t - gamma_s) k_t.k_s;
 *                 T = (I - A)^-1 as 8x8 diagonal blocks by substitution and
 *                 the off-diagonal blocks row block by row block,
 *                 T_IJ = T_II sum_K A_IK T_KJ; then U = T diag(beta) V and
 *                 Wn = -T diag(beta e^gamma) K.
 *   dnf_state     per (16 value columns, v-head), over the chunks in order
 *                 with the state slice in threadgroup memory: d = U + Wn S
 *                 (written over U), Q S (the inter-chunk output), then
 *                 S = e^gamma_last S + K^T diag(e^(gamma_last - gamma)) d.
 *   dnf_out       per (chunk, v-head): o = e^gamma Q S + (Q K^T, causal,
 *                 decayed) d, then the gated RMS norm into z.
 * y holds q, k and v head-major (q heads, then k heads: [head][L][head_k];
 * v: [head][L][head_v]) so an 8-row tile is 8 rows of one head. Scratch
 * (floats, L = the call's tokens padded to 64): y L*cd | beta L*nvh |
 * log-decay L*nvh | old conv state 3*cd | gamma nvh*L | Wn nvh*L*dk |
 * U/d nvh*L*dv | o nvh*L*dv. Concatenated after metal_deltanet_source
 * (struct P, silu1); every kernel takes the serial kernel's ten buffers plus
 * scratch at buffer(10) and the params at buffer(11). DNF_DVB is the
 * host's dnf_state column block. */
#define GEIST_DN_SIG(NAME)                                                            \
    "kernel void " NAME "(device float*qkv[[buffer(0)]],device float*z[[buffer(1)]]," \
    "device const float*beta[[buffer(2)]],device const float*alpha[[buffer(3)]],"     \
    "device const float*cw[[buffer(4)]],device const float*aw[[buffer(5)]],"          \
    "device const float*dt[[buffer(6)]],device const float*nw[[buffer(7)]],"          \
    "device float*cs[[buffer(8)]],device float*Sb[[buffer(9)]],"                      \
    "device float*scr[[buffer(10)]],constant P&p[[buffer(11)]],"

static const char metal_dnf_prep_source[] =
        "struct DF{uint lp,cd,keyd,vd,y,bt,gg,oc,gam,wn,u,o;};"
        "static inline DF dfl(constant P&p){DF l;l.lp=(p.seq+63u)&~63u;l.keyd=p.nkh*p.dk;"
        "l.vd=p.nvh*p.dv;l.cd=2u*l.keyd+l.vd;l.y=0u;l.bt=l.lp*l.cd;l.gg=l.bt+l.lp*p.nvh;"
        "l.oc=l.gg+l.lp*p.nvh;l.gam=l.oc+(p.K-1u)*l.cd;l.wn=l.gam+p.nvh*l.lp;"
        "l.u=l.wn+p.nvh*l.lp*p.dk;l.o=l.u+p.nvh*l.lp*p.dv;return l;}" GEIST_DN_SIG(
                "dnf_cst_copy") "uint gid[[thread_position_in_grid]]){DF l=dfl(p);"
                                "if(gid<(p.K-1u)*l.cd)scr[l.oc+gid]=cs[p.cso+gid];}" GEIST_DN_SIG(
                                        "dnf_conv") "uint2 gid[[thread_position_in_grid]]){DF "
                                                    "l=dfl(p);uint c=gid.x,tb=gid.y*16u;"
                                                    "if(c>=l.cd||tb>=l.lp)return;device const "
                                                    "float*w=cw+p.cwo+c*4u;"
                                                    "float w0=w[0],w1=w[1],w2=w[2],w3=w[3];float "
                                                    "x0,x1,x2;{float x[3];"
                                                    "for(uint j=0u;j<3u;j++){int "
                                                    "src=int(tb+j)-3;x[j]=src<0?scr[l.oc+(tb+j)*l."
                                                    "cd+c]:"
                                                    "(uint(src)<p.seq?qkv[p.qo+uint(src)*l.cd+c]:0."
                                                    "0f);}x0=x[0];x1=x[1];x2=x[2];}"
                                                    "uint kd2=2u*l.keyd,st=c<kd2?p.dk:p.dv;device "
                                                    "float*yo=c<kd2?"
                                                    "scr+l.y+((c/p.dk)*l.lp+tb)*p.dk+c%p.dk:"
                                                    "scr+l.y+kd2*l.lp+(((c-kd2)/"
                                                    "p.dv)*l.lp+tb)*p.dv+(c-kd2)%p.dv;"
                                                    "for(uint u=0u;u<16u;u++){uint t=tb+u;float "
                                                    "xn=t<p.seq?qkv[p.qo+t*l.cd+c]:0.0f;"
                                                    "float "
                                                    "a=0.0f;a+=w0*x0;a+=w1*x1;a+=w2*x2;a+=w3*xn;yo["
                                                    "u*st]=t<p.seq?silu1(a):0.0f;"
                                                    "x0=x1;x1=x2;x2=xn;}}" GEIST_DN_SIG(
                                                            "dnf_norm") "uint "
                                                                        "t[[threadgroup_position_"
                                                                        "in_grid]],ushort "
                                                                        "ti[[thread_index_in_"
                                                                        "threadgroup]],"
                                                                        "ushort "
                                                                        "sg[[simdgroup_index_in_"
                                                                        "threadgroup]],ushort "
                                                                        "ln[[thread_index_in_"
                                                                        "simdgroup]]){"
                                                                        "DF "
                                                                        "l=dfl(p);if(t>=l.lp)"
                                                                        "return;uint "
                                                                        "nq=p.dk/32u;float "
                                                                        "qs=rsqrt(float(p.dk));"
                                                                        "for(uint "
                                                                        "hh=sg;hh<2u*p.nkh;hh+=8u){"
                                                                        "device "
                                                                        "float*yv=scr+l.y+(hh*l.lp+"
                                                                        "t)*p.dk;"
                                                                        "float "
                                                                        "v[4]={0.0f,0.0f,0.0f,0.0f}"
                                                                        ";float ss=0.0f;"
                                                                        "for(uint "
                                                                        "u=0u;u<nq;u++){v[u]=yv[u*"
                                                                        "32u+ln];ss+=v[u]*v[u];}ss="
                                                                        "simd_sum(ss);"
                                                                        "float "
                                                                        "sc=rsqrt(ss+p.eps)*(hh<p."
                                                                        "nkh?qs:1.0f);for(uint "
                                                                        "u=0u;u<nq;u++)yv[u*32u+ln]"
                                                                        "=v[u]*sc;}"
                                                                        "for(uint "
                                                                        "h=ti;h<p.nvh;h+=256u){"
                                                                        "float "
                                                                        "b=0.0f,g=0.0f;if(t<p.seq){"
                                                                        "b=1.0f/"
                                                                        "(1.0f+exp(-beta[p.bo+t*p."
                                                                        "nvh+h]));"
                                                                        "g=aw[p.awo+h]*log(1.0f+"
                                                                        "exp(alpha[p.ao+t*p.nvh+h]+"
                                                                        "dt[p.dto+h]));}"
                                                                        "scr[l.bt+t*p.nvh+h]=b;scr["
                                                                        "l.gg+t*p.nvh+h]=g;}"
                                                                        "}" GEIST_DN_SIG(
                                                                                "dnf_roll") "uint "
                                                                                            "gid[["
                                                                                            "thread"
                                                                                            "_posit"
                                                                                            "ion_"
                                                                                            "in_"
                                                                                            "grid]]"
                                                                                            "){DF "
                                                                                            "l=dfl("
                                                                                            "p);"
                                                                                            "uint "
                                                                                            "hist="
                                                                                            "p.K-"
                                                                                            "1u;"
                                                                                            "if("
                                                                                            "gid>="
                                                                                            "hist*"
                                                                                            "l.cd)"
                                                                                            "return"
                                                                                            ";uint "
                                                                                            "j=gid/"
                                                                                            "l.cd,"
                                                                                            "c=gid%"
                                                                                            "l.cd;"
                                                                                            "int "
                                                                                            "src="
                                                                                            "int(p."
                                                                                            "seq+j)"
                                                                                            "-int("
                                                                                            "hist);"
                                                                                            "cs[p."
                                                                                            "cso+j*"
                                                                                            "l.cd+"
                                                                                            "c]="
                                                                                            "src>="
                                                                                            "0?qkv["
                                                                                            "p.qo+"
                                                                                            "uint("
                                                                                            "src)*"
                                                                                            "l.cd+"
                                                                                            "c]:"
                                                                                            "scr[l."
                                                                                            "oc+(p."
                                                                                            "seq+j)"
                                                                                            "*l.cd+"
                                                                                            "c];}"
                                                                                            "\n";

static const char metal_dnf_chunk_source[] =
        "\n#define DNF_UNROLL _Pragma(\"clang loop unroll(full)\")\n"
        "typedef simdgroup_float8x8 M8;" GEIST_DN_SIG(
                "dnf_chunk") "uint2 tg[[threadgroup_position_in_grid]],ushort "
                             "ti[[thread_index_in_threadgroup]],"
                             "ushort sg[[simdgroup_index_in_threadgroup]]){"
                             "threadgroup float Am[4096];threadgroup float gm[64];threadgroup "
                             "float bs[64];"
                             "DF l=dfl(p);uint c=tg.x,h=tg.y;if(h>=p.nvh||c*64u>=l.lp)return;"
                             "uint hk=h%p.nkh,t0=c*64u,dk=p.dk,dv=p.dv;"
                             "if(ti<64u){gm[ti]=scr[l.gg+(t0+ti)*p.nvh+h];bs[ti]=scr[l.bt+(t0+ti)*"
                             "p.nvh+h];}"
                             "threadgroup_barrier(mem_flags::mem_threadgroup);"
                             "if(ti==0u){float a=0.0f;for(uint t=0u;t<64u;t++){a+=gm[t];gm[t]=a;}}"
                             "device const float*Kp=scr+l.y+((p.nkh+hk)*l.lp+t0)*dk;"
                             "device const float*Vp=scr+l.y+2u*l.keyd*l.lp+(h*l.lp+t0)*dv;"
                             "for(ushort ii=0;ii<2;ii++){uint i=ii==0?sg:7u-sg;for(uint "
                             "j=0u;j<=i;j++){"
                             "M8 acc=make_filled_simdgroup_matrix<float,8>(0.0f);for(uint "
                             "x=0u;x<dk;x+=8u){M8 a,b;"
                             "simdgroup_load(a,Kp+i*8u*dk+x,dk);simdgroup_load(b,Kp+j*8u*dk+x,dk,0,"
                             "true);"
                             "simdgroup_multiply_accumulate(acc,a,b,acc);}simdgroup_store(acc,Am+i*"
                             "512u+j*8u,64);}}"
                             "threadgroup_barrier(mem_flags::mem_threadgroup);"
                             "if(ti<64u)scr[l.gam+h*l.lp+t0+ti]=gm[ti];"
                             "for(uint e=ti;e<4096u;e+=128u){uint t=e>>6,s=e&63u;"
                             "Am[e]=s<t?-bs[t]*exp(gm[t]-gm[s])*Am[e]:0.0f;}"
                             "threadgroup_barrier(mem_flags::mem_threadgroup);"
                             "{uint o=((ti>>3)&7u)*520u,cc=ti&7u;float x[8];"
                             "DNF_UNROLL for(uint r=0u;r<8u;r++){float a=r==cc?1.0f:0.0f;"
                             "DNF_UNROLL for(uint "
                             "s=0u;s<8u;s++){if(s<r&&s>=cc)a+=Am[o+r*64u+s]*x[s];}x[r]=r<cc?0.0f:a;"
                             "}"
                             "threadgroup_barrier(mem_flags::mem_threadgroup);"
                             "if(ti<64u){DNF_UNROLL for(uint r=0u;r<8u;r++)Am[o+r*64u+cc]=x[r];}}"
                             "threadgroup_barrier(mem_flags::mem_threadgroup);"
                             "for(uint bi=1u;bi<8u;bi++){M8 R[2];"
                             "for(uint u=0u;u<2u;u++){uint "
                             "bj=sg+4u*u;if(bj<bi){R[u]=make_filled_simdgroup_matrix<float,8>(0.0f)"
                             ";"
                             "for(uint bk=bj;bk<bi;bk++){M8 "
                             "a,b;simdgroup_load(a,Am+bi*512u+bk*8u,64);"
                             "simdgroup_load(b,Am+bk*512u+bj*8u,64);simdgroup_multiply_accumulate("
                             "R[u],a,b,R[u]);}}}"
                             "threadgroup_barrier(mem_flags::mem_threadgroup);M8 "
                             "dg;simdgroup_load(dg,Am+bi*520u,64);"
                             "for(uint u=0u;u<2u;u++){uint bj=sg+4u*u;if(bj<bi){M8 "
                             "t;simdgroup_multiply(t,dg,R[u]);"
                             "simdgroup_store(t,Am+bi*512u+bj*8u,64);}}threadgroup_barrier(mem_"
                             "flags::mem_threadgroup);}"
                             "for(uint e=ti;e<4096u;e+=128u)Am[e]*=bs[e&63u];"
                             "threadgroup_barrier(mem_flags::mem_threadgroup);"
                             "device float*Up=scr+l.u+(h*l.lp+t0)*dv;"
                             "for(ushort ii=0;ii<2;ii++){uint i=ii==0?sg:7u-sg;for(uint "
                             "cg=0u;cg<dv;cg+=32u){M8 acc[4];"
                             "for(ushort "
                             "q=0;q<4;q++)acc[q]=make_filled_simdgroup_matrix<float,8>(0.0f);"
                             "for(uint k=0u;k<=i;k++){M8 "
                             "a;simdgroup_load(a,Am+i*512u+k*8u,64);for(ushort q=0;q<4;q++){"
                             "M8 "
                             "b;simdgroup_load(b,Vp+k*8u*dv+cg+q*8u,dv);simdgroup_multiply_"
                             "accumulate(acc[q],a,b,acc[q]);}}"
                             "for(ushort "
                             "q=0;q<4;q++)simdgroup_store(acc[q],Up+i*8u*dv+cg+q*8u,dv);}}"
                             "threadgroup_barrier(mem_flags::mem_threadgroup);"
                             "for(uint e=ti;e<4096u;e+=128u)Am[e]*=-exp(gm[e&63u]);"
                             "threadgroup_barrier(mem_flags::mem_threadgroup);"
                             "device float*Wp=scr+l.wn+(h*l.lp+t0)*dk;"
                             "for(ushort ii=0;ii<2;ii++){uint i=ii==0?sg:7u-sg;for(uint "
                             "cg=0u;cg<dk;cg+=32u){M8 acc[4];"
                             "for(ushort "
                             "q=0;q<4;q++)acc[q]=make_filled_simdgroup_matrix<float,8>(0.0f);"
                             "for(uint k=0u;k<=i;k++){M8 "
                             "a;simdgroup_load(a,Am+i*512u+k*8u,64);for(ushort q=0;q<4;q++){"
                             "M8 "
                             "b;simdgroup_load(b,Kp+k*8u*dk+cg+q*8u,dk);simdgroup_multiply_"
                             "accumulate(acc[q],a,b,acc[q]);}}"
                             "for(ushort "
                             "q=0;q<4;q++)simdgroup_store(acc[q],Wp+i*8u*dk+cg+q*8u,dk);}}}\n";

static const char metal_dnf_state_source[] = "\n#define DNF_DVB 16\n" GEIST_DN_SIG(
        "dnf_state") "uint2 tg[[threadgroup_position_in_grid]],ushort "
                     "ti[[thread_index_in_threadgroup]],"
                     "ushort sg[[simdgroup_index_in_threadgroup]]){"
                     "threadgroup float Ss[128*DNF_DVB];threadgroup float "
                     "Dt[64*DNF_DVB];threadgroup float gm[64];"
                     "DF l=dfl(p);uint jb=tg.x*DNF_DVB,h=tg.y;if(h>=p.nvh||jb>=p.dv)return;"
                     "uint hk=h%p.nkh,dk=p.dk,dv=p.dv;device float*Sg=Sb+p.so+h*dk*dv+jb;"
                     "for(uint e=ti;e<dk*DNF_DVB;e+=128u)Ss[e]=Sg[(e/DNF_DVB)*dv+e%DNF_DVB];"
                     "for(uint t0=0u;t0<l.lp;t0+=64u){if(ti<64u)gm[ti]=scr[l.gam+h*l.lp+t0+ti];"
                     "threadgroup_barrier(mem_flags::mem_threadgroup);"
                     "device float*Up=scr+l.u+(h*l.lp+t0+sg*16u)*dv+jb;"
                     "device const float*Wp=scr+l.wn+(h*l.lp+t0+sg*16u)*dk;"
                     "device const float*Qp=scr+l.y+(hk*l.lp+t0+sg*16u)*dk;"
                     "device float*Op=scr+l.o+(h*l.lp+t0+sg*16u)*dv+jb;M8 acc[DNF_DVB/4];"
                     "for(ushort r=0;r<2;r++)for(ushort q=0;q<DNF_DVB/8;q++)"
                     "simdgroup_load(acc[r*(DNF_DVB/8)+q],Up+r*8u*dv+q*8u,dv);"
                     "for(uint x=0u;x<dk;x+=8u){M8 "
                     "a0,a1;simdgroup_load(a0,Wp+x,dk);simdgroup_load(a1,Wp+8u*dk+x,dk);"
                     "for(ushort q=0;q<DNF_DVB/8;q++){M8 "
                     "b;simdgroup_load(b,Ss+x*DNF_DVB+q*8u,DNF_DVB);"
                     "simdgroup_multiply_accumulate(acc[q],a0,b,acc[q]);"
                     "simdgroup_multiply_accumulate(acc[DNF_DVB/8+q],a1,b,acc[DNF_DVB/8+q]);}}"
                     "for(ushort r=0;r<2;r++)for(ushort q=0;q<DNF_DVB/8;q++){"
                     "simdgroup_store(acc[r*(DNF_DVB/8)+q],Up+r*8u*dv+q*8u,dv);"
                     "simdgroup_store(acc[r*(DNF_DVB/8)+q],Dt+(sg*16u+r*8u)*DNF_DVB+q*8u,DNF_DVB);}"
                     "for(ushort "
                     "q=0;q<DNF_DVB/4;q++)acc[q]=make_filled_simdgroup_matrix<float,8>(0.0f);"
                     "for(uint x=0u;x<dk;x+=8u){M8 "
                     "a0,a1;simdgroup_load(a0,Qp+x,dk);simdgroup_load(a1,Qp+8u*dk+x,dk);"
                     "for(ushort q=0;q<DNF_DVB/8;q++){M8 "
                     "b;simdgroup_load(b,Ss+x*DNF_DVB+q*8u,DNF_DVB);"
                     "simdgroup_multiply_accumulate(acc[q],a0,b,acc[q]);"
                     "simdgroup_multiply_accumulate(acc[DNF_DVB/8+q],a1,b,acc[DNF_DVB/8+q]);}}"
                     "for(ushort r=0;r<2;r++)for(ushort q=0;q<DNF_DVB/8;q++)"
                     "simdgroup_store(acc[r*(DNF_DVB/8)+q],Op+r*8u*dv+q*8u,dv);"
                     "threadgroup_barrier(mem_flags::mem_threadgroup);float gl=gm[63],eg=exp(gl);"
                     "for(uint e=ti;e<64u*DNF_DVB;e+=128u)Dt[e]*=exp(gl-gm[e/DNF_DVB]);"
                     "for(uint e=ti;e<dk*DNF_DVB;e+=128u)Ss[e]*=eg;"
                     "threadgroup_barrier(mem_flags::mem_threadgroup);"
                     "device const float*Kp=scr+l.y+((p.nkh+hk)*l.lp+t0)*dk;"
                     "for(uint rt=sg;rt<dk/8u;rt+=4u){M8 sa[DNF_DVB/8];"
                     "for(ushort "
                     "q=0;q<DNF_DVB/8;q++)simdgroup_load(sa[q],Ss+rt*8u*DNF_DVB+q*8u,DNF_DVB);"
                     "for(uint k=0u;k<64u;k+=8u){M8 a;simdgroup_load(a,Kp+k*dk+rt*8u,dk,0,true);"
                     "for(ushort q=0;q<DNF_DVB/8;q++){M8 "
                     "b;simdgroup_load(b,Dt+k*DNF_DVB+q*8u,DNF_DVB);"
                     "simdgroup_multiply_accumulate(sa[q],a,b,sa[q]);}}"
                     "for(ushort "
                     "q=0;q<DNF_DVB/8;q++)simdgroup_store(sa[q],Ss+rt*8u*DNF_DVB+q*8u,DNF_DVB);}"
                     "threadgroup_barrier(mem_flags::mem_threadgroup);}"
                     "for(uint e=ti;e<dk*DNF_DVB;e+=128u)Sg[(e/DNF_DVB)*dv+e%DNF_DVB]=Ss[e];}\n";

static const char metal_dnf_out_source[] = GEIST_DN_SIG(
        "dnf_out") "uint2 tg[[threadgroup_position_in_grid]],ushort "
                   "ti[[thread_index_in_threadgroup]],"
                   "ushort sg[[simdgroup_index_in_threadgroup]],ushort "
                   "ln[[thread_index_in_simdgroup]]){"
                   "threadgroup float Am[4096];threadgroup float gm[64];threadgroup float "
                   "stg[1024];"
                   "DF l=dfl(p);uint c=tg.x,h=tg.y;if(h>=p.nvh||c*64u>=l.lp)return;"
                   "uint hk=h%p.nkh,t0=c*64u,dk=p.dk,dv=p.dv;"
                   "if(ti<64u)gm[ti]=scr[l.gam+h*l.lp+t0+ti];"
                   "device const float*Qp=scr+l.y+(hk*l.lp+t0)*dk;device const "
                   "float*Kp=Qp+p.nkh*l.lp*dk;"
                   "for(ushort ii=0;ii<2;ii++){uint i=ii==0?sg:7u-sg;for(uint j=0u;j<=i;j++){"
                   "M8 acc=make_filled_simdgroup_matrix<float,8>(0.0f);for(uint "
                   "x=0u;x<dk;x+=8u){M8 a,b;"
                   "simdgroup_load(a,Qp+i*8u*dk+x,dk);simdgroup_load(b,Kp+j*8u*dk+x,dk,0,true);"
                   "simdgroup_multiply_accumulate(acc,a,b,acc);}simdgroup_store(acc,Am+i*512u+j*8u,"
                   "64);}}"
                   "threadgroup_barrier(mem_flags::mem_threadgroup);"
                   "for(uint e=ti;e<4096u;e+=128u){uint "
                   "t=e>>6,s=e&63u;Am[e]=s<=t?exp(gm[t]-gm[s])*Am[e]:0.0f;}"
                   "threadgroup_barrier(mem_flags::mem_threadgroup);"
                   "device const float*Dp=scr+l.u+(h*l.lp+t0)*dv;device "
                   "float*Op=scr+l.o+(h*l.lp+t0)*dv;"
                   "threadgroup float*sb=stg+sg*256u;uint lr=ln>>2,lc=(ln&3u)*8u;"
                   "for(ushort ii=0;ii<2;ii++){uint i=ii==0?sg:7u-sg;uint r=i*8u+lr;float "
                   "eg=exp(gm[r]),ss=0.0f;"
                   "device float*orow=Op+r*dv;"
                   "for(uint cg=0u;cg<dv;cg+=32u){M8 acc[4];"
                   "for(ushort q=0;q<4;q++)acc[q]=make_filled_simdgroup_matrix<float,8>(0.0f);"
                   "for(uint k=0u;k<=i;k++){M8 a;simdgroup_load(a,Am+i*512u+k*8u,64);for(ushort "
                   "q=0;q<4;q++){"
                   "M8 "
                   "b;simdgroup_load(b,Dp+k*8u*dv+cg+q*8u,dv);simdgroup_multiply_accumulate(acc[q],"
                   "a,b,acc[q]);}}"
                   "for(ushort q=0;q<4;q++)simdgroup_store(acc[q],sb+q*8u,32);"
                   "simdgroup_barrier(mem_flags::mem_threadgroup);"
                   "for(uint u=0u;u<8u;u++){uint col=cg+lc+u;float "
                   "o=sb[lr*32u+lc+u]+eg*orow[col];orow[col]=o;ss+=o*o;}"
                   "simdgroup_barrier(mem_flags::mem_threadgroup);}"
                   "ss+=simd_shuffle_xor(ss,ushort(1));ss+=simd_shuffle_xor(ss,ushort(2));"
                   "uint t=t0+r;if(t<p.seq){float inv=rsqrt(ss/float(dv)+p.eps);"
                   "device float*zr=z+p.zo+t*l.vd+h*dv;"
                   "for(uint cg=0u;cg<dv;cg+=32u)for(uint u=0u;u<8u;u++){uint col=cg+lc+u;"
                   "zr[col]=orow[col]*inv*nw[p.no+col]*silu1(zr[col]);}}}}\n";

#endif /* GEIST_METAL_SHADERS_H */
