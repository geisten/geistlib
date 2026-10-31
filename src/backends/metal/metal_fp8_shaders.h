/* Native FP8 E4M3 + E8M0 block scales, Metal 4.1 (macOS 27).
 * Both operands reach MPP as FP8; FP32 accumulation, packed GGUF weights.
 * Each 32-value block uses a power-of-two scale and round-to-nearest-even.
 * The exponent/mantissa bits select ceil(log2(amax/448)) without a
 * transcendental; the clamp covers finite FP32 and E8M0 scale ranges.
 * FP8 rows have the required 128-byte stride; all tile storage is bounded. */

static const char metal_mm_fp8_helper_source[] =
        "#include <metal_packed_numeric>\n"
        "using namespace metal;\n"
        "using FP8Plane=tensor_blockwise<tensor_plane_scales,threadgroup "
        "metal_fp8_ue8m0_format,32,1>;\n"
        "using FP8Tensor=tensor<threadgroup "
        "metal_fp8_e4m3_format,dextents<int,2>,tensor_inline,FP8Plane>;\n"
        "static inline void stage_fp8(thread float (&v)[32],threadgroup uchar* data,threadgroup "
        "uchar* scales,int row,int block){\n"
        "float amax=0.0f;for(int i=0;i<32;i++)amax=max(amax,abs(v[i]));\n"
        "uint bits=as_type<uint>(amax);\n"
        "int e=clamp(int(bits>>23)-135+int((bits&0x7fffffu)>0x600000u),-126,120);\n"
        "float inv=as_type<float>(uint(127-e)<<23);scales[row*4+block]=uchar(e+127);\n"
        "for(int i=0;i<32;i+=4){float4 f=float4(v[i],v[i+1],v[i+2],v[i+3])*inv;\n"
        "auto q=pack<metal_fp8_e4m3_format>(f);\n"
        "*((threadgroup packed_uchar4*)(data+row*128+block*32+i))=q.as_storage_type();}\n"
        "}\n";

#define GEIST_METAL_MM_FP8_KERNEL(NAME, BLK, DQ, QKNL)                                             \
    "kernel void matmul_" NAME                                                                     \
    "_mm_tc(device const float*x[[buffer(0)]],device const uchar*w[[buffer(1)]],device "           \
    "float*y[[buffer(2)]],constant P&p[[buffer(3)]],threadgroup uchar*s[[threadgroup(0)]],uint3 "  \
    "tg[[threadgroup_position_in_grid]],ushort ti[[thread_index_in_threadgroup]]){\n"              \
    "using namespace mpp::tensor_ops;\n"                                                           \
    "constexpr int NRA=64,NRB=128,NK=128,QK_NL=" QKNL ";\n"                                        \
    "const int "                                                                                   \
    "K=int(p.ni),M=int(p.no),N=int(p.rows),XS=int(p.xs),ra=int(tg.y)*NRA,rb=int(tg.x)*NRB;\n"      \
    "threadgroup uchar*sa=s;threadgroup uchar*sb=s+8192;\n"                                        \
    "threadgroup uchar*wa=s+24576;threadgroup uchar*xa=s+24832;\n"                                 \
    "FP8Plane wp(wa),xp(xa);\n"                                                                    \
    "FP8Tensor tA(sa,dextents<int,2>(NK,NRA),array<int,2>{1,NK},wp);\n"                            \
    "FP8Tensor tB(sb,dextents<int,2>(NK,NRB),array<int,2>{1,NK},xp);\n"                            \
    "matmul2d<matmul2d_descriptor(NRB,NRA,NK,false,true,true,matmul2d_descriptor::mode::multiply_" \
    "accumulate),execution_simdgroups<8>> mm;\n"                                                   \
    "auto c=mm.get_destination_cooperative_tensor<decltype(tB),decltype(tA),float>();\n"           \
    "for(uint16_t i=0;i<c.get_capacity();i++){if(c.is_valid_element(i))c[i]=0.0f;}\n"              \
    "for(int k0=0;k0<K;k0+=NK){\n"                                                                 \
    "const int wr=int(ti)/4,wc=int(ti)%4;\n"                                                       \
    "float v[32];for(int h=0;h<2;h++){const int kp=k0+wc*32+h*16;half4x4 t=half4x4(0.0h);\n"       \
    "if(ra+wr<M&&kp<K){device const " BLK "*b=(device const " BLK                                  \
    "*)(w+p.wo)+(ra+wr)*int(p.bpr);" DQ "(b+kp/(16*QK_NL),short((kp/16)%QK_NL),t);}\n"             \
    "for(int i=0;i<16;i++)v[h*16+i]=float(t[i/4][i%4]);}\n"                                        \
    "stage_fp8(v,sa,wa,wr,wc);\n"                                                                  \
    "for(int j=int(ti);j<NRB*4;j+=256){const int xr=j/4,xc=j%4;\n"                                 \
    "for(int i=0;i<32;i++){int k=k0+xc*32+i;v[i]=(rb+xr<N&&k<K)?x[p.xo+(rb+xr)*XS+k]:0.0f;}\n"     \
    "stage_fp8(v,sb,xa,xr,xc);}\n"                                                                 \
    "threadgroup_barrier(mem_flags::mem_threadgroup);mm.run(tB,tA,c);threadgroup_barrier(mem_"     \
    "flags::mem_threadgroup);\n"                                                                   \
    "}\n"                                                                                          \
    "auto "                                                                                        \
    "out=tensor(y+p.yo,dextents<int,2>(M,N),array<int,2>{1,int(p.ys)});c.store(out.slice(ra,rb));" \
    "\n"                                                                                           \
    "}\n"

static const char metal_mm_fp8_q4k_source[] =
        GEIST_METAL_MM_FP8_KERNEL("q4k", "block_q4_K", "dq4", "16");

static const char metal_mm_fp8_q6k_source[] =
        GEIST_METAL_MM_FP8_KERNEL("q6k", "block_q6_K", "dq6b", "16");

static const char metal_mm_fp8_q40_source[] = GEIST_METAL_MM_FP8_KERNEL("q40", "bq40", "dq40", "2");

static const char metal_mm_fp8_q80_source[] = GEIST_METAL_MM_FP8_KERNEL("q80", "bq80", "dq80", "2");

static const char metal_mm_fp8_q41_source[] = GEIST_METAL_MM_FP8_KERNEL("q41", "bq41", "dq41", "2");

static const char metal_mm_fp8_q5k_source[] =
        GEIST_METAL_MM_FP8_KERNEL("q5k", "bq5k", "dq5k", "16");

static const char metal_mm_fp8_iq4nl_source[] =
        GEIST_METAL_MM_FP8_KERNEL("iq4nl", "biq4nl", "dqiq4nl", "2");

static const char metal_mm_fp8_iq4xs_source[] =
        GEIST_METAL_MM_FP8_KERNEL("iq4xs", "biq4xs", "dqiq4xs", "16");

static const char metal_mm_fp8_q3k_source[] =
        GEIST_METAL_MM_FP8_KERNEL("q3k", "bq3k", "dq3k", "16");

static const char metal_mm_fp8_iq3s_source[] =
        GEIST_METAL_MM_FP8_KERNEL("iq3s", "biq3s", "dqiq3s", "16");

static const char metal_mm_fp8_pq2_source[] =
        GEIST_METAL_MM_FP8_KERNEL("pq2", "bpq2", "dqpq2", "8");
