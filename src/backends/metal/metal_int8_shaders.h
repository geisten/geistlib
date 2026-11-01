/* INT8 prefill: retain original Q8_0/Q4_0/Q4_K codes and block scales.
 * X is quantized once per call and reused across output tiles. Each
 * 32-term INT32 dot is rescaled into FP32, preventing long-K overflow.
 * FP32 running sums are bit-cast into a second INT32 cooperative tensor
 * of the same type/layout; it is accessed elementwise only and never
 * passed to an integer matmul or integer tensor store.
 * Three predeclared row tiles permit an identical-binary tuning sweep.
 */

static const char metal_int8_head_source[] =
        "#include <metal_tensor>\n"
        "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
        "static inline float i8half(device const uchar*b){return "
        "float(as_type<half>(ushort(ushort(b[0])|(ushort(b[1])<<8))));}\n"
        "static inline uint2 i8sm(uint j,device const uchar*q){return "
        "j<4?uint2(q[j]&63u,q[j+4]&63u):uint2((q[j+4]&15u)|((q[j-4]>>6)<<4),(q[j+4]>>4)|((q[j]>>6)<"
        "<4));}\n"
        "template<ushort F>static inline device const uchar* i8block(device const uchar*w,constant "
        "P&p,uint row,uint b){if constexpr(F==2)return "
        "w+p.wo+(ulong(row)*(p.ni/256u)+b/8u)*144u;else return "
        "w+p.wo+(ulong(row)*p.bpr+b)*(F==0?34u:18u);}\n"
        "template<ushort F>static inline int8_t i8code(device const uchar*w,constant P&p,uint "
        "row,uint b,uint k){device const uchar*z=i8block<F>(w,p,row,b);if constexpr(F==0)return "
        "int8_t(as_type<char>(z[2+k]));else if constexpr(F==1)return "
        "int8_t(int((z[2+(k&15u)]>>(k<16u?0u:4u))&15u)-8);else{uint g=b&7u;return "
        "int8_t((z[16u+(g/2u)*32u+k]>>(4u*(g&1u)))&15u);}}\n"
        "template<ushort F>static inline float2 i8scale(device const uchar*w,constant P&p,uint "
        "row,uint b){device const uchar*z=i8block<F>(w,p,row,b);if constexpr(F==2){uint2 "
        "sm=i8sm(b&7u,z+4);return float2(i8half(z)*float(sm.x),i8half(z+2)*float(sm.y));}else "
        "return float2(i8half(z),0.0f);}\n";

static const char metal_int8_quant_source[] =
        "kernel void int8_quant_x(device const float*x[[buffer(0)]],device "
        "int8_t*q[[buffer(1)]],constant P&p[[buffer(2)]],uint3 "
        "tg[[threadgroup_position_in_grid]],ushort ti[[thread_index_in_threadgroup]]){\n"
        "uint lane=ti&31u,b=tg.x*8u+ti/32u;const ulong "
        "count=ulong(p.rows)*p.bpr;if(ulong(b)>=count)return;\n"
        "uint row=b/p.bpr,k=(b%p.bpr)*32u+lane;\n"
        "float v=x[ulong(p.xo)+ulong(row)*p.xs+k];bool finite=simd_all(isfinite(v));\n"
        "float a=simd_max(abs(v));float d=a>0.0f?max(a/127.0f,0x1p-126f):1.0f;\n"
        "int8_t z=finite?int8_t(clamp(rint(v/d),-127.0f,127.0f)):int8_t(0);\n"
        "q[ulong(row)*p.ni+k]=z;\n"
        "int sum=simd_sum(int(z));\n"
        "if(lane==0){device float*sc=(device float*)(q+ulong(p.rows)*p.ni);device int*ss=(device "
        "int*)(sc+count);sc[b]=finite?d:NAN;ss[b]=sum;}\n"
        "}\n";

#define GEIST_METAL_INT8_KERNEL(NAME, TILE, KIND)                                                  \
    "kernel void " NAME "(device int8_t*x[[buffer(0)]],device const uchar*w[[buffer(1)]],device "  \
    "float*y[[buffer(2)]],constant P&p[[buffer(3)]],threadgroup uchar*sm[[threadgroup(0)]],uint3 " \
    "tg[[threadgroup_position_in_grid]],ushort ti[[thread_index_in_threadgroup]]){\n"              \
    "using namespace mpp::tensor_ops;constexpr int TM=" TILE                                       \
    ",TN=64,BK=32;constexpr ushort F=" KIND ";\n"                                                  \
    "int ra=int(tg.y)*TN,rb=int(tg.x)*TM,K=int(p.ni),M=int(p.no),N=int(p.rows);\n"                 \
    "threadgroup int8_t*sw=(threadgroup int8_t*)sm;threadgroup float*ds=(threadgroup "             \
    "float*)(sm+2048);threadgroup float*ms=ds+64;\n"                                               \
    "device float*xs=(device float*)(x+ulong(p.rows)*p.ni);device int*sx=(device "                 \
    "int*)(xs+ulong(p.rows)*p.bpr);\n"                                                             \
    "auto a0=tensor(x,dextents<int,2>(BK,TM),array<int,2>{1,K});auto "                             \
    "b=tensor(sw,extents<int,BK,TN>());\n"                                                         \
    "matmul2d<matmul2d_descriptor(TM,TN,BK,false,true,true,matmul2d_descriptor::mode::multiply),"  \
    "execution_simdgroups<8>> mm;\n"                                                               \
    "auto dot=mm.get_destination_cooperative_tensor<decltype(a0),decltype(b),int>();auto "         \
    "sum=mm.get_destination_cooperative_tensor<decltype(a0),decltype(b),int>();\n"                 \
    "#pragma unroll\n"                                                                             \
    "for(uint16_t "                                                                                \
    "i=0;i<sum.get_capacity();i++)if(sum.is_valid_element(i))sum[i]=as_type<int>(0.0f);\n"         \
    "for(uint block=0;block<p.bpr;block++){\n"                                                     \
    "for(uint j=ti;j<2048u;j+=256u){uint r=j/32u,k=j%32u;int "                                     \
    "row=ra+int(r);sw[j]=row<M?i8code<F>(w,p,uint(row),block,k):int8_t(0);}\n"                     \
    "if(ti<TN){int row=ra+int(ti);float2 "                                                         \
    "sc=row<M?i8scale<F>(w,p,uint(row),block):float2(0);ds[ti]=sc.x;ms[ti]=sc.y;}\n"               \
    "threadgroup_barrier(mem_flags::mem_threadgroup);\n"                                           \
    "auto "                                                                                        \
    "a=tensor(x+ulong(rb)*p.ni+block*32u,dextents<int,2>(BK,N-rb),array<int,2>{1,K});mm.run(a,b,"  \
    "dot);\n"                                                                                      \
    "#pragma unroll\n"                                                                             \
    "for(uint16_t i=0;i<dot.get_capacity();i++)if(dot.is_valid_element(i)){auto "                  \
    "ij=dot.get_multidimensional_index(i);int r=rb+ij[1];float "                                   \
    "scale=r<N?xs[ulong(r)*p.bpr+block]:0.0f;float bias=0.0f;if constexpr(F==2){float "            \
    "sxv=r<N?float(sx[ulong(r)*p.bpr+block]):0.0f;bias=sxv*ms[ij[0]];}sum[i]=as_type<int>(as_"     \
    "type<float>(sum[i])+scale*(float(dot[i])*ds[ij[0]]-bias));}\n"                                \
    "threadgroup_barrier(mem_flags::mem_threadgroup);\n"                                           \
    "}\n"                                                                                          \
    "#pragma unroll\n"                                                                             \
    "for(uint16_t i=0;i<sum.get_capacity();i++)if(sum.is_valid_element(i)){auto "                  \
    "ij=sum.get_multidimensional_index(i);int "                                                    \
    "r=rb+ij[1],o=ra+ij[0];if(r<N&&o<M)y[ulong(p.yo)+ulong(r)*p.ys+o]=as_type<float>(sum[i]);}\n"  \
    "}\n"

static const char metal_int8_q80_32_source[] = GEIST_METAL_INT8_KERNEL("int8_q80_32", "32", "0");

static const char metal_int8_q80_64_source[] = GEIST_METAL_INT8_KERNEL("int8_q80_64", "64", "0");

static const char metal_int8_q80_128_source[] = GEIST_METAL_INT8_KERNEL("int8_q80_128", "128", "0");

static const char metal_int8_q40_32_source[] = GEIST_METAL_INT8_KERNEL("int8_q40_32", "32", "1");

static const char metal_int8_q40_64_source[] = GEIST_METAL_INT8_KERNEL("int8_q40_64", "64", "1");

static const char metal_int8_q40_128_source[] = GEIST_METAL_INT8_KERNEL("int8_q40_128", "128", "1");

static const char metal_int8_q4k_32_source[] = GEIST_METAL_INT8_KERNEL("int8_q4k_32", "32", "2");

static const char metal_int8_q4k_64_source[] = GEIST_METAL_INT8_KERNEL("int8_q4k_64", "64", "2");

static const char metal_int8_q4k_128_source[] = GEIST_METAL_INT8_KERNEL("int8_q4k_128", "128", "2");
