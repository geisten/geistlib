/* Native W8A8 prefill: prepare row-scaled INT8 activations and weights
 * once per matrix call, shared by all tiles. One full-K INT32 tensor dot
 * is rescaled once into FP32. Host caps K <=65536 to prevent overflow.
 * Row scales use ceil(log2(amax/127)) without approximate division:
 * normalized 127 is 1.984375 * 2^6, with mantissa bits 0x7e0000.
 * Zero rows use scale 1; tiny scales clamp to minimum normal FP32.
 * Original GGUF weights remain mapped; row requantization adds error.
 * Workspace holds one prepared matrix pair, not a model-sized cache. */

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
        "return float2(i8half(z),0.0f);}\n"
        "static inline float i8amax(float a,threadgroup float*t,ushort "
        "ti){a=simd_max(a);if((ti&31u)==0)t[ti/"
        "32u]=a;threadgroup_barrier(mem_flags::mem_threadgroup);if(ti<32u){a=simd_max(ti<8u?t[ti]:"
        "0.0f);if(ti==0)t[0]=a;}threadgroup_barrier(mem_flags::mem_threadgroup);return t[0];}\n";

static const char metal_int8_quant_source[] =
        "kernel void int8_quant_x(device const float*x[[buffer(0)]],device "
        "int8_t*q[[buffer(1)]],constant P&p[[buffer(2)]],uint "
        "row[[threadgroup_position_in_grid]],ushort ti[[thread_index_in_threadgroup]]){\n"
        "threadgroup float tmp[8];float a=0;for(uint k=ti;k<p.ni;k+=256u){float "
        "v=x[ulong(p.xo)+ulong(row)*p.xs+k];a=isfinite(v)?max(a,abs(v)):INFINITY;}a=i8amax(a,tmp,"
        "ti);bool finite=isfinite(a);float "
        "d=a>0?as_type<float>(uint(clamp(int(as_type<uint>(a)>>23)-133+int((as_type<uint>(a)&"
        "0x7fffffu)>0x7e0000u),-126,127)+127)<<23):1.0f;\n"
        "for(uint k=ti;k<p.ni;k+=256u){float "
        "v=x[ulong(p.xo)+ulong(row)*p.xs+k];q[ulong(row)*p.ni+k]=finite?int8_t(clamp(rint(v*(1.0f/"
        "d)),-127.0f,127.0f)):int8_t(0);}\n"
        "if(ti==0)((device float*)(q+ulong(p.rows+p.no)*p.ni))[row]=finite?d:NAN;\n"
        "}\n";

static const char metal_int8_pack_q80_source[] =
        "kernel void int8_pack_q80(device const uchar*w[[buffer(0)]],device "
        "int8_t*q[[buffer(1)]],constant P&p[[buffer(2)]],uint "
        "row[[threadgroup_position_in_grid]],ushort ti[[thread_index_in_threadgroup]]){\n"
        "threadgroup float tmp[8];float a=0;for(uint k=ti;k<p.ni;k+=256u){float2 "
        "sc=i8scale<0>(w,p,row,k/32u);float "
        "v=float(i8code<0>(w,p,row,k/"
        "32u,k%32u))*sc.x-sc.y;a=isfinite(v)?max(a,abs(v)):INFINITY;}a=i8amax(a,tmp,ti);bool "
        "finite=isfinite(a);float "
        "d=a>0?as_type<float>(uint(clamp(int(as_type<uint>(a)>>23)-133+int((as_type<uint>(a)&"
        "0x7fffffu)>0x7e0000u),-126,127)+127)<<23):1.0f;\n"
        "for(uint k=ti;k<p.ni;k+=256u){float2 sc=i8scale<0>(w,p,row,k/32u);float "
        "v=float(i8code<0>(w,p,row,k/"
        "32u,k%32u))*sc.x-sc.y;q[ulong(p.rows+row)*p.ni+k]=finite?int8_t(clamp(rint(v*(1.0f/"
        "d)),-127.0f,127.0f)):int8_t(0);}\n"
        "if(ti==0)((device float*)(q+ulong(p.rows+p.no)*p.ni))[p.rows+row]=finite?d:NAN;\n"
        "}\n";

static const char metal_int8_gemm_32_source[] =
        "kernel void int8_gemm_32(device int8_t*q[[buffer(0)]],device "
        "float*y[[buffer(1)]],constant P&p[[buffer(2)]],uint3 "
        "tg[[threadgroup_position_in_grid]]){\n"
        "using namespace mpp::tensor_ops;const int "
        "K=int(p.ni),N=int(p.no),M=int(p.rows),ra=int(tg.y)*64,rb=int(tg.x)*32;\n"
        "auto a=tensor(q,dextents<int,2>(K,M));auto b=tensor(q+ulong(M)*K,dextents<int,2>(K,N));\n"
        "matmul2d<matmul2d_descriptor(32,64,static_cast<int>(dynamic_extent),false,true,true,"
        "matmul2d_descriptor::mode::multiply),execution_simdgroups<8>> mm;\n"
        "auto c=mm.get_destination_cooperative_tensor<decltype(a),decltype(b),int>();auto "
        "av=a.slice(0,rb);auto bv=b.slice(0,ra);mm.run(av,bv,c);\n"
        "device float*sc=(device float*)(q+ulong(M+N)*K);\n"
        "for(uint16_t i=0;i<c.get_capacity();i++)if(c.is_valid_element(i)){auto "
        "ij=c.get_multidimensional_index(i);int "
        "row=rb+ij[1],o=ra+ij[0];if(row<M&&o<N)y[ulong(p.yo)+ulong(row)*p.ys+o]=float(c[i])*sc[row]"
        "*sc[M+o];}\n"
        "}\n";

static const char metal_int8_gemm_64_source[] =
        "kernel void int8_gemm_64(device int8_t*q[[buffer(0)]],device "
        "float*y[[buffer(1)]],constant P&p[[buffer(2)]],uint3 "
        "tg[[threadgroup_position_in_grid]]){\n"
        "using namespace mpp::tensor_ops;const int "
        "K=int(p.ni),N=int(p.no),M=int(p.rows),ra=int(tg.y)*64,rb=int(tg.x)*64;\n"
        "auto a=tensor(q,dextents<int,2>(K,M));auto b=tensor(q+ulong(M)*K,dextents<int,2>(K,N));\n"
        "matmul2d<matmul2d_descriptor(64,64,static_cast<int>(dynamic_extent),false,true,true,"
        "matmul2d_descriptor::mode::multiply),execution_simdgroups<8>> mm;\n"
        "auto c=mm.get_destination_cooperative_tensor<decltype(a),decltype(b),int>();auto "
        "av=a.slice(0,rb);auto bv=b.slice(0,ra);mm.run(av,bv,c);\n"
        "device float*sc=(device float*)(q+ulong(M+N)*K);\n"
        "for(uint16_t i=0;i<c.get_capacity();i++)if(c.is_valid_element(i)){auto "
        "ij=c.get_multidimensional_index(i);int "
        "row=rb+ij[1],o=ra+ij[0];if(row<M&&o<N)y[ulong(p.yo)+ulong(row)*p.ys+o]=float(c[i])*sc[row]"
        "*sc[M+o];}\n"
        "}\n";

static const char metal_int8_gemm_128_source[] =
        "kernel void int8_gemm_128(device int8_t*q[[buffer(0)]],device "
        "float*y[[buffer(1)]],constant P&p[[buffer(2)]],uint3 "
        "tg[[threadgroup_position_in_grid]]){\n"
        "using namespace mpp::tensor_ops;const int "
        "K=int(p.ni),N=int(p.no),M=int(p.rows),ra=int(tg.y)*64,rb=int(tg.x)*128;\n"
        "auto a=tensor(q,dextents<int,2>(K,M));auto b=tensor(q+ulong(M)*K,dextents<int,2>(K,N));\n"
        "matmul2d<matmul2d_descriptor(128,64,static_cast<int>(dynamic_extent),false,true,true,"
        "matmul2d_descriptor::mode::multiply),execution_simdgroups<8>> mm;\n"
        "auto c=mm.get_destination_cooperative_tensor<decltype(a),decltype(b),int>();auto "
        "av=a.slice(0,rb);auto bv=b.slice(0,ra);mm.run(av,bv,c);\n"
        "device float*sc=(device float*)(q+ulong(M+N)*K);\n"
        "for(uint16_t i=0;i<c.get_capacity();i++)if(c.is_valid_element(i)){auto "
        "ij=c.get_multidimensional_index(i);int "
        "row=rb+ij[1],o=ra+ij[0];if(row<M&&o<N)y[ulong(p.yo)+ulong(row)*p.ys+o]=float(c[i])*sc[row]"
        "*sc[M+o];}\n"
        "}\n";

static const char metal_int8_pack_q40_source[] =
        "kernel void int8_pack_q40(device const uchar*w[[buffer(0)]],device "
        "int8_t*q[[buffer(1)]],constant P&p[[buffer(2)]],uint "
        "row[[threadgroup_position_in_grid]],ushort ti[[thread_index_in_threadgroup]]){\n"
        "threadgroup float tmp[8];float a=0;for(uint k=ti;k<p.ni;k+=256u){float2 "
        "sc=i8scale<1>(w,p,row,k/32u);float "
        "v=float(i8code<1>(w,p,row,k/"
        "32u,k%32u))*sc.x-sc.y;a=isfinite(v)?max(a,abs(v)):INFINITY;}a=i8amax(a,tmp,ti);bool "
        "finite=isfinite(a);float "
        "d=a>0?as_type<float>(uint(clamp(int(as_type<uint>(a)>>23)-133+int((as_type<uint>(a)&"
        "0x7fffffu)>0x7e0000u),-126,127)+127)<<23):1.0f;\n"
        "for(uint k=ti;k<p.ni;k+=256u){float2 sc=i8scale<1>(w,p,row,k/32u);float "
        "v=float(i8code<1>(w,p,row,k/"
        "32u,k%32u))*sc.x-sc.y;q[ulong(p.rows+row)*p.ni+k]=finite?int8_t(clamp(rint(v*(1.0f/"
        "d)),-127.0f,127.0f)):int8_t(0);}\n"
        "if(ti==0)((device float*)(q+ulong(p.rows+p.no)*p.ni))[p.rows+row]=finite?d:NAN;\n"
        "}\n";

static const char metal_int8_pack_q4k_source[] =
        "kernel void int8_pack_q4k(device const uchar*w[[buffer(0)]],device "
        "int8_t*q[[buffer(1)]],constant P&p[[buffer(2)]],uint "
        "row[[threadgroup_position_in_grid]],ushort ti[[thread_index_in_threadgroup]]){\n"
        "threadgroup float tmp[8];float a=0;for(uint k=ti;k<p.ni;k+=256u){float2 "
        "sc=i8scale<2>(w,p,row,k/32u);float "
        "v=float(i8code<2>(w,p,row,k/"
        "32u,k%32u))*sc.x-sc.y;a=isfinite(v)?max(a,abs(v)):INFINITY;}a=i8amax(a,tmp,ti);bool "
        "finite=isfinite(a);float "
        "d=a>0?as_type<float>(uint(clamp(int(as_type<uint>(a)>>23)-133+int((as_type<uint>(a)&"
        "0x7fffffu)>0x7e0000u),-126,127)+127)<<23):1.0f;\n"
        "for(uint k=ti;k<p.ni;k+=256u){float2 sc=i8scale<2>(w,p,row,k/32u);float "
        "v=float(i8code<2>(w,p,row,k/"
        "32u,k%32u))*sc.x-sc.y;q[ulong(p.rows+row)*p.ni+k]=finite?int8_t(clamp(rint(v*(1.0f/"
        "d)),-127.0f,127.0f)):int8_t(0);}\n"
        "if(ti==0)((device float*)(q+ulong(p.rows+p.no)*p.ni))[p.rows+row]=finite?d:NAN;\n"
        "}\n";

static const char metal_int8_q4k_128_source[] =
        "kernel void int8_q4k_128(device int8_t*q[[buffer(0)]],device "
        "float*y[[buffer(1)]],constant P&p[[buffer(2)]],uint3 "
        "tg[[threadgroup_position_in_grid]]){\n"
        "using namespace mpp::tensor_ops;const int "
        "K=int(p.ni),N=int(p.no),M=int(p.rows),ra=int(tg.y)*64,rb=int(tg.x)*128;\n"
        "auto a=tensor(q,dextents<int,2>(K,M));auto b=tensor(q+ulong(M)*K,dextents<int,2>(K,N));\n"
        "matmul2d<matmul2d_descriptor(128,64,static_cast<int>(dynamic_extent),false,true,true,"
        "matmul2d_descriptor::mode::multiply),execution_simdgroups<8>> mm;\n"
        "auto c=mm.get_destination_cooperative_tensor<decltype(a),decltype(b),int>();auto "
        "av=a.slice(0,rb);auto bv=b.slice(0,ra);mm.run(av,bv,c);\n"
        "device float*sc=(device float*)(q+ulong(M+N)*K);\n"
        "for(uint16_t i=0;i<c.get_capacity();i++)if(c.is_valid_element(i)){auto "
        "ij=c.get_multidimensional_index(i);int "
        "row=rb+ij[1],o=ra+ij[0];if(row<M&&o<N)y[ulong(p.yo)+ulong(row)*p.ys+o]=float(c[i])*sc[row]"
        "*sc[M+o];}\n"
        "}\n";
