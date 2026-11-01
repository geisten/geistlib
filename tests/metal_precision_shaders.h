/* Controlled precision benchmark: identical integer-valued inputs are
 * exactly representable in every operand format. Conversion is timed
 * separately; this measures cast/pack, not activation range calibration. */

static const char precision_common[] =
        "#include <metal_stdlib>\n"
        "#include <metal_tensor>\n"
        "#include <metal_packed_numeric>\n"
        "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
        "using namespace metal;\n"
        "struct P{uint m,n,k,stride;};\n";

static const char precision_convert[] =
        "kernel void cvt_f16(device const float*x[[buffer(0)]],device half*y[[buffer(1)]],constant "
        "uint&n4[[buffer(2)]],uint i[[thread_position_in_grid]]){if(i<n4)((device "
        "half4*)y)[i]=half4(((device const float4*)x)[i]);}\n"
        "kernel void cvt_fp8(device const float*x[[buffer(0)]],device "
        "uchar*y[[buffer(1)]],constant uint&n4[[buffer(2)]],uint "
        "i[[thread_position_in_grid]]){if(i<n4)((device "
        "packed_uchar4*)y)[i]=pack<metal_fp8_e4m3_format>(((device const "
        "float4*)x)[i]).as_storage_type();}\n"
        "kernel void cvt_i8(device const float*x[[buffer(0)]],device char*y[[buffer(1)]],constant "
        "uint&n4[[buffer(2)]],uint i[[thread_position_in_grid]]){if(i<n4)((device "
        "char4*)y)[i]=char4(rint(((device const float4*)x)[i]));}\n";

static const char precision_gemm_f16[] =
        "kernel void gemm_f16(device const half*x[[buffer(0)]],device "
        "float*y[[buffer(1)]],constant P&p[[buffer(2)]],uint3 "
        "tg[[threadgroup_position_in_grid]]){\n"
        "using namespace mpp::tensor_ops;\n"
        "const int M=int(p.m),N=int(p.n),K=int(p.k),S=int(p.stride);const int "
        "ra=int(tg.y)*64,rb=int(tg.x)*128;\n"
        "auto a=tensor((device half*)x,dextents<int,2>(K,M),array<int,2>{1,S});auto "
        "b=tensor((device half*)x+ulong(M)*S,dextents<int,2>(K,N),array<int,2>{1,S});\n"
        "matmul2d<matmul2d_descriptor(128,64,static_cast<int>(dynamic_extent),false,true,true,"
        "matmul2d_descriptor::mode::multiply_accumulate),execution_simdgroups<8>> mm;\n"
        "auto c=mm.get_destination_cooperative_tensor<decltype(a),decltype(b),float>();\n"
        "for(uint16_t i=0;i<c.get_capacity();i++)if(c.is_valid_element(i))c[i]=float(0);\n"
        "auto av=a.slice(0,rb);auto bv=b.slice(0,ra);mm.run(av,bv,c);\n"
        "auto out=tensor(y,dextents<int,2>(N,M));c.store(out.slice(ra,rb));}\n";

static const char precision_gemm_fp8[] =
        "kernel void gemm_fp8(device const uchar*x[[buffer(0)]],device "
        "float*y[[buffer(1)]],constant P&p[[buffer(2)]],uint3 "
        "tg[[threadgroup_position_in_grid]]){\n"
        "using namespace mpp::tensor_ops;\n"
        "const int M=int(p.m),N=int(p.n),K=int(p.k),S=int(p.stride);const int "
        "ra=int(tg.y)*64,rb=int(tg.x)*128;\n"
        "using T=tensor<device metal_fp8_e4m3_format,dextents<int,2>,tensor_inline>;T a((device "
        "uchar*)x,dextents<int,2>(K,M),array<int,2>{1,S});T b((device "
        "uchar*)x+ulong(M)*S,dextents<int,2>(K,N),array<int,2>{1,S});\n"
        "matmul2d<matmul2d_descriptor(128,64,static_cast<int>(dynamic_extent),false,true,true,"
        "matmul2d_descriptor::mode::multiply_accumulate),execution_simdgroups<8>> mm;\n"
        "auto c=mm.get_destination_cooperative_tensor<decltype(a),decltype(b),float>();\n"
        "for(uint16_t i=0;i<c.get_capacity();i++)if(c.is_valid_element(i))c[i]=float(0);\n"
        "auto av=a.slice(0,rb);auto bv=b.slice(0,ra);mm.run(av,bv,c);\n"
        "auto out=tensor(y,dextents<int,2>(N,M));c.store(out.slice(ra,rb));}\n";

static const char precision_gemm_i8[] =
        "kernel void gemm_i8(device const int8_t*x[[buffer(0)]],device int*y[[buffer(1)]],constant "
        "P&p[[buffer(2)]],uint3 tg[[threadgroup_position_in_grid]]){\n"
        "using namespace mpp::tensor_ops;\n"
        "const int M=int(p.m),N=int(p.n),K=int(p.k),S=int(p.stride);const int "
        "ra=int(tg.y)*64,rb=int(tg.x)*128;\n"
        "auto a=tensor((device int8_t*)x,dextents<int,2>(K,M),array<int,2>{1,S});auto "
        "b=tensor((device int8_t*)x+ulong(M)*S,dextents<int,2>(K,N),array<int,2>{1,S});\n"
        "matmul2d<matmul2d_descriptor(128,64,static_cast<int>(dynamic_extent),false,true,true,"
        "matmul2d_descriptor::mode::multiply_accumulate),execution_simdgroups<8>> mm;\n"
        "auto c=mm.get_destination_cooperative_tensor<decltype(a),decltype(b),int>();\n"
        "for(uint16_t i=0;i<c.get_capacity();i++)if(c.is_valid_element(i))c[i]=int(0);\n"
        "auto av=a.slice(0,rb);auto bv=b.slice(0,ra);mm.run(av,bv,c);\n"
        "auto out=tensor(y,dextents<int,2>(N,M));c.store(out.slice(ra,rb));}\n";

static const char precision_calibrate[] =
        "kernel void calibrate_i8(device const float*x[[buffer(0)]],device "
        "int8_t*q[[buffer(1)]],constant uint&n4[[buffer(2)]],device uchar*meta[[buffer(3)]],uint3 "
        "tg[[threadgroup_position_in_grid]],ushort ti[[thread_index_in_threadgroup]]){\n"
        "uint lane=ti&31u,b=tg.x*8u+ti/32u,nb=n4/8u;if(b>=nb)return;float "
        "v=x[ulong(b)*32u+lane];float a=simd_max(abs(v));float "
        "d=a>0?max(a/127.0f,0x1p-126f):1.0f;int8_t "
        "z=int8_t(clamp(rint(v/d),-127.0f,127.0f));q[ulong(b)*32u+lane]=z;int "
        "sum=simd_sum(int(z));if(lane==0){((device float*)meta)[b]=d;((device "
        "int*)meta)[nb+b]=sum;}}\n"
        "kernel void calibrate_fp8(device const float*x[[buffer(0)]],device "
        "uchar*q[[buffer(1)]],constant uint&n4[[buffer(2)]],device uchar*meta[[buffer(3)]],uint3 "
        "tg[[threadgroup_position_in_grid]],ushort ti[[thread_index_in_threadgroup]]){\n"
        "uint lane=ti&31u,b=tg.x*8u+ti/32u,nb=n4/8u;if(b>=nb)return;float "
        "v=x[ulong(b)*32u+lane];float a=simd_max(abs(v));uint bits=as_type<uint>(a);int "
        "e=clamp(int(bits>>23)-135+int((bits&0x7fffffu)>0x600000u),-126,120);float "
        "inv=as_type<float>(uint(127-e)<<23);uint start=lane&~3u;\n"
        "float4 "
        "vv=float4(simd_shuffle(v,start),simd_shuffle(v,start+1),simd_shuffle(v,start+2),simd_"
        "shuffle(v,start+3))*inv;\n"
        "if((lane&3u)==0)((device "
        "packed_uchar4*)(q+ulong(b)*32u))[lane/"
        "4u]=pack<metal_fp8_e4m3_format>(vv).as_storage_type();if(lane==0)meta[b]=uchar(e+127);}\n";
