// SPDX-License-Identifier: Apache-2.0 AND MIT
// Modified: cooperative projections, fused recurrence and shared transfers.
// Recurrence/convolution arithmetic: SpeechLens, Apache-2.0.
// Cooperative-tensor fragment layout: Copyright © 2025 Apple Inc., MIT.
// Tiled cooperative projections keep their intermediates in threadgroup memory.
import Foundation

enum FusedMambaSource {
    // Conventional SIMD-group MMA path for GPUs without neural accelerators.
    static let portableInput = #"""
    inline void reuse_input(threadgroup const half* tile,device const half* w,threadgroup half* output,uint group){
        for(uint m=0;m<MAMBA_ROWS;m+=8){
            for(uint n=group*64;n<group*64+64;n+=8){
                simdgroup_float8x8 accum=make_filled_simdgroup_matrix<float,8>(0.0f);
                for(uint k=0;k<64;k+=8){
                    simdgroup_half8x8 a,b;
                    simdgroup_load(a,tile+m*64+k,64);simdgroup_load(b,w+k*1024+n,1024);
                    simdgroup_multiply_accumulate(accum,a,b,accum);
                }
                simdgroup_half8x8 result;
                result.thread_elements()[0]=half(accum.thread_elements()[0]);
                result.thread_elements()[1]=half(accum.thread_elements()[1]);
                simdgroup_store(result,output+m*512+n,512);
            }
        }
    }
    """#

    static func portableHelper(rows: Int) -> String {
        """
        #define MAMBA_ROWS \(rows)
        template<bool PROJECTION> inline void reuse_direct(threadgroup const half* tile,device const half* w,threadgroup half* projected,device half* out,uint group,uint begin,uint length,uint batch,uint height,uint width,uint sequence,uint direction,uint positions){
            if(PROJECTION && group>=5)return;
            for(uint m=0;m<MAMBA_ROWS;m+=8){
                simdgroup_float8x8 accum=make_filled_simdgroup_matrix<float,8>(0.0f);
                for(uint k=0;k<256;k+=8){
                    simdgroup_half8x8 a,b;
                    simdgroup_load(a,tile+m*256+k,256);simdgroup_load(b,w+k*64+group*8,64);
                    simdgroup_multiply_accumulate(accum,a,b,accum);
                }
                simdgroup_half8x8 result;
                result.thread_elements()[0]=half(accum.thread_elements()[0]);
                result.thread_elements()[1]=half(accum.thread_elements()[1]);
                simdgroup_store(result,projected+m*64+group*8,64);
            }
        }
        """
    }

    static func portableSource(unroll: Bool) -> String {
        var value=fullSource(unroll:unroll)
        let tail=value.range(of:"    reuse_direct<false>")!
        value.replaceSubrange(tail.lowerBound..<value.endIndex,with:#"""
            reuse_direct<false>(tile,out_weight+direction*256*64,projected,out,group,begin,length,batch,height,width,sequence,direction,positions);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for(uint ix=d;ix<MAMBA_ROWS*64;ix+=256){
                uint i=begin+ix/64;
                if(i<length){
                    uint step=direction==0?i:length-1-i;
                    uint row=TIME_AXIS?step:sequence%height,col=TIME_AXIS?sequence%width:step;
                    uint pos=(batch*height+row)*width+col;
                    out[(direction*positions+pos)*64+ix%64]=projected[ix];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        """#)
        return value
    }

    static let inputHelper = #"""
    inline void reuse_input(threadgroup const half* tile,device const half* w,threadgroup half* output,uint group){
        uint lane=__metal_get_thread_index_in_simdgroup(ushort()),q=lane>>2;
        uint fm=(q&4)|((lane>>1)&3),fn=((q&2)|(lane&1))*4;
        constexpr auto desc=mpp::tensor_ops::matmul2d_descriptor(16,32,16,false,false,true,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
        mpp::tensor_ops::matmul2d<desc,metal::execution_simdgroup> op;
        for(uint part=0;part<2;part++){
            uint cn=group*64+part*32;
            auto a=op.get_left_input_cooperative_tensor<half,half,float>();
            auto b=op.get_right_input_cooperative_tensor<half,half,float>();
            auto c=op.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(a)>,metal::remove_addrspace_t<decltype(b)>,float>();
            #pragma clang loop unroll(full)
            for(uint i=0;i<16;i++)c[i]=0.0f;
            for(uint k=0;k<64;k+=16){
                #pragma clang loop unroll(full)
                for(uint i=0;i<8;i++){
                    uint r=fm+(i/4)*8,col=fn+i%4;
                    a[i]=tile[r*64+k+col];
                    b[i]=w[(k+r)*1024+cn+col];b[i+8]=w[(k+r)*1024+cn+16+col];
                }
                op.run(a,b,c);
            }
            #pragma clang loop unroll(full)
            for(uint i=0;i<8;i++){
                uint r=fm+(i/4)*8,col=cn+fn+i%4;
                output[r*512+col]=half(c[i]);output[r*512+col+16]=half(c[i+8]);
            }
        }
    }
    """#

    static func fullSource(unroll: Bool) -> String {
        var value=directSource(unroll:unroll).replacingOccurrences(of:"xz_shape",with:"x_shape")
        value=value.replacingOccurrences(of:"threadgroup half tile[MAMBA_ROWS*256];",with:"threadgroup half tile[MAMBA_ROWS*256];\nthreadgroup half input_tile[MAMBA_ROWS*512];")
        value=value.replacingOccurrences(of:"for(uint begin=0;begin<length;begin+=MAMBA_ROWS){",with:"""
        for(uint begin=0;begin<length;begin+=MAMBA_ROWS){
            for(uint ix=d;ix<MAMBA_ROWS*64;ix+=256){
                uint i=begin+ix/64;
                half v=0;
                if(i<length){
                    uint step=direction==0?i:length-1-i;
                    uint row=TIME_AXIS?step:sequence%height,col=TIME_AXIS?sequence%width:step;
                    uint pos=(batch*height+row)*width+col;
                    v=half(x[pos*64+ix%64]);
                }
                projected[ix]=v;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            reuse_input(projected,input_weight+direction*512,input_tile,group);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        """)
        value=value.replacingOccurrences(of:"xz[pos*1024+direction*512+d]",with:"input_tile[j*512+d]")
            .replacingOccurrences(of:"xz[pos*1024+direction*512+256+d]",with:"input_tile[j*512+256+d]")
        return value
    }

    static func directHelper(rows: Int) -> String {
        var value=naxHelper(rows:rows)
        value=value.replacingOccurrences(of:"inline void reuse_matrix(threadgroup const half* tile,device const half* w,threadgroup float* matrix,uint group){",with:"""
        template<bool PROJECTION> inline void reuse_direct(threadgroup const half* tile,device const half* w,threadgroup half* projected,device half* out,uint group,uint begin,uint length,uint batch,uint height,uint width,uint sequence,uint direction,uint positions){
        """)
        value=value.replacingOccurrences(of:"matrix[r*64+col]=c[i];matrix[r*64+col+16]=c[i+8];",with:"""
        if(PROJECTION){projected[r*64+col]=half(c[i]);projected[r*64+col+16]=half(c[i+8]);}
        else if(begin+r<length){
            uint step=direction==0?begin+r:length-1-begin-r;
            uint row=TIME_AXIS?step:sequence%height,column=TIME_AXIS?sequence%width:step;
            uint pos=(batch*height+row)*width+column,index=(direction*positions+pos)*64+col;
            out[index]=half(c[i]);out[index+16]=half(c[i+8]);
        }
        """)
        return value
    }

    static func directSource(unroll: Bool) -> String {
        var value=configured(unroll:unroll,nax:true)
        value=value.replacingOccurrences(of:"threadgroup float matrix[MAMBA_ROWS*64];",with:"")
        let arguments="projected,out,group,begin,length,batch,height,width,sequence,direction,positions"
        value=value.replacingOccurrences(of:"reuse_matrix(tile,x_weight+direction*256*64,matrix,group);",with:"reuse_direct<true>(tile,x_weight+direction*256*64,\(arguments));")
        value=value.replacingOccurrences(of:"    for(uint ix=d;ix<MAMBA_ROWS*64;ix+=256)if(ix%64<36)projected[ix]=half(matrix[ix]);\n    threadgroup_barrier(mem_flags::mem_threadgroup);",with:"")
        let tail=value.range(of:"    reuse_matrix(tile,out_weight+")!
        value.replaceSubrange(tail.lowerBound..<value.endIndex,with:"""
            reuse_direct<false>(tile,out_weight+direction*256*64,\(arguments));
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        """)
        return value
    }

    // Cooperative-tensor fragment layout follows MLX's MIT-licensed NAX helper.
    static func naxHelper(rows: Int) -> String {
        """
        #define MAMBA_ROWS \(rows)
        inline void reuse_matrix(threadgroup const half* tile,device const half* w,threadgroup float* matrix,uint group){
            if(group>=MAMBA_ROWS/8)return;
            uint lane=simd_lane_id, q=lane>>2;
            uint fm=(q&4)|((lane>>1)&3),fn=((q&2)|(lane&1))*4;
            uint rm=(group/2)*16,cn=(group%2)*32;
            constexpr auto desc=mpp::tensor_ops::matmul2d_descriptor(16,32,16,false,false,true,
                mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
            mpp::tensor_ops::matmul2d<desc,metal::execution_simdgroup> op;
            auto a=op.get_left_input_cooperative_tensor<half,half,float>();
            auto b=op.get_right_input_cooperative_tensor<half,half,float>();
            auto c=op.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(a)>,metal::remove_addrspace_t<decltype(b)>,float>();
            #pragma clang loop unroll(full)
            for(uint i=0;i<16;i++)c[i]=0.0f;
            for(uint k=0;k<256;k+=16){
                #pragma clang loop unroll(full)
                for(uint i=0;i<8;i++){
                    uint r=fm+(i/4)*8,col=fn+i%4;
                    a[i]=tile[(rm+r)*256+k+col];
                    b[i]=w[(k+r)*64+cn+col];
                    b[i+8]=w[(k+r)*64+cn+16+col];
                }
                op.run(a,b,c);
            }
            #pragma clang loop unroll(full)
            for(uint i=0;i<8;i++){
                uint r=rm+fm+(i/4)*8,col=cn+fn+i%4;
                matrix[r*64+col]=c[i];matrix[r*64+col+16]=c[i+8];
            }
        }
        """.replacingOccurrences(of:"uint lane=simd_lane_id",with:"uint lane=__metal_get_thread_index_in_simdgroup(ushort())")
    }

    static func helper(rows: Int) -> String {
        """
        #define MAMBA_ROWS \(rows)
        inline void reuse_matrix(threadgroup const half* tile,device const half* w,threadgroup float* matrix,uint group){
            for(uint m=0;m<MAMBA_ROWS;m+=8){
                simdgroup_float8x8 accum=make_filled_simdgroup_matrix<float,8>(0.0f);
                for(uint k=0;k<256;k+=8){
                    simdgroup_half8x8 a,b;
                    simdgroup_load(a,tile+m*256+k,256);
                    simdgroup_load(b,w+k*64+group*8,64);
                    simdgroup_multiply_accumulate(accum,a,b,accum);
                }
                simdgroup_store(accum,matrix+m*64+group*8,64);
            }
        }
        """
    }

    static func configured(unroll: Bool, nax: Bool = false) -> String {
        var value = source
        let endToken="    simdgroup_store(accum,matrix+group*8,64);"
        let first=value.range(of:"    simdgroup_float8x8 accum=")!
        let firstEnd=value.range(of:endToken,range:first.upperBound..<value.endIndex)!
        value.replaceSubrange(first.lowerBound..<firstEnd.upperBound,with:nax ? "    reuse_matrix(tile,x_weight+direction*256*64,matrix,group);" : "    if(group<5)reuse_matrix(tile,x_weight+direction*256*64,matrix,group);")
        let second=value.range(of:"    accum=make_filled_simdgroup_matrix")!
        let secondEnd=value.range(of:endToken,range:second.upperBound..<value.endIndex)!
        value.replaceSubrange(second.lowerBound..<secondEnd.upperBound,with:"    reuse_matrix(tile,out_weight+direction*256*64,matrix,group);")
        value=value.replacingOccurrences(of:"8*256",with:"MAMBA_ROWS*256")
            .replacingOccurrences(of:"8*64",with:"MAMBA_ROWS*64")
            .replacingOccurrences(of:"begin+=8",with:"begin+=MAMBA_ROWS")
            .replacingOccurrences(of:"j<8",with:"j<MAMBA_ROWS")
            .replacingOccurrences(of:"projected[d]=half(matrix[d]);projected[d+256]=half(matrix[d+256]);",
                with:"for(uint ix=d;ix<MAMBA_ROWS*64;ix+=256)if(ix%64<36)projected[ix]=half(matrix[ix]);")
        if unroll {
            value=value.replacingOccurrences(of:"for(uint n=0;n<16;n++)",with:"\n#pragma clang loop unroll(full)\nfor(uint n=0;n<16;n++)")
        }
        return value
    }

    static let source = #"""
uint gid=threadgroup_position_in_grid.x,d=thread_position_in_threadgroup.x;
uint group=simdgroup_index_in_threadgroup;
uint height=xz_shape[1],width=xz_shape[2],batches=xz_shape[0];
uint positions=batches*height*width,sequence_batch=TIME_AXIS?batches*width:batches*height;
uint length=TIME_AXIS?height:width,direction=gid/sequence_batch,sequence=gid%sequence_batch;
uint batch=TIME_AXIS?sequence/width:sequence/height,pd=direction*256+d;
threadgroup half tile[8*256];
threadgroup half projected[8*64];
threadgroup float matrix[8*64];
float state[16],decay_base[16];
for(uint n=0;n<16;n++){state[n]=0.0f;decay_base[n]=-fast::exp(A_log[pd*16+n]);}
half4 dtw=half4(dt_weight[pd*4],dt_weight[pd*4+1],dt_weight[pd*4+2],dt_weight[pd*4+3]);
float delta_bias=dt_bias[pd],skip=Dparam[pd];
float w0=weight[pd*4],w1=weight[pd*4+1],w2=weight[pd*4+2],w3=weight[pd*4+3];
float bias_d=bias[pd],x_m3=0.0f,x_m2=0.0f,x_m1=0.0f;
for(uint begin=0;begin<length;begin+=8){
    for(uint j=0;j<8;j++){
        uint i=begin+j;
        half h=0;
        if(i<length){
            uint step=direction==0?i:length-1-i;
            uint row=TIME_AXIS?step:sequence%height,col=TIME_AXIS?sequence%width:step;
            uint pos=(batch*height+row)*width+col;
            float current=xz[pos*1024+direction*512+d];
            float value=bias_d+w0*x_m3+w1*x_m2+w2*x_m1+w3*current;
            h=half(value);h=h*reuse_sigmoid(h);
            x_m3=x_m2;x_m2=x_m1;x_m1=current;
        }
        tile[j*256+d]=h;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 accum=make_filled_simdgroup_matrix<float,8>(0.0f);
    for(uint k=0;k<256;k+=8){
        simdgroup_half8x8 a,b;
        simdgroup_load(a,tile+k,256);
        simdgroup_load(b,x_weight+direction*256*64+k*64+group*8,64);
        simdgroup_multiply_accumulate(accum,a,b,accum);
    }
    simdgroup_store(accum,matrix+group*8,64);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    projected[d]=half(matrix[d]);projected[d+256]=half(matrix[d+256]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint j=0;j<8;j++){
        uint i=begin+j;
        half result=0;
        if(i<length){
            uint step=direction==0?i:length-1-i;
            uint row=TIME_AXIS?step:sequence%height,col=TIME_AXIS?sequence%width:step;
            uint pos=(batch*height+row)*width+col,bc=j*64;
            float projection=float(projected[bc])*float(dtw.x);
            projection+=float(projected[bc+1])*float(dtw.y);
            projection+=float(projected[bc+2])*float(dtw.z);
            projection+=float(projected[bc+3])*float(dtw.w);
            float delta=selective_scan_softplus(float(half(projection))+delta_bias);
            float input=tile[j*256+d],injection=delta*input,y=0;
            for(uint n=0;n<16;n++){
                float a=fast::exp(delta*decay_base[n]);
                float q=injection*float(projected[bc+4+n]);
                float next=a*state[n]+q;
                state[n]=next;y+=next*float(projected[bc+20+n]);
            }
            y+=input*skip;
            y*=selective_scan_silu(xz[pos*1024+direction*512+256+d]);
            result=half(y);
        }
        tile[j*256+d]=result;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    accum=make_filled_simdgroup_matrix<float,8>(0.0f);
    for(uint k=0;k<256;k+=8){
        simdgroup_half8x8 a,b;
        simdgroup_load(a,tile+k,256);
        simdgroup_load(b,out_weight+direction*256*64+k*64+group*8,64);
        simdgroup_multiply_accumulate(accum,a,b,accum);
    }
    simdgroup_store(accum,matrix+group*8,64);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(d<64){
        for(uint j=0;j<8 && begin+j<length;j++){
            uint i=begin+j,step=direction==0?i:length-1-i;
            uint row=TIME_AXIS?step:sequence%height,col=TIME_AXIS?sequence%width:step;
            uint pos=(batch*height+row)*width+col;
            out[(direction*positions+pos)*64+d]=half(matrix[j*64+d]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
"""#
}
