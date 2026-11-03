// SPDX-License-Identifier: Apache-2.0
// Experimental kernels retain all 16 states and the full sequence context.
import Foundation

enum SpeedKernelSource {
    static let warpNorm = #"""
uint tid=thread_position_in_grid.x,lane=tid%8,row=tid/8;
uint rows=input_shape[0]*input_shape[1]*input_shape[2];
if(row>=rows)return;
float values[8],mean=0.0f;
for(uint i=0;i<8;i++){values[i]=input[row*64+lane*8+i];mean+=values[i];}
for(uint step=1;step<8;step*=2)mean+=simd_shuffle_xor(mean,step);
mean/=64.0f;
float variance=0.0f;
for(uint i=0;i<8;i++){values[i]-=mean;variance+=values[i]*values[i];}
for(uint step=1;step<8;step*=2)variance+=simd_shuffle_xor(variance,step);
float inverse=metal::precise::rsqrt(variance/64.0f+1e-5f);
for(uint i=0;i<8;i++){
    uint c=lane*8+i;
    out[row*64+c]=scale[c]*(values[i]*inverse)+offset[c];
}
"""#

    static let fusedMerge = #"""
uint tid=thread_position_in_grid.x,lane=tid%32,pos=tid/32;
uint positions=x_shape[0]*x_shape[1]*x_shape[2];
if(pos>=positions)return;
float a0=0,a1=0,b0=0,b1=0,c0=0,c1=0,d0=0,d1=0;
for(uint k=0;k<128;k+=4){
    float v[4];
    for(uint j=0;j<4;j++){
        uint channel=(k+j)%64,direction=(k+j)/64;
        v[j]=float(y[(direction*positions+pos)*64+channel])+x[pos*64+channel];
    }
    a0+=v[0]*weight[k*64+lane];a1+=v[0]*weight[k*64+lane+32];
    b0+=v[1]*weight[(k+1)*64+lane];b1+=v[1]*weight[(k+1)*64+lane+32];
    c0+=v[2]*weight[(k+2)*64+lane];c1+=v[2]*weight[(k+2)*64+lane+32];
    d0+=v[3]*weight[(k+3)*64+lane];d1+=v[3]*weight[(k+3)*64+lane+32];
}
float v0=(a0+b0)+(c0+d0)+bias[lane],v1=(a1+b1)+(c1+d1)+bias[lane+32];
float mean=simd_sum(v0+v1)/64.0f;
float variance=max(simd_sum(v0*v0+v1*v1)/64.0f-mean*mean,0.0f);
float inverse=rsqrt(variance+1e-5f);
out[pos*64+lane]=(v0-mean)*inverse*scale[lane]+offset[lane];
out[pos*64+lane+32]=(v1-mean)*inverse*scale[lane+32]+offset[lane+32];
"""#

    static func scan(_ mode: ReuseOptimization, time: Bool) -> String {
        let base = ReuseKernelSource.scan
        if mode.adjustsRate(time: time) {
            return base.replacingOccurrences(of: "state[n]=next;\n        y+=next*float(projected[bc+20+n]);",
                with: "float second=a*next+q;\n        state[n]=second;\n        y+=(next+second)*0.5f*float(projected[bc+20+n]);")
        }
        switch mode {
        case .foldUnroll:
            return base.replacingOccurrences(of:"for(uint n=0;n<16;n++)",with:"\n#pragma clang loop unroll(full)\nfor(uint n=0;n<16;n++)")
        case .foldVector: return scan(.vector4,time:time)
        case .foldHalf: return scan(.halfVector,time:time)
        case .foldHalfExp:
            return scan(.halfVector,time:time).replacingOccurrences(of:"fast::exp(float4(half4(delta)*decay_base[n]))",with:"metal::exp(half4(delta)*decay_base[n])")
        case .simd4, .simd8: return simdScan
        case .exp2:
            return base.replacingOccurrences(of: "-fast::exp(A_log[pd*16+n])", with: "-fast::exp(A_log[pd*16+n])*1.4426950408889634f")
                .replacingOccurrences(of: "fast::exp(delta*decay_base[n])", with: "fast::exp2(delta*decay_base[n])")
        case .fusedFMA:
            return base.replacingOccurrences(of: "a*state[n]+q", with: "fma(a,state[n],q)")
        case .vector4, .halfVector, .foldedVector:
            let type = mode == .halfVector ? "half4" : "float4"
            let initial = """
            \(type) state[4],decay_base[4];
            for(uint n=0;n<4;n++){
                state[n]=\(type)(0);
                decay_base[n]=\(type)(-fast::exp(float4(A_log[pd*16+n*4],A_log[pd*16+n*4+1],A_log[pd*16+n*4+2],A_log[pd*16+n*4+3])));
            }
            """
            let update = """
            for(uint n=0;n<4;n++){
                uint offset=n*4;
                \(type) a=\(type)(fast::exp(float4(\(type)(delta)*decay_base[n])));
                \(type) bb=\(type)(*((device const half4*)(projected+bc+4+offset)));
                float4 cc=float4(*((device const half4*)(projected+bc+20+offset)));
                \(type) next=a*state[n]+\(type)(injection)*bb;
                state[n]=next;
                float4 product=float4(next)*cc;
                y+=product.x;y+=product.y;y+=product.z;y+=product.w;
            }
            """
            let oldInitial = "float state[16],decay_base[16];\nfor(uint n=0;n<16;n++){state[n]=0.0f;decay_base[n]=-fast::exp(A_log[pd*16+n]);}"
            let range = base.range(of: "    for(uint n=0;n<16;n++){", options: .backwards)!
            let end = base.range(of: "    y+=input*skip;", range: range.upperBound..<base.endIndex)!
            return base.replacingCharacters(in: range.lowerBound..<end.lowerBound, with: update + "\n")
                .replacingOccurrences(of: oldInitial, with: initial)
        default: return base
        }
    }

    static let parallelConv = #"""
uint tid=thread_position_in_grid.x;
uint batches=xz_shape[0],height=xz_shape[1],width=xz_shape[2],positions=batches*height*width;
if(tid>=2*positions*256)return;
uint d=tid%256,packed=tid/256,direction=packed/positions,pos=packed%positions;
uint row=(pos/width)%height,col=pos%width;
uint step=TIME_AXIS?row:col,length=TIME_AXIS?height:width;
uint i=direction==0?step:length-1-step;
int stride=int(TIME_AXIS?width:1)*(direction==0?1:-1);
uint pd=direction*256+d;
float x0=i>=3?float(xz[(int(pos)-3*stride)*1024+direction*512+d]):0.0f;
float x1=i>=2?float(xz[(int(pos)-2*stride)*1024+direction*512+d]):0.0f;
float x2=i>=1?float(xz[(int(pos)-stride)*1024+direction*512+d]):0.0f;
float x3=xz[pos*1024+direction*512+d];
float value=float(bias[pd])+float(weight[pd*4])*x0+float(weight[pd*4+1])*x1+float(weight[pd*4+2])*x2+float(weight[pd*4+3])*x3;
half h=half(value);h=h*reuse_sigmoid(h);out[tid]=h;
"""#

    static let simdScan = #"""
uint tid=thread_position_in_grid.x,lane=tid%LANES;tid/=LANES;
uint batches=xz_shape[0],height=xz_shape[1],width=xz_shape[2],channels=256;
uint positions=batches*height*width,sequence_batch=TIME_AXIS?batches*width:batches*height;
uint length=TIME_AXIS?height:width;
if(tid>=2*sequence_batch*channels)return;
uint d=tid%channels,branch_sequence=tid/channels;
uint direction=branch_sequence/sequence_batch,sequence=branch_sequence%sequence_batch;
uint batch=TIME_AXIS?sequence/width:sequence/height,pd=direction*channels+d;
float state[STATES],decay_base[STATES];
for(uint j=0;j<STATES;j++){state[j]=0.0f;decay_base[j]=-fast::exp(A_log[pd*16+lane*STATES+j]);}
half4 dtw=half4(dt_weight[pd*4],dt_weight[pd*4+1],dt_weight[pd*4+2],dt_weight[pd*4+3]);
float bias=dt_bias[pd],skip=Dparam[pd];
for(uint i=0;i<length;i++){
    uint step=direction==0?i:length-1-i;
    uint row=TIME_AXIS?step:sequence%height,col=TIME_AXIS?sequence%width:step;
    uint pos=(batch*height+row)*width+col,packed=direction*positions+pos;
    uint index=packed*256+d,bc=packed*36;
    float projection=float(projected[bc])*float(dtw.x);
    projection+=float(projected[bc+1])*float(dtw.y);
    projection+=float(projected[bc+2])*float(dtw.z);
    projection+=float(projected[bc+3])*float(dtw.w);
    float delta=selective_scan_softplus(float(half(projection))+bias);
    float input=u[index],injection=delta*input,y=0.0f;
    for(uint j=0;j<STATES;j++){
        uint n=lane*STATES+j;
        float a=fast::exp(delta*decay_base[j]);
        float next=a*state[j]+injection*float(projected[bc+4+n]);
        state[j]=next;y+=next*float(projected[bc+20+n]);
    }
    for(uint offset=LANES/2;offset;offset/=2)y+=simd_shuffle_down(y,offset);
    if(lane==0){
        y+=input*skip;
        y*=selective_scan_silu(xz[pos*1024+direction*512+256+d]);
        out[index]=y;
    }
}
"""#
}
