// SPDX-License-Identifier: Apache-2.0
// Modified: spatially batched scan and convolution kernels.
// SpeechLens portions: Apache-2.0.
import Foundation

enum ReuseKernelSource {
    static let conv = #"""

uint tid=thread_position_in_grid.x;
uint batches=xz_shape[0], height=xz_shape[1], width=xz_shape[2], channels=256;
uint positions=batches*height*width;
uint sequence_batch=TIME_AXIS?batches*width:batches*height;
uint length=TIME_AXIS?height:width;
if(tid>=2*sequence_batch*channels)return;
uint d=tid%channels, branch_sequence=tid/channels;
uint direction=branch_sequence/sequence_batch, sequence=branch_sequence%sequence_batch;
uint batch=TIME_AXIS?sequence/width:sequence/height;
uint pd=direction*channels+d;

float w0=weight[pd*4],w1=weight[pd*4+1],w2=weight[pd*4+2],w3=weight[pd*4+3];
float bias_d=bias[pd],x_m3=0.0f,x_m2=0.0f,x_m1=0.0f;
for(uint i=0;i<length;i++){

uint step=direction==0?i:length-1-i;
uint row=TIME_AXIS?step:sequence%height;
uint col=TIME_AXIS?sequence%width:step;
uint pos=(batch*height+row)*width+col;
uint packed=direction*positions+pos;

    float current=xz[pos*1024+direction*512+d];
    float value=bias_d+w0*x_m3+w1*x_m2+w2*x_m1+w3*current;
    half h=half(value);
    h=h*reuse_sigmoid(h);
    out[packed*256+d]=h;
    x_m3=x_m2;x_m2=x_m1;x_m1=current;
}

"""#
    static let scan = #"""

uint tid=thread_position_in_grid.x;
uint batches=xz_shape[0], height=xz_shape[1], width=xz_shape[2], channels=256;
uint positions=batches*height*width;
uint sequence_batch=TIME_AXIS?batches*width:batches*height;
uint length=TIME_AXIS?height:width;
if(tid>=2*sequence_batch*channels)return;
uint d=tid%channels, branch_sequence=tid/channels;
uint direction=branch_sequence/sequence_batch, sequence=branch_sequence%sequence_batch;
uint batch=TIME_AXIS?sequence/width:sequence/height;
uint pd=direction*channels+d;

float state[16],decay_base[16];
for(uint n=0;n<16;n++){state[n]=0.0f;decay_base[n]=-fast::exp(A_log[pd*16+n]);}
half4 dtw=half4(dt_weight[pd*4],dt_weight[pd*4+1],dt_weight[pd*4+2],dt_weight[pd*4+3]);
float bias=dt_bias[pd],skip=Dparam[pd];
for(uint i=0;i<length;i++){

uint step=direction==0?i:length-1-i;
uint row=TIME_AXIS?step:sequence%height;
uint col=TIME_AXIS?sequence%width:step;
uint pos=(batch*height+row)*width+col;
uint packed=direction*positions+pos;

    uint index=packed*256+d,bc=packed*36;
    float projection=float(projected[bc])*float(dtw.x);
    projection+=float(projected[bc+1])*float(dtw.y);
    projection+=float(projected[bc+2])*float(dtw.z);
    projection+=float(projected[bc+3])*float(dtw.w);
    float delta=selective_scan_softplus(float(half(projection))+bias);
    float input=u[index],injection=delta*input,y=0.0f;
    for(uint n=0;n<16;n++){
        float a=fast::exp(delta*decay_base[n]);
        float q=injection*float(projected[bc+4+n]);
        float next=a*state[n]+q;
        state[n]=next;
        y+=next*float(projected[bc+20+n]);
    }
    y+=input*skip;
    y*=selective_scan_silu(xz[pos*1024+direction*512+256+d]);
    out[index]=y;
}

"""#
    static let scanHeader = #"""
    inline float selective_scan_sigmoid(float x) {
        return 1.0f / (1.0f + exp(-x));
    }

    inline float selective_scan_silu(float x) {
        return x * selective_scan_sigmoid(x);
    }

    // Softplus uses the same expression as MLX's eager graph; selective-scan
    // parity tests cover the fused implementation.
    // That graph is 3 separate dispatches per Mamba call (exp, +1, log);
    // folding it in here lets the scan kernel consume the raw `dtProj(dt)`
    // output and do the transform per thread inline with the scan work.
    inline float selective_scan_softplus(float x) {
        return log(1.0f + exp(x));
    }
"""#
}
