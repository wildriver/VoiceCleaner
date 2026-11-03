// SPDX-License-Identifier: Apache-2.0
import MLX

/// Experimental full-resolution residual path with pooled Mamba corrections.
/// This is an architectural approximation, never the default denoising path.
enum FeatureReduction {
    static let join = MLXFast.metalKernel(name: "reuse_feature_inner_join",inputNames:["y","x","steps"],outputNames:["out"],source:"""
        uint tid=thread_position_in_grid.x,T=x_shape[1],F=x_shape[2],B=x_shape[0];
        if(tid>=B*T*F*128)return;
        uint channel=tid%128,c=channel%64,direction=channel/64;
        uint f=(tid/128)%F,t=(tid/(128*F))%T,b=tid/(128*F*T);
        uint nt=y_shape[2],nf=y_shape[3];
        float shift=steps[2]==1?0.0f:0.5f;
        float rt=clamp((float(t)+shift)/float(steps[0])-shift,0.0f,float(nt-1));
        float rf=clamp((float(f)+shift)/float(steps[1])-shift,0.0f,float(nf-1));
        uint t0=uint(rt),f0=uint(rf),t1=min(t0+1,nt-1),f1=min(f0+1,nf-1);
        float at=rt-float(t0),af=rf-float(f0);
        uint base=(direction*B+b)*nt*nf*64;
        float q00=y[base+(t0*nf+f0)*64+c],q01=y[base+(t0*nf+f1)*64+c];
        float q10=y[base+(t1*nf+f0)*64+c],q11=y[base+(t1*nf+f1)*64+c];
        float first=q00*(1-af)+q01*af,last=q10*(1-af)+q11*af;
        out[tid]=first*(1-at)+last*at+x[((b*T+t)*F+f)*64+c];
        """)

    static func pooled(_ x: MLXArray,time: Int,frequency: Int,sampleOnly: Bool = false) -> MLXArray {
        let steps = MLXArray([UInt32(time),UInt32(frequency),sampleOnly ? 1 : 0])
        let shape = [x.shape[0],(x.shape[1]+time-1)/time,(x.shape[2]+frequency-1)/frequency,64]
        return pool([x,steps],grid:(shape.reduce(1,*),1,1),threadGroup:(256,1,1),outputShapes:[shape],outputDTypes:[.float32])[0]
    }
    static let pool = MLXFast.metalKernel(name: "reuse_feature_pool", inputNames: ["x","steps"], outputNames: ["out"], source: """
        uint tid=thread_position_in_grid.x,T=x_shape[1],F=x_shape[2],C=64;
        uint st=steps[0],sf=steps[1],nt=(T+st-1)/st,nf=(F+sf-1)/sf;
        if(tid>=x_shape[0]*nt*nf*C)return;
        uint c=tid%C,f=(tid/C)%nf,t=(tid/(C*nf))%nt,b=tid/(C*nf*nt);
        if(steps[2]==1){out[tid]=x[((b*T+t*st)*F+f*sf)*C+c];return;}
        float sum=0;
        for(uint i=0;i<st;i++)for(uint j=0;j<sf;j++){
            uint row=min(t*st+i,T-1),col=min(f*sf+j,F-1);
            sum+=x[((b*T+row)*F+col)*C+c];
        }
        out[tid]=sum/float(st*sf);
        """)
    static let restore = MLXFast.metalKernel(name: "reuse_feature_restore", inputNames: ["small","before","x","steps"], outputNames: ["out"], source: """
        uint tid=thread_position_in_grid.x,T=x_shape[1],F=x_shape[2],C=64;
        if(tid>=x_shape[0]*T*F*C)return;
        uint c=tid%C,f=(tid/C)%F,t=(tid/(C*F))%T,b=tid/(C*F*T);
        uint nt=small_shape[1],nf=small_shape[2];
        float rt=clamp((float(t)+0.5f)/float(steps[0])-0.5f,0.0f,float(nt-1));
        float rf=clamp((float(f)+0.5f)/float(steps[1])-0.5f,0.0f,float(nf-1));
        uint t0=uint(rt),f0=uint(rf),t1=min(t0+1,nt-1),f1=min(f0+1,nf-1);
        float at=rt-float(t0),af=rf-float(f0);
        uint i00=((b*nt+t0)*nf+f0)*C+c,i01=((b*nt+t0)*nf+f1)*C+c;
        uint i10=((b*nt+t1)*nf+f0)*C+c,i11=((b*nt+t1)*nf+f1)*C+c;
        float q00=small[i00]-before[i00],q01=small[i01]-before[i01];
        float q10=small[i10]-before[i10],q11=small[i11]-before[i11];
        float first=q00*(1-af)+q01*af,last=q10*(1-af)+q11*af;
        out[tid]=x[tid]+first*(1-at)+last*at;
        """)

    static func block(_ x: MLXArray, block: TFMambaBlock, time: Int, frequency: Int) -> MLXArray {
        let steps = MLXArray([UInt32(time),UInt32(frequency),0])
        let shape = [x.shape[0],(x.shape[1]+time-1)/time,(x.shape[2]+frequency-1)/frequency,64]
        let small = pool([x,steps],grid:(shape.reduce(1,*),1,1),threadGroup:(256,1,1),
            outputShapes:[shape],outputDTypes:[.float32])[0]
        let output = block(small)
        return restore([output,small,x,steps],grid:(x.size,1,1),threadGroup:(256,1,1),
            outputShapes:[x.shape],outputDTypes:[.float32])[0]
    }
}
