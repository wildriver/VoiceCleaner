// SPDX-License-Identifier: Apache-2.0
import Foundation

/// Full-state kernels: no frame, frequency, channel, or stage removal.
enum NextKernelSource {
    static let normResidual = SpeedKernelSource.warpNorm.replacingOccurrences(
        of:"+offset[c];",with:"+offset[c]+residual[row*64+c];")
    static let foldNorm = SpeedKernelSource.warpNorm
        .replacingOccurrences(of:"input_shape",with:"x_shape")
        .replacingOccurrences(of:"values[i]=input[row*64+lane*8+i];",with:"""
        uint index=row*64+lane*8+i;
        values[i]=((float(y[index])+float(y[rows*64+index]))+float(r[index]))*4.0f+bias[lane*8+i];
        """)
        .replacingOccurrences(of:"+offset[c];",with:"+offset[c]+x[row*64+c];")

    static func decay(_ mode: ReuseOptimization) -> String {
        switch mode {
        case .poly4:
            return """
            inline float reuse_decay(float x){
                if(x>=32.0f)return 0.0f;
                float t=-x*0.0625f;
                float p=fma(t,fma(t,fma(t,fma(t,0.0416666666666667f,0.166666666666667f),0.5f),1.0f),1.0f);
                p*=p;p*=p;p*=p;p*=p;return p;
            }
            """
        case .poly6,.poly6Residual:
            return """
            inline float reuse_decay(float x){
                if(x>=16.0f)return 0.0f;
                float t=-x*0.125f;
                float p=fma(t,0.00138888888888889f,0.00833333333333333f);
                p=fma(t,p,0.0416666666666667f);p=fma(t,p,0.166666666666667f);
                p=fma(t,p,0.5f);p=fma(t,p,1.0f);p=fma(t,p,1.0f);
                p*=p;p*=p;p*=p;return p;
            }
            """
        case .chord3,.chord4,.chord4Residual:
            let coefficients = mode == .chord3
                ? [0.3068497755437399,0.06668637259146996,0.010857934162990046,0.001897062889239776]
                : [0.30685290342677396,0.0666238162656001,0.01113943607211502,0.001459171722729394,0.00021894558325524053]
            var source = """
            inline float reuse_decay(float x){
                if(x>=80.0f)return 0.0f;
                float z=-x*1.4426950408889634f;
                int power=int(floor(z));float f=z-float(power);
                float g=\(coefficients.last!)f;
            """
            for coefficient in coefficients.dropLast().reversed(){source+="\ng=fma(f,g,\(coefficient)f);"}
            source+="\nfloat p=fma(f*(f-1.0f),g,1.0f+f);return p*as_type<float>(uint(power+127)<<23);\n}\n"
            return source
        default:return ""
        }
    }
}
