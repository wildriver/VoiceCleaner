// SPDX-License-Identifier: Apache-2.0
// Modified: spatial batching, mixed precision and fused-kernel selection.
// Original scan and convolution: SpeechLens, Apache-2.0.
import Foundation
import MLX
import MLXNN

public enum ReuseOptimization: String, Codable, Sendable, CaseIterable {
    case baseline, splitGemm, paddedGemm, parallelConv, simd4, simd8, exp2, fusedFMA, vector4, halfVector, halfMerge, folded, foldedVector, fusedMerge, staged2, staged5, staged10, warpNorm, fastNonlinear, warpFast
    case coarseTime, coarseFreq, coarse2D, coarseMiddle
    case innerTime, innerFreq, inner2D, innerTimeRate, inner2DRate
    case orthogonal, orthogonalMean
    case residualFuse, foldSeparate, poly4, poly6, chord3, chord4, poly6Residual, chord4Residual
    case fusedCore, fusedCoreFold, tg32, tg64, tg128, tg512
    case fused16, fused16Fold, fused32, fused32Fold, fused16UnrollFold
    case nax16Fold, nax32Fold
    case foldUnroll, foldVector, foldHalf, foldHalfExp, foldPadded, foldTG64, foldParallel
    case naxDirect16, naxDirect32, naxDirect16Unroll, naxDirect32Unroll
    case naxFull16, naxFull16Unroll
    case naxShared32, naxFullShared16
    case foldShared
    case metalFull8, metalFull16, metalFull16Shared
    static var standard: Self { .baseline }
    /// Use the fused cooperative-tensor path only on the tested GPU family.
    public static var recommended: Self {
        #if os(iOS) && !targetEnvironment(simulator)
        if #available(iOS 26.2, *), GPU.deviceInfo().architecture == "applegpu_g18p" {
            return .naxFullShared16
        }
        #endif
        return .warpFast
    }
    var isCoarse: Bool { [.coarseTime,.coarseFreq,.coarse2D,.coarseMiddle].contains(self) }
    var isInner: Bool { [.innerTime,.innerFreq,.inner2D,.innerTimeRate,.inner2DRate,.orthogonal,.orthogonalMean].contains(self) }
    var portableFull: Bool { [.metalFull8,.metalFull16,.metalFull16Shared].contains(self) }
    var sharedCNN: Bool { [.foldShared,.metalFull16Shared,.naxShared32,.naxFullShared16].contains(self) }
    var fullInput: Bool { portableFull || [.naxFull16,.naxFull16Unroll,.naxFullShared16].contains(self) }
    var directNAX: Bool { (fullInput && !portableFull) || [.naxDirect16,.naxDirect32,.naxDirect16Unroll,.naxDirect32Unroll,.naxShared32].contains(self) }
    var unrollCore: Bool { [.metalFull16Shared,.fused16UnrollFold,.naxDirect16Unroll,.naxDirect32Unroll,.naxFull16Unroll,.naxShared32,.naxFullShared16].contains(self) }
    var usesNAX: Bool { directNAX || [.nax16Fold,.nax32Fold].contains(self) }
    var isCore: Bool { portableFull || usesNAX || [.fusedCore,.fusedCoreFold,.fused16,.fused16Fold,.fused32,.fused32Fold,.fused16UnrollFold].contains(self) }
    var coreFold: Bool { portableFull || usesNAX || [.fusedCoreFold,.fused16Fold,.fused32Fold,.fused16UnrollFold].contains(self) }
    var coreRows: Int { [.metalFull8,.fusedCore,.fusedCoreFold].contains(self) ? 8 : [.fused32,.fused32Fold,.nax32Fold,.naxDirect32,.naxDirect32Unroll,.naxShared32].contains(self) ? 32 : 16 }
    var foldsOutput: Bool { coreFold || [.foldShared,.foldSeparate,.foldUnroll,.foldVector,.foldHalf,.foldHalfExp,.foldPadded,.foldTG64,.foldParallel].contains(self) }
    var isNext: Bool { isCore || foldsOutput || [.residualFuse,.poly4,.poly6,.chord3,.chord4,.poly6Residual,.chord4Residual,.tg32,.tg64,.tg128,.tg512].contains(self) }
    var fusesResidual: Bool { isCore || foldsOutput || [.residualFuse,.poly6Residual,.chord4Residual].contains(self) }
    var threadCount: Int { self == .tg32 ? 32 : [.tg64,.foldTG64].contains(self) ? 64 : self == .tg128 ? 128 : self == .tg512 ? 512 : 256 }
    var usesTransferPipeline: Bool { self == .warpFast || isNext }
    var usesWarpNorm: Bool { self == .warpNorm || self == .warpFast || isCoarse || isInner || isNext }
    var usesFastNonlinear: Bool { self == .fastNonlinear || self == .warpFast || isCoarse || isInner || isNext }
    func reduction(time: Bool) -> (Int,Int) {
        if self == .orthogonal || self == .orthogonalMean { return time ? (1,2) : (2,1) }
        return self == .innerFreq ? (1,2) : [.innerTime,.innerTimeRate].contains(self) ? (2,1) : (2,2)
    }
    func adjustsRate(time: Bool) -> Bool { self == .inner2DRate || (self == .innerTimeRate && time) }
}
typealias InferenceMode = ReuseOptimization

final class Mamba: Module {
    let inProj = Linear(64, 512, bias: false)
    let conv1d = Conv1d(inputChannels: 256, outputChannels: 256, kernelSize: 4,
                      padding: 3, groups: 256, bias: true)
    let xProj = Linear(256, 36, bias: false)
    let dtProj = Linear(4, 256, bias: true)
    @ParameterInfo var A_log = MLX.zeros([256, 16])
    @ParameterInfo var D = MLX.ones([256])
    let outProj = Linear(256, 64, bias: false)
}

private final class PreparedDirection {
    let input, conv, bias, projected, output, dt, deltaBias, aLog, skip: MLXArray
    let tileProjection,tileOutput: MLXArray
    init(_ fw: Mamba, _ bw: Mamba, adjustRate: Bool = false) {
        input = concatenated([fw.inProj.weight, bw.inProj.weight], axis: 0)
        let originalConv = concatenated([fw.conv1d.weight, bw.conv1d.weight], axis: 0)
        if adjustRate {
            let w = originalConv.asType(.float32).reshaped(512,4)
            conv = stacked([MLXArray.zeros([512]),w[0...,0]*0.5,
                w[0...,0]*0.5+w[0...,1]+w[0...,2]*0.5,w[0...,2]*0.5+w[0...,3]],axis:1)
                .reshaped(originalConv.shape).asType(.float16)
        } else { conv = originalConv }
        bias = concatenated([fw.conv1d.bias!, bw.conv1d.bias!], axis: 0)
        projected = stacked([fw.xProj.weight, bw.xProj.weight])
        output = stacked([fw.outProj.weight, bw.outProj.weight])
        dt = concatenated([fw.dtProj.weight, bw.dtProj.weight], axis: 0)
        deltaBias = concatenated([fw.dtProj.bias!, bw.dtProj.bias!], axis: 0)
        aLog = concatenated([fw.A_log, bw.A_log], axis: 0)
        skip = concatenated([fw.D, bw.D], axis: 0)
        tileProjection = contiguous(padded(projected,widths:[IntOrPair(0),IntOrPair((0,28)),IntOrPair(0)]).swappedAxes(-1,-2))
        tileOutput = contiguous(output.swappedAxes(-1,-2))
        eval(input, conv, bias, projected, output, dt, deltaBias, aLog, skip)
        eval(tileProjection,tileOutput)
    }
}

private final class SpatialKernels {
    let conv, scan, join: MLXFast.MLXFastKernel
    let fold: MLXFast.MLXFastKernel
    let fusedMerge: MLXFast.MLXFastKernel
    let warpNorm: MLXFast.MLXFastKernel
    let normResidual, foldNorm: MLXFast.MLXFastKernel
    let fusedCore: MLXFast.MLXFastKernel
    init(time: Bool, mode: ReuseOptimization) {
        let axis = "constant bool TIME_AXIS=\(time ? "true" : "false");\n"
        let nonlinearFast = mode.usesFastNonlinear
        let exponential = nonlinearFast ? "metal::fast::exp(float(metal::abs(x)))" : "metal::precise::exp(metal::abs(x))"
        let sigmoid = "template <typename T> inline T reuse_sigmoid(T x){auto y=1/(1+\(exponential));return x<0?y:1-y;}\n"
        let suffix = time ? "time" : "freq"
        conv = MLXFast.metalKernel(name: "reuse_spatial_conv_\(suffix)_\(mode.rawValue)",
            inputNames: ["xz", "weight", "bias"], outputNames: ["out"],
            source: [.parallelConv,.foldParallel].contains(mode) ? SpeedKernelSource.parallelConv : ReuseKernelSource.conv, header: axis + sigmoid)
        let lanes = mode == .simd4 ? 4 : mode == .simd8 ? 8 : 1
        var scanSource = SpeedKernelSource.scan(mode, time: time)
        let decayHeader = NextKernelSource.decay(mode)
        if !decayHeader.isEmpty {
            scanSource = scanSource.replacingOccurrences(of: "fast::exp(delta*decay_base[n])",with:"reuse_decay(-delta*decay_base[n])")
        }
        let projectedWidth = [.paddedGemm,.foldPadded].contains(mode) ? 64 : 36
        let scanHeader = nonlinearFast ? ReuseKernelSource.scanHeader.replacingOccurrences(of: " exp(", with: " fast::exp(")
            .replacingOccurrences(of: "return log(", with: "return fast::log(") : ReuseKernelSource.scanHeader
        scan = MLXFast.metalKernel(name: "reuse_spatial_scan_\(suffix)_\(mode.rawValue)",
            inputNames: ["xz", "A_log", "projected", "dt_weight", "dt_bias", "u", "Dparam"],
            outputNames: ["out"], source: scanSource.replacingOccurrences(of: "packed*36", with: "packed*\(projectedWidth)"),
            header: axis + "constant uint LANES=\(lanes),STATES=\(16/lanes);\n" + scanHeader + decayHeader)
        join = MLXFast.metalKernel(name: "reuse_spatial_join", inputNames: ["y", "x"],
            outputNames: ["out"], source: """
            uint index=thread_position_in_grid.x;
            uint positions=x_shape[0]*x_shape[1]*x_shape[2];
            if(index>=positions*128)return;
            uint pos=index/128,c=index%128,d=c%64,direction=c/64;
            out[index]=float(y[(direction*positions+pos)*64+d])+x[pos*64+d];
            """)
        fold = MLXFast.metalKernel(name: "reuse_spatial_fold", inputNames: ["y", "x"], outputNames: ["out"], source: """
            uint tid=thread_position_in_grid.x,positions=x_shape[0]*x_shape[1]*x_shape[2];
            if(tid>=positions*576)return;
            uint pos=tid/576,c=tid%576;
            out[tid]=c<512?half(y[((c/256)*positions+pos)*256+c%256]):half(x[pos*64+c-512]);
            """)
        fusedMerge = MLXFast.metalKernel(name: "reuse_fused_merge", inputNames: ["y","x","weight","bias","scale","offset"],
            outputNames: ["out"], source: SpeedKernelSource.fusedMerge)
        warpNorm = MLXFast.metalKernel(name: "reuse_warp_norm64", inputNames: ["input","scale","offset"],
            outputNames: ["out"], source: SpeedKernelSource.warpNorm)
        normResidual = MLXFast.metalKernel(name:"reuse_norm_residual64",inputNames:["input","scale","offset","residual"],
            outputNames:["out"],source:NextKernelSource.normResidual)
        foldNorm = MLXFast.metalKernel(name:"reuse_fold_norm64",inputNames:["y","r","x","bias","scale","offset"],
            outputNames:["out"],source:NextKernelSource.foldNorm)
        let optimizedCore = mode.isCore && mode.coreRows != 8
        fusedCore = MLXFast.metalKernel(name:"reuse_fused_core_\(suffix)_\(mode.rawValue)",
            inputNames:[mode.fullInput ? "x" : "xz","weight","bias","A_log","dt_weight","dt_bias","Dparam","x_weight","out_weight"] + (mode.fullInput ? ["input_weight"] : []),
            outputNames:["out"],source:mode.portableFull ? FusedMambaSource.portableSource(unroll:mode.unrollCore) : mode.fullInput ? FusedMambaSource.fullSource(unroll:mode.unrollCore) : mode.directNAX ? FusedMambaSource.directSource(unroll:mode.unrollCore) : optimizedCore ? FusedMambaSource.configured(unroll:mode.unrollCore,nax:mode.usesNAX) : FusedMambaSource.source,
            header:"#include <metal_simdgroup_matrix>\n" + (mode.usesNAX ? "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n" : "") + axis + sigmoid + scanHeader + "\n" + (mode.portableFull ? FusedMambaSource.portableHelper(rows:mode.coreRows) + "\n" + FusedMambaSource.portableInput : (mode.fullInput ? FusedMambaSource.inputHelper + "\n" : "") + (optimizedCore ? (mode.directNAX ? FusedMambaSource.directHelper(rows:mode.coreRows) : mode.usesNAX ? FusedMambaSource.naxHelper(rows:mode.coreRows) : FusedMambaSource.helper(rows:mode.coreRows)) : "")))
    }
}

final class BidirectionalMamba: Module, UnaryLayer {
    let forward = Mamba()
    let backward = Mamba()
    let outputProj = Linear(128, 64)
    let norm = LayerNorm(dimensions: 64)
    private let time: Bool
    private let mode: ReuseOptimization
    private let kernels: SpatialKernels
    private var prepared: PreparedDirection?
    private var mergeWeight: MLXArray?
    private var foldedWeight: MLXArray?
    private var fusedWeight: MLXArray?
    private var separateWeight: MLXArray?
    private var residualWeight: MLXArray?
    private var tileSeparate: MLXArray?
    private var tileInput: MLXArray?

    init(time: Bool, mode: ReuseOptimization = .baseline) {
        self.time = time
        self.mode = mode
        self.kernels = SpatialKernels(time: time, mode: mode)
        super.init()
    }

    func prepare() {
        prepared = PreparedDirection(forward, backward, adjustRate: mode.adjustsRate(time: time))
        if mode.fullInput { tileInput=contiguous(prepared!.input.T);eval(tileInput!) }
        if mode == .halfMerge { mergeWeight = outputProj.weight.asType(.float16); eval(mergeWeight!) }
        if mode == .fusedMerge { fusedWeight = contiguous(outputProj.weight.T); eval(fusedWeight!) }
        if mode == .folded || mode == .foldedVector {
            let left = outputProj.weight[0..., ..<64], right = outputProj.weight[0..., 64...]
            foldedWeight = (concatenated([matmul(left, forward.outProj.weight.asType(.float32)),
                matmul(right, backward.outProj.weight.asType(.float32)), left + right], axis: 1) / 4).asType(.float16)
            eval(foldedWeight!)
        }
        if mode.foldsOutput {
            let left = outputProj.weight[0...,..<64],right = outputProj.weight[0...,64...]
            separateWeight = (stacked([matmul(left,forward.outProj.weight.asType(.float32)),
                matmul(right,backward.outProj.weight.asType(.float32))])/4).asType(.float16)
            residualWeight = ((left+right)/4).asType(.float16)
            tileSeparate = contiguous(separateWeight!.swappedAxes(-1,-2))
            eval(separateWeight!,residualWeight!)
            eval(tileSeparate!)
        }
    }

    func callAsFunction(_ original: MLXArray) -> MLXArray {
        let reduction = mode.reduction(time: time)
        let sampleOnly = mode == .orthogonal
        let x = mode.isInner ? FeatureReduction.pooled(original,time: reduction.0,frequency: reduction.1,sampleOnly: sampleOnly) : original
        let w = prepared!
        let b = x.shape[0], t = x.shape[1], f = x.shape[2], c = x.shape[3]
        let p = b * t * f, d = 256
        let xz = mode.fullInput ? x : matmul(x.asType(.float16), w.input.T)
        let sequences = time ? b*f : b*t
        let grid = (2*sequences*d, 1, 1)
        let projectedOut: MLXArray
        if mode.isCore {
            projectedOut = kernels.fusedCore([xz,w.conv,w.bias,w.aLog,w.dt,w.deltaBias,w.skip,w.tileProjection,tileSeparate ?? w.tileOutput] + (mode.fullInput ? [tileInput!] : []),
                grid:grid,threadGroup:(256,1,1),outputShapes:[[2,b,t,f,c]],outputDTypes:[.float16])[0]
            if let residualWeight {
                let residual = matmul(x.asType(.float16),residualWeight.T)
                return kernels.foldNorm([projectedOut,residual,x,outputProj.bias!,norm.weight!,norm.bias!],
                    grid:(p*8,1,1),threadGroup:(128,1,1),outputShapes:[x.shape],outputDTypes:[.float32])[0]
            }
        } else {
        let u = kernels.conv([xz, w.conv, w.bias], grid: [.parallelConv,.foldParallel].contains(mode) ? (2*p*d, 1, 1) : grid, threadGroup: (mode.threadCount, 1, 1),
            outputShapes: [[2, b, t, f, d]], outputDTypes: [.float16])[0]
        let projected: MLXArray
        if mode == .splitGemm {
            projected = stacked([matmul(u[0].reshaped(p, d), forward.xProj.weight.T),
                                 matmul(u[1].reshaped(p, d), backward.xProj.weight.T)])
                .reshaped(2, b, t, f, 36)
        } else if [.paddedGemm,.foldPadded].contains(mode) {
            let weight = padded(w.projected, widths: [IntOrPair(0), IntOrPair((0, 28)), IntOrPair(0)])
            projected = matmul(u.reshaped(2, p, d), weight.swappedAxes(-1, -2)).reshaped(2, b, t, f, 64)
        } else {
            projected = matmul(u.reshaped(2, p, d), w.projected.swappedAxes(-1, -2)).reshaped(2, b, t, f, 36)
        }
        let lanes = mode == .simd4 ? 4 : mode == .simd8 ? 8 : 1
        let y = kernels.scan([xz, w.aLog, projected, w.dt, w.deltaBias, u, w.skip],
            grid: (grid.0*lanes, 1, 1), threadGroup: (mode.threadCount, 1, 1), outputShapes: [u.shape], outputDTypes: [.float16])[0]
        if let separateWeight,let residualWeight {
            let projected = matmul(y.reshaped(2,p,d),separateWeight.swappedAxes(-1,-2)).reshaped(2,b,t,f,c)
            let residual = matmul(x.asType(.float16),residualWeight.T)
            return kernels.foldNorm([projected,residual,x,outputProj.bias!,norm.weight!,norm.bias!],
                grid:(p*8,1,1),threadGroup:(128,1,1),outputShapes:[x.shape],outputDTypes:[.float32])[0]
        }
        if let foldedWeight {
            let joined = kernels.fold([y,x], grid: (p*576,1,1), threadGroup: (256,1,1),
                outputShapes: [[b,t,f,576]], outputDTypes: [.float16])[0]
            let value = matmul(joined, foldedWeight.T).asType(.float32) * 4 + outputProj.bias!
            return norm(value)
        }
        if mode == .splitGemm {
            projectedOut = stacked([matmul(y[0].reshaped(p, d), forward.outProj.weight.T),
                                    matmul(y[1].reshaped(p, d), backward.outProj.weight.T)])
                .reshaped(2, b, t, f, c)
        } else {
            projectedOut = matmul(y.reshaped(2, p, d), w.output.swappedAxes(-1, -2)).reshaped(2, b, t, f, c)
        }
        }
        if let fusedWeight {
            return kernels.fusedMerge([projectedOut,x,fusedWeight,outputProj.bias!,norm.weight!,norm.bias!],
                grid: (p*32,1,1), threadGroup: (128,1,1), outputShapes: [[b,t,f,64]], outputDTypes: [.float32])[0]
        }
        let finalShape = mode.isInner ? original.shape : x.shape
        let positions = finalShape[0]*finalShape[1]*finalShape[2]
        let merged: MLXArray
        if mode.isInner {
            merged = FeatureReduction.join([projectedOut,original,MLXArray([UInt32(reduction.0),UInt32(reduction.1),sampleOnly ? 1 : 0])],
                grid:(positions*128,1,1),threadGroup:(256,1,1),
                outputShapes:[[finalShape[0],finalShape[1],finalShape[2],128]],outputDTypes:[.float32])[0]
        } else {
            merged = kernels.join([projectedOut, x], grid: (p*128, 1, 1), threadGroup: (256, 1, 1),
                outputShapes: [[b, t, f, 128]], outputDTypes: [mode == .halfMerge ? .float16 : .float32])[0]
        }
        if mode.usesWarpNorm {
            let value = outputProj(merged)
            if mode.fusesResidual {
                return kernels.normResidual([value,norm.weight!,norm.bias!,original],grid:(positions*8,1,1),
                    threadGroup:(128,1,1),outputShapes:[finalShape],outputDTypes:[.float32])[0]
            }
            return kernels.warpNorm([value,norm.weight!,norm.bias!], grid: (positions*8,1,1), threadGroup: (128,1,1),
                outputShapes: [finalShape], outputDTypes: [.float32])[0]
        }
        if let mergeWeight { return norm(matmul(merged, mergeWeight.T).asType(.float32) + outputProj.bias!) }
        return norm(outputProj(merged))
    }

    func profileParts(_ x: MLXArray) -> [ReuseDiagnosticRow] {
        let w = prepared!, b = x.shape[0], t = x.shape[1], f = x.shape[2], p = b*t*f
        let grid = (2*(time ? b*f : b*t)*256, 1, 1)
        var rows = [ReuseDiagnosticRow]()
        func measure(_ name: String, _ body: () -> MLXArray) -> MLXArray {
            var output = body(); eval(output)
            var seconds = [Double]()
            for _ in 0..<7 {
                let began = ProcessInfo.processInfo.systemUptime
                output = body(); eval(output)
                seconds.append(ProcessInfo.processInfo.systemUptime - began)
            }
            rows.append(ReuseDiagnosticRow(part: "\(time ? "time" : "freq")_\(name)", variant: mode.rawValue,
                seconds: seconds, relativeRMSE: 0, maximumError: 0))
            return output
        }
        let xz = measure("in_gemm") { matmul(x.asType(.float16), w.input.T) }
        let u = measure("conv") { kernels.conv([xz, w.conv, w.bias], grid: grid, threadGroup: (256,1,1),
            outputShapes: [[2,b,t,f,256]], outputDTypes: [.float16])[0] }
        let projection = measure("x_gemm") { matmul(u.reshaped(2,p,256), w.projected.swappedAxes(-1,-2)).reshaped(2,b,t,f,36) }
        let y = measure("scan") { kernels.scan([xz,w.aLog,projection,w.dt,w.deltaBias,u,w.skip],
            grid: grid, threadGroup: (256,1,1), outputShapes: [u.shape], outputDTypes: [.float16])[0] }
        let out = measure("out_gemm") { matmul(y.reshaped(2,p,256), w.output.swappedAxes(-1,-2)).reshaped(2,b,t,f,64) }
        let joined = measure("join") { kernels.join([out,x], grid: (p*128,1,1), threadGroup: (256,1,1),
                outputShapes: [[b,t,f,128]], outputDTypes: [.float32])[0] }
        let projected = measure("merge_gemm") { outputProj(joined) }
        _ = measure("norm") { norm(projected) }
        return rows
    }
}

final class TFMambaBlock: Module, UnaryLayer {
    let timeMamba: BidirectionalMamba
    let freqMamba: BidirectionalMamba
    private let fused: Bool
    init(hidFeature: Int, mode: InferenceMode = .standard) {
        timeMamba = BidirectionalMamba(time: true, mode: mode)
        freqMamba = BidirectionalMamba(time: false, mode: mode)
        fused = mode.fusesResidual
        super.init()
    }
    func prepare() { timeMamba.prepare(); freqMamba.prepare() }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let time = timeMamba(x)
        let y = fused ? time : time+x
        let frequency = freqMamba(y)
        return fused ? frequency : frequency+y
    }
}
