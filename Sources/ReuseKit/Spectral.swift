// SPDX-License-Identifier: Apache-2.0
// Modified for RE-USE Cleaner: the STFT size and hop are derived from the
// sample rate (as in NVIDIA's inference_chunk.py) instead of being fixed to
// 16 kHz, and the overlap-add kernel takes the hop as an input.
import Foundation
import MLX
import MLXFFT

/// STFT geometry for a sample rate. RE-USE was trained with n_fft=320 and
/// hop=40 at 8 kHz; other rates scale both values and round up to even numbers.
public struct STFTConfig: Sendable, Equatable {
    public let nFFT: Int
    public let hop: Int

    public init(nFFT: Int, hop: Int) {
        self.nFFT = nFFT
        self.hop = hop
    }

    public static func forSampleRate(_ sampleRate: Int) -> STFTConfig {
        func makeEven(_ value: Int) -> Int { value % 2 == 0 ? value : value + 1 }
        return STFTConfig(nFFT: makeEven(320 * sampleRate / 8000), hop: makeEven(40 * sampleRate / 8000))
    }

    /// The geometry used by the original 16 kHz pipeline (640 / 80).
    public static let base16k = STFTConfig.forSampleRate(16000)

    public var bins: Int { nFFT / 2 + 1 }
}

enum Spectral {
    private static let ola = MLXFast.metalKernel(name: "reuse_fp32_ola_hop",
        inputNames: ["frames", "params"], outputNames: ["out"],
        source: """
        uint tid=thread_position_in_grid.x;
        uint batches=frames_shape[0],count=frames_shape[1],width=frames_shape[2];
        uint length=params[0],hop=params[1];
        if(tid>=batches*length)return;
        uint b=tid/length,t=tid%length;
        uint first=t>=width?(t-width)/hop+1:0;
        uint last=min(t/hop,count-1);
        float value=0.0f;
        for(uint frame=first;frame<=last;frame++){
            value+=frames[b*frames_strides[0]+frame*frames_strides[1]+(t-frame*hop)*frames_strides[2]];
        }
        out[tid]=value;
        """, ensureRowContiguous: false)

    static func window(_ nFFT: Int) -> MLXArray {
        let n = MLXArray(0..<nFFT).asType(.float32)
        return 0.5 - 0.5 * cos(2 * Float.pi * n / Float(nFFT))
    }

    static func analysis(_ samples: [Float], config: STFTConfig = .base16k) throws -> (MLXArray, MLXArray) {
        let nFFT = config.nFFT, hop = config.hop
        guard samples.count > nFFT/2 + 1 else { throw ReuseError.invalid("音声が短すぎます。") }
        let pad = nFFT/2
        let padded = Array(samples[1...pad].reversed()) + samples
            + Array(samples[(samples.count-pad-1)..<(samples.count-1)].reversed())
        let count = (padded.count-nFFT)/hop+1
        let frames = asStrided(MLXArray(padded), [count, nFFT], strides: [hop, 1])
        let spectrum = rfft(frames * window(nFFT).reshaped(1, nFFT), n: nFFT, axis: -1)
        let re = spectrum.realPart(), im = spectrum.imaginaryPart()
        let mag = sqrt(re*re+im*im).log1p().T.expandedDimensions(axis: 0)
        let phase = atan2(im, re).T.expandedDimensions(axis: 0)
        return (mag, phase)
    }

    static func synthesis(_ amp: MLXArray, _ phase: MLXArray, samples: Int, config: STFTConfig = .base16k) -> [Float] {
        let nFFT = config.nFFT, hop = config.hop
        let magnitude = expm1(maximum(amp, MLXArray(Float(0))))
        // Same "strange sweep artifact" guard as NVIDIA's inference script:
        // frames whose magnitude is mostly exactly zero are silenced.
        let zeroCount = (magnitude .== 0).sum(axis: 1, keepDims: true)
        let filtered = which(zeroCount .> (amp.shape[1]/2), MLXArray(Float(0)), amp)
        let expanded = expm1(maximum(filtered, MLXArray(Float(0))))
        let spectrum = (expanded*cos(phase) + (expanded*sin(phase)).asImaginary())
            .transposed(0, 2, 1)
        let count = spectrum.shape[1]
        let win = window(nFFT).reshaped(1, 1, nFFT)
        let frames = irfft(spectrum, n: nFFT, axis: -1) * win
        let length = nFFT+hop*(count-1)
        let params = MLXArray([UInt32(length), UInt32(hop)])
        let signal = ola([frames, params], grid: (length, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [[1, length]], outputDTypes: [.float32])[0]
        let envelopeFrames = broadcast(win*win, to: [1, count, nFFT])
        let envelope = ola([envelopeFrames, params], grid: (length, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [[1, length]], outputDTypes: [.float32])[0]
        let normalized = signal/maximum(envelope, MLXArray(Float(1e-10)))
        let output = normalized[0, (nFFT/2)..<(length-nFFT/2)]
        eval(output)
        var values = output.asArray(Float.self)
        if values.count < samples { values += Array(repeating: 1e-8, count: samples-values.count) }
        return Array(values.prefix(samples))
    }
}
