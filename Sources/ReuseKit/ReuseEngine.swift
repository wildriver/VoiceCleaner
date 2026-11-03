// SPDX-License-Identifier: Apache-2.0
// Modified for RE-USE Cleaner: added enhanceWindow(_:sampleRate:) and prepare().
import Foundation
import MLX
import MLXNN
import CoreML

public enum ReuseMode: String, Codable, Sendable, CaseIterable {
    case mlx, hybrid
    public var title: String { self == .mlx ? "MLX GPU" : "ANE + GPU" }
}

public struct ReuseMetrics: Encodable, Sendable {
    public let optimization: ReuseOptimization
    public let pipelinedTransfers: Bool
    public let cacheLimitBytes = Memory.cacheLimit
    public let metalBufferOps = ProcessInfo.processInfo.environment["MLX_MAX_OPS_PER_BUFFER"] ?? "default"
    public let metalBufferMB = ProcessInfo.processInfo.environment["MLX_MAX_MB_PER_BUFFER"] ?? "default"
    public let mode: ReuseMode
    public let inputSamples: Int
    public let sampleRate: Int
    public let processingSeconds: Double
    public let modelLoadSeconds: Double
    public let mlxPeakBytes: Int
    public let aneWindows: Int
    public let mlxWindows: Int
    public let zeroWindows: Int
    public let skippedTinyTailSamples: Int
    public let thermalStateBefore: Int
    public let thermalStateAfter: Int
    public let operatingSystem: String
    public let physicalMemory: UInt64
    public let inputSHA256: String
    public let outputSHA256: String
    public let originalStages = 30
    public let recurrentStates = 16
    public let chunkSeconds = 5
    public let overlap = 0.05
    public let mappedModelSHA256 = "c38f2a9d48024cb79f98c59ae81215db7c3ccd22daae1b3c832bb23b1b229067"
    public let modelRevision = "31f82f2efad1095034492f236e8dac9239ba713b"
    public let mlxSwiftVersion = "0.33.0"
    public let lowPowerModeEnabled = ProcessInfo.processInfo.isLowPowerModeEnabled
    public let inputSeconds: Double
    public let rtf: Double
}

public struct ReuseResult: Sendable {
    public let samples: [Float]
    public let metrics: ReuseMetrics
}

/// All non-Sendable MLX arrays, compiled closures and Core ML objects stay here.
public actor ReuseEngine {
    private let pack: ReuseModelPack
    private let cache: URL
    private let optimization: ReuseOptimization
    private var model: SEMamba?
    private var cnn: CNNBank?
    private var forward: (([MLXArray]) -> [MLXArray])?
    private var trunk: ((MLXArray) -> MLXArray)?
    private var validated = false

    public init(modelPack: ReuseModelPack, cacheDirectory: URL, optimization: ReuseOptimization = .recommended) {
        self.pack = modelPack; self.cache = cacheDirectory; self.optimization = optimization
    }

    private func load() throws {
        if model != nil { return }
        if !validated { try pack.validate(); validated = true }
        let defaultCacheMB = ProcessInfo.processInfo.physicalMemory >= 10*1024*1024*1024 ? 1024 : 256
        let cacheMB = ProcessInfo.processInfo.environment["REUSE_CACHE_MB"].flatMap(Int.init) ?? defaultCacheMB
        Memory.cacheLimit = max(1,min(cacheMB,2048))*1024*1024
        var weights = try loadArrays(url: pack.weightsURL)
        guard weights.count == 1416 else { throw ReuseError.invalid("重みの数が一致しません。") }
        for (name, value) in weights {
            let branch = name.hasPrefix("tfMamba.") && (name.contains(".forward.") || name.contains(".backward."))
            if branch && !name.hasSuffix(".A_log") && !name.hasSuffix(".D") && !name.hasSuffix(".dtProj.bias") {
                weights[name] = value.asType(.float16)
            }
        }
        let loaded = SEMamba(mode: optimization)
        try loaded.update(parameters: ModuleParameters.unflattened(weights), verify: [.noUnusedKeys, .allModelKeysSet])
        let parameters = Dictionary(uniqueKeysWithValues: loaded.parameters().flattened())
        guard parameters.count == weights.count,
              weights.allSatisfy({ parameters[$0.key]?.shape == $0.value.shape && parameters[$0.key]?.dtype == $0.value.dtype }) else {
            throw ReuseError.invalid("重みの割り当てを検証できませんでした。")
        }
        loaded.train(false)
        eval(loaded.parameters())
        for block in loaded.tfMamba { block.prepare() }
        model = loaded
        forward = compile(inputs: [loaded]) { (inputs: [MLXArray]) -> [MLXArray] in
            let (amp, phase) = loaded(inputs[0], inputs[1]); return [amp, phase]
        }
        let selectedOptimization = optimization
        trunk = compile(inputs: [loaded]) { (input: MLXArray) -> MLXArray in
            var value = input
            for (index,block) in loaded.tfMamba.enumerated() {
                if selectedOptimization.isCoarse && (selectedOptimization != .coarseMiddle || (3..<27).contains(index)) {
                    value = FeatureReduction.block(value, block: block,
                        time: selectedOptimization == .coarseFreq ? 1 : 2,
                        frequency: selectedOptimization == .coarseTime ? 1 : 2)
                } else { value = block(value) }
            }
            return value
        }
        let groupSize = optimization == .staged2 ? 2 : optimization == .staged5 ? 5 : optimization == .staged10 ? 10 : 0
        if groupSize > 0 {
            let groups = stride(from: 0, to: 30, by: groupSize).map { start in
                compile(inputs: [loaded]) { (input: MLXArray) -> MLXArray in
                    var value = input
                    for index in start..<min(start+groupSize,30) { value = loaded.tfMamba[index](value) }
                    return value
                }
            }
            trunk = { input in
                var value = input
                for group in groups { value = group(value); eval(value) }
                return value
            }
        }
        cnn = try CNNBank(pack: pack, cache: cache, sharedTransfers: optimization.sharedCNN)
    }

    public func enhance(_ audio: [Float], mode: ReuseMode) throws -> ReuseResult {
        try Task.checkCancellation()
        guard audio.count > 320, audio.allSatisfy(\.isFinite) else { throw ReuseError.invalid("0.02秒より長い有限値のmono音声が必要です。") }
        return try Device.withDefaultDevice(.gpu) {
            try withError {
                let loading = ProcessInfo.processInfo.systemUptime
                try load()
                let stride = 76000, chunk = 80000, overlap = 4000
                let allStarts = Array(Swift.stride(from: 0, to: audio.count, by: stride))
                let starts = allStarts.filter { $0 == 0 || audio.count-$0 > 320 }
                var available = Set<Int>()
                if mode == .hybrid {
                    for start in starts {
                        let length = min(chunk, audio.count-start), frames = length/80+1
                        if try cnn!.prepare(frames: frames) { available.insert(frames) }
                    }
                }
                let loadSeconds = ProcessInfo.processInfo.systemUptime-loading
                Memory.peakMemory = 0
                let thermal = ProcessInfo.processInfo.thermalState.rawValue
                let began = ProcessInfo.processInfo.systemUptime
                var accumulation = [Float](repeating: 0, count: audio.count)
                var normalization = [Float](repeating: 0, count: audio.count)
                var aneCount = 0, mlxCount = 0, zeroCount = 0
                let jobs = try starts.map { start -> Job in
                    try Task.checkCancellation()
                    let values = Array(audio[start..<min(start+chunk, audio.count)])
                    if values.allSatisfy({ $0 == 0 }) { return Job(start: start, count: values.count) }
                    let (mag, phase) = try Spectral.analysis(values)
                    eval(mag, phase)
                    return Job(start: start, count: values.count, mag: mag, phase: phase,
                               ane: mode == .hybrid && available.contains(mag.shape[2]))
                }
                let work = jobs.filter { $0.mag != nil }
                let pipelineTransfers = optimization.usesTransferPipeline
                if mode == .hybrid, !work.isEmpty {
                    // A transfer queued after the next trunk would wait for that
                    // trunk on the same Metal stream. Pack CPU inputs first, then
                    // run only the Core ML predictions while the trunk is active.
                    if pipelineTransfers {
                        for job in work where job.ane {
                            try Task.checkCancellation()
                            let input = stacked([job.mag!.transposed(0,2,1),job.phase!.transposed(0,2,1)],axis:1)
                            job.encoderInput = try cnn!.encoderInput(input)
                        }
                    }
                    func encode(_ job: Job) throws -> MLXArray {
                        if job.ane {
                            if let input = job.encoderInput {
                                job.encoderInput = nil
                                return try cnn!.encodePrepared(input, frames: job.frames)
                            }
                            let input = stacked([job.mag!.transposed(0, 2, 1), job.phase!.transposed(0, 2, 1)], axis: 1)
                            return try cnn!.encode(input, frames: job.frames)
                        }
                        var input = stacked([job.mag!.transposed(0, 2, 1), job.phase!.transposed(0, 2, 1)], axis: -1)
                        input = padded(input, widths: [IntOrPair(0), IntOrPair((0, 2)), IntOrPair((0, 2)), IntOrPair(0)])
                        let result = model!.denseEncoder(input); eval(result); return result
                    }
                    func decode(_ job: Job, _ features: MLXArray) throws {
                        if job.ane {
                            let result = try cnn!.decode(features, frames: job.frames)
                            job.output = (result.0, result.1); aneCount += 1
                        } else {
                            let amp = model!.maskDecoder(features)[0..., ..<job.frames, ..<321, 0...]
                                .squeezed(axis: -1).transposed(0, 2, 1)
                            let phase = model!.phaseDecoder(features)[0..., ..<job.frames, ..<321, 0...]
                                .squeezed(axis: -1).transposed(0, 2, 1)
                            eval(amp, phase); job.output = (amp, phase); mlxCount += 1
                        }
                    }
                    var encoded = try encode(work[0])
                    var previous: (Job, MLXArray)?
                    for (index, job) in work.enumerated() {
                        try Task.checkCancellation()
                        let pendingDecode: MLMultiArray?
                        if pipelineTransfers, let (lastJob,lastValue) = previous, lastJob.ane {
                            pendingDecode = try cnn!.decoderInput(lastValue)
                        } else { pendingDecode = nil }
                        let features = trunk!(encoded)
                        asyncEval(features)
                        if let (lastJob, lastValue) = previous {
                            if let pendingDecode {
                                lastJob.output = try cnn!.decodePrepared(pendingDecode, frames: lastJob.frames)
                                aneCount += 1
                            } else { try decode(lastJob, lastValue) }
                        }
                        if index+1 < work.count { encoded = try encode(work[index+1]) }
                        eval(features)
                        previous = (job, features)
                    }
                    if let (lastJob, lastValue) = previous { try decode(lastJob, lastValue) }
                }
                for (index, job) in jobs.enumerated() {
                    try Task.checkCancellation()
                    var values: [Float]
                    if let mag = job.mag, let phase = job.phase {
                        let result: (MLXArray, MLXArray)
                        if let output = job.output { result = output }
                        else {
                            let output = forward!([mag, phase]); result = (output[0], output[1]); mlxCount += 1
                        }
                        values = Spectral.synthesis(result.0, result.1, samples: job.count)
                    } else { values = Array(repeating: 0, count: job.count); zeroCount += 1 }
                    guard values.count == job.count, values.allSatisfy(\.isFinite) else { throw ReuseError.invalid("推論出力に非有限値が含まれています。") }
                    let effective = min(overlap, job.count-1)
                    for i in values.indices {
                        var weight: Float = 1
                        if index > 0 && i < effective {
                            let angle = Float.pi*0.5*Float(i)/Float(effective)
                            let sine = Foundation.sin(angle); weight = sine*sine
                        }
                        if index+1 < jobs.count && i >= job.count-effective {
                            let angle = Float.pi*0.5*Float(i-(job.count-effective))/Float(effective)
                            let cosine = Foundation.cos(angle); weight = cosine*cosine
                        }
                        accumulation[job.start+i] += values[i]*weight
                        normalization[job.start+i] += weight
                    }
                    job.output = nil
                }
                for i in audio.indices { accumulation[i] /= max(normalization[i], 1e-8) }
                let elapsed = ProcessInfo.processInfo.systemUptime-began
                let metrics = ReuseMetrics(optimization: optimization, pipelinedTransfers: mode == .hybrid && pipelineTransfers,
                    mode: mode, inputSamples: audio.count, sampleRate: 16000,
                    processingSeconds: elapsed, modelLoadSeconds: loadSeconds, mlxPeakBytes: Memory.peakMemory,
                    aneWindows: aneCount, mlxWindows: mlxCount, zeroWindows: zeroCount,
                    skippedTinyTailSamples: allStarts.last.map { starts.contains($0) ? 0 : audio.count-$0 } ?? 0,
                    thermalStateBefore: thermal, thermalStateAfter: ProcessInfo.processInfo.thermalState.rawValue,
                    operatingSystem: ProcessInfo.processInfo.operatingSystemVersionString,
                    physicalMemory: ProcessInfo.processInfo.physicalMemory,
                    inputSHA256: Self.hash(audio), outputSHA256: Self.hash(accumulation),
                    inputSeconds: Double(audio.count)/16000, rtf: elapsed/(Double(audio.count)/16000))
                return ReuseResult(samples: accumulation, metrics: metrics)
            }
        }
    }

    /// Added for RE-USE Cleaner: enhances exactly one analysis window (for
    /// example 5 s) at the given sample rate on the MLX GPU path. The caller
    /// performs the windowed overlap-add across windows, so arbitrarily long
    /// inputs can be streamed with bounded memory.
    public func enhanceWindow(_ samples: [Float], sampleRate: Int) throws -> [Float] {
        try Task.checkCancellation()
        let config = STFTConfig.forSampleRate(sampleRate)
        guard samples.count > config.nFFT else {
            throw ReuseError.invalid("処理窓が短すぎます。")
        }
        guard samples.allSatisfy(\.isFinite) else { throw ReuseError.invalid("入力に非有限値が含まれています。") }
        if samples.allSatisfy({ $0 == 0 }) { return samples }
        return try Device.withDefaultDevice(.gpu) {
            try withError {
                try load()
                let (mag, phase) = try Spectral.analysis(samples, config: config)
                eval(mag, phase)
                let output = forward!([mag, phase])
                let values = Spectral.synthesis(output[0], output[1], samples: samples.count, config: config)
                guard values.count == samples.count, values.allSatisfy(\.isFinite) else {
                    throw ReuseError.invalid("推論出力に非有限値が含まれています。")
                }
                return values
            }
        }
    }

    /// Loads and validates the model without processing audio.
    public func prepare() throws {
        try Device.withDefaultDevice(.gpu) { try withError { try load() } }
    }

    private static func hash(_ samples: [Float]) -> String {
        samples.withUnsafeBytes { ReuseModelPack.hash(Data($0)) }
    }

    public func release() {
        forward = nil; trunk = nil; model = nil; cnn = nil
        Memory.clearCache()
    }

    func diagnoseLoaded(_ audio: [Float], progressURL: URL?, requestedVariants: [ReuseOptimization]?, kernelsOnly: Bool) throws -> ReuseDiagnostics {
        try Device.withDefaultDevice(.gpu) {
            try withError {
                try load()
                func progress(_ name: String) {
                    if let progressURL {
                        try? ReuseAudioFile.writeJSON(["stage": name,"date": Date().ISO8601Format()], to: progressURL)
                    }
                }
                let thermal = ProcessInfo.processInfo.thermalState.rawValue
                let samples = Array(audio.prefix(80000))
                let (mag, phase) = try Spectral.analysis(samples)
                eval(mag, phase)
                let frames = mag.shape[2]
                let input = stacked([mag.transposed(0, 2, 1), phase.transposed(0, 2, 1)], axis: -1)
                let features = padded(input, widths: [IntOrPair(0), IntOrPair((0, 2)), IntOrPair((0, 2)), IntOrPair(0)])
                let encoded = model!.denseEncoder(features)
                eval(encoded)
                var rows = [ReuseDiagnosticRow]()
                func measure(_ part: String, _ operation: () throws -> [MLXArray]) throws {
                    progress(part)
                    eval(try operation())
                    var values = [Double]()
                    for _ in 0..<5 {
                        try Task.checkCancellation()
                        let began = ProcessInfo.processInfo.systemUptime
                        eval(try operation())
                        values.append(ProcessInfo.processInfo.systemUptime - began)
                    }
                    rows.append(ReuseDiagnosticRow(part: part, variant: optimization.rawValue, seconds: values,
                                                   relativeRMSE: 0, maximumError: 0))
                }
                if !kernelsOnly {
                let encoder = compile(inputs: [model!]) { (x: MLXArray) in self.model!.denseEncoder(x) }
                try measure("mlx_encoder") { [encoder(features)] }
                try measure("trunk_30") { [trunk!(encoded)] }
                let trunkOutput = trunk!(encoded); eval(trunkOutput)
                // Standalone parts do not represent the fused kernel's execution.
                if !optimization.isCore {
                    rows += model!.tfMamba[0].timeMamba.profileParts(encoded)
                    let timeResult = model!.tfMamba[0].timeMamba(encoded)
                    let timeOutput = optimization.fusesResidual ? timeResult : timeResult + encoded
                    eval(timeOutput)
                    rows += model!.tfMamba[0].freqMamba.profileParts(timeOutput)
                }
                if try cnn!.prepare(frames: frames) {
                    let cnnInput = input.transposed(0, 3, 1, 2)
                    try measure("coreml_encoder_with_transfer") { [try cnn!.encode(cnnInput, frames: frames)] }
                    try measure("coreml_decoder_with_transfer") {
                        let pair = try cnn!.decode(trunkOutput, frames: frames); return [pair.0, pair.1]
                    }
                }
                }
                var blocks = [TFMambaBlock]()
                var functions = [([MLXArray]) -> [MLXArray]]()
                // Keep the default diagnostic bounded and avoid compiling A19-only
                // experiments on older GPUs. Every experiment remains explicit via CLI.
                let defaults: [ReuseOptimization] = [.baseline,.warpFast,.foldSeparate]
                    + (ReuseOptimization.recommended.usesNAX ? [.nax16Fold,.naxDirect32Unroll,.naxFullShared16] : [])
                let variants = requestedVariants ?? defaults
                for variant in variants {
                    progress("prepare_\(variant.rawValue)")
                    let block = TFMambaBlock(hidFeature: 64, mode: variant)
                    try block.update(parameters: model!.tfMamba[0].parameters(), verify: [.allModelKeysSet, .noUnusedKeys])
                    block.prepare(); blocks.append(block)
                    functions.append(compile(inputs: [block]) { (x: [MLXArray]) in [block(x[0])] })
                }
                var timings = [[Double]](repeating: [], count: functions.count)
                for index in functions.indices {
                    progress("warmup_\(variants[index].rawValue)")
                    eval(functions[index]([encoded]))
                }
                for round in 0..<7 {
                    let order = round % 2 == 0 ? Array(functions.indices) : Array(functions.indices.reversed())
                    for index in order {
                        try Task.checkCancellation()
                        let began = ProcessInfo.processInfo.systemUptime
                        eval(functions[index]([encoded]))
                        timings[index].append(ProcessInfo.processInfo.systemUptime - began)
                    }
                }
                let reference = functions[0]([encoded])[0]; eval(reference)
                let denominator = max((reference * reference).mean().item(Float.self), 1e-30)
                for index in functions.indices {
                    let candidate = functions[index]([encoded])[0]
                    let difference = candidate - reference
                    let error = sqrt((difference * difference).mean() / denominator).item(Float.self)
                    let maximum = abs(difference).max().item(Float.self)
                    rows.append(ReuseDiagnosticRow(part: "tf_block_first", variant: variants[index].rawValue,
                        seconds: timings[index], relativeRMSE: Double(error), maximumError: Double(maximum)))
                }
                progress("complete")
                return ReuseDiagnostics(date: Date().ISO8601Format(), arguments: Array(CommandLine.arguments.dropFirst()),
                    gpu: GPU.deviceInfo().architecture, thermalBefore: thermal,
                    thermalAfter: ProcessInfo.processInfo.thermalState.rawValue,
                    lowPower: ProcessInfo.processInfo.isLowPowerModeEnabled, encodedShape: encoded.shape, rows: rows)
            }
        }
    }

    private final class Job {
        let start, count: Int
        let mag, phase: MLXArray?
        let ane: Bool
        var output: (MLXArray, MLXArray)?
        var encoderInput: MLMultiArray?
        var frames: Int { mag!.shape[2] }
        init(start: Int, count: Int, mag: MLXArray? = nil, phase: MLXArray? = nil, ane: Bool = false) {
            self.start = start; self.count = count; self.mag = mag; self.phase = phase; self.ane = ane
        }
    }
}
