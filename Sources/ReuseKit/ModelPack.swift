// SPDX-License-Identifier: Apache-2.0
import Foundation
import CryptoKit
import CoreML
import MLX

public enum ReuseError: LocalizedError {
    case invalid(String)
    public var errorDescription: String? {
        switch self { case .invalid(let message): return message }
    }
}

public struct ReuseModelPack: Sendable {
    public let root: URL
    public init(root: URL) { self.root = root }
    public var weightsURL: URL { root.appendingPathComponent("model_mlx.safetensors") }

    public func validate() throws {
        let data = try Data(contentsOf: root.appendingPathComponent("manifest.json"))
        let manifest = try JSONDecoder().decode(Manifest.self, from: data)
        guard manifest.stages == 30,
              manifest.mapped_model_sha256 == "c38f2a9d48024cb79f98c59ae81215db7c3ccd22daae1b3c832bb23b1b229067",
              manifest.files["model_mlx.safetensors"] == manifest.mapped_model_sha256 else {
            throw ReuseError.invalid("モデルの識別情報が一致しません。")
        }
        for (name, expected) in manifest.files {
            let url = root.appendingPathComponent(name).standardizedFileURL
            guard url.path.hasPrefix(root.standardizedFileURL.path + "/") else {
                throw ReuseError.invalid("モデルのパスが不正です。")
            }
            let actual = try autoreleasepool { Self.hash(try Data(contentsOf: url)) }
            guard actual == expected else { throw ReuseError.invalid("モデルの検証に失敗: \(name)") }
        }
    }

    public static func hash(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private struct Manifest: Decodable {
        let stages: Int
        let mapped_model_sha256: String
        let files: [String: String]
    }
}

/// Confined to ReuseEngine. CNNs may use ANE; their exact placement is Core ML's decision.
final class CNNBank {
    struct Pair { let front: MLModel; let back: MLModel }
    private var pairs: [Int: Pair] = [:]
    private let pack: ReuseModelPack
    private let cache: URL
    private let identity: String
    private let sharedTransfers: Bool

    init(pack: ReuseModelPack, cache: URL, sharedTransfers: Bool = false) throws {
        self.pack = pack; self.cache = cache
        self.sharedTransfers = sharedTransfers
        identity = String(ReuseModelPack.hash(try Data(contentsOf: pack.root.appendingPathComponent("manifest.json"))).prefix(16))
        try FileManager.default.createDirectory(at: cache, withIntermediateDirectories: true)
    }

    func prepare(frames: Int) throws -> Bool {
        if pairs[frames] != nil { return true }
        let names = ["front", "back"].map { "hybrid_\($0)_f\(frames)_b321" }
        guard names.allSatisfy({ FileManager.default.fileExists(atPath: pack.root.appendingPathComponent($0 + ".mlpackage").path) }) else { return false }
        let models = try names.map { name -> MLModel in
            let destination = cache.appendingPathComponent(name + "_" + identity + ".mlmodelc")
            if !FileManager.default.fileExists(atPath: destination.path) {
                let compiled = try MLModel.compileModel(at: pack.root.appendingPathComponent(name + ".mlpackage"))
                try FileManager.default.moveItem(at: compiled, to: destination)
            }
            let config = MLModelConfiguration()
            config.computeUnits = .cpuAndNeuralEngine
            return try MLModel(contentsOf: destination, configuration: config)
        }
        pairs[frames] = Pair(front: models[0], back: models[1])
        return true
    }

    func encode(_ features: MLXArray, frames: Int) throws -> MLXArray {
        try encodePrepared(encoderInput(features), frames: frames)
    }

    func encoderInput(_ features: MLXArray) throws -> MLMultiArray { try multiArray(features) }

    func encodePrepared(_ input: MLMultiArray, frames: Int) throws -> MLXArray {
        let result = try pairs[frames]!.front.prediction(from: MLDictionaryFeatureProvider(dictionary: ["features": input]))
        guard let value = result.featureValue(for: "output")?.multiArrayValue else { throw ReuseError.invalid("CNNの出力がありません。") }
        return try array(value).asType(.float32).transposed(0, 2, 3, 1)
    }

    func decode(_ features: MLXArray, frames: Int) throws -> (MLXArray, MLXArray) {
        try decodePrepared(decoderInput(features), frames: frames)
    }

    func decoderInput(_ features: MLXArray) throws -> MLMultiArray { try multiArray(features.transposed(0, 3, 1, 2)) }

    func decodePrepared(_ input: MLMultiArray, frames: Int) throws -> (MLXArray, MLXArray) {
        let result = try pairs[frames]!.back.prediction(from: MLDictionaryFeatureProvider(dictionary: ["features": input]))
        func read(_ name: String) throws -> MLXArray {
            guard let value = result.featureValue(for: name)?.multiArrayValue else { throw ReuseError.invalid("CNNの出力がありません: \(name)") }
            return try array(value).asType(.float32).squeezed(axis: 1).transposed(0, 2, 1)
        }
        let amp = try read("amp"), real = try read("phase_real"), imag = try read("phase_imag")
        return (amp, atan2(imag, real))
    }

    private func multiArray(_ array: MLXArray) throws -> MLMultiArray {
        if sharedTransfers {
            let value = contiguous(array.asType(.float16))
            let bytes = value.asData(access: .noCopy)
            return try bytes.data.withUnsafeBytes { buffer in
                try MLMultiArray(dataPointer: UnsafeMutableRawPointer(mutating: buffer.baseAddress!),
                    shape: bytes.shape.map(NSNumber.init), dataType: .float16,
                    strides: bytes.strides.map(NSNumber.init), deallocator: { [value] _ in
                        // Core ML borrows the MLX allocation until its input is released.
                        withExtendedLifetime(value) {}
                    })
            }
        }
        let values = array.asType(.float16).asArray(Float16.self)
        let output = try MLMultiArray(shape: array.shape.map(NSNumber.init), dataType: .float16)
        values.withUnsafeBufferPointer { pointer in
            output.dataPointer.assumingMemoryBound(to: Float16.self).update(from: pointer.baseAddress!, count: values.count)
        }
        return output
    }

    private func array(_ array: MLMultiArray) throws -> MLXArray {
        let shape = array.shape.map(\.intValue), strides = array.strides.map(\.intValue)
        let count = shape.reduce(1, *)
        if sharedTransfers {
            let span = zip(shape, strides).reduce(1) { $0 + ($1.0-1)*$1.1 }
            let buffer: MLXArray
            switch array.dataType {
            case .float16: buffer = MLXArray(UnsafeBufferPointer(start: array.dataPointer.assumingMemoryBound(to: Float16.self),count:span))
            case .float32: buffer = MLXArray(UnsafeBufferPointer(start: array.dataPointer.assumingMemoryBound(to: Float.self),count:span))
            default: throw ReuseError.invalid("CNNの出力型が未対応です。")
            }
            // The MLX initializer copies the source, so no Core ML lifetime is borrowed here.
            return asStrided(buffer, shape, strides: strides)
        }
        // Core ML may return padded row strides. Preserve the logical tensor,
        // rather than treating its backing buffer as a contiguous array.
        func copy<T>(_ pointer: UnsafePointer<T>) -> [T] {
            var expectedStride = 1, contiguous = true
            for axis in shape.indices.reversed() {
                if strides[axis] != expectedStride { contiguous = false }
                expectedStride *= shape[axis]
            }
            if contiguous { return Array(UnsafeBufferPointer(start: pointer, count: count)) }
            if strides.last == 1, let width = shape.last {
                return Array<T>(unsafeUninitializedCapacity: count) { destination, initializedCount in
                    for row in 0..<(count/width) {
                        var remainder = row, offset = 0
                        for axis in shape.indices.dropLast().reversed() {
                            offset += (remainder % shape[axis])*strides[axis]
                            remainder /= shape[axis]
                        }
                        destination.baseAddress!.advanced(by: row*width).initialize(from: pointer.advanced(by: offset), count: width)
                    }
                    initializedCount = count
                }
            }
            var result = [T](); result.reserveCapacity(count)
            for index in 0..<count {
                var remainder = index, offset = 0
                for axis in shape.indices.reversed() {
                    offset += (remainder % shape[axis]) * strides[axis]
                    remainder /= shape[axis]
                }
                result.append(pointer[offset])
            }
            return result
        }
        switch array.dataType {
        case .float16: return MLXArray(copy(array.dataPointer.assumingMemoryBound(to: Float16.self)), shape)
        case .float32: return MLXArray(copy(array.dataPointer.assumingMemoryBound(to: Float.self)), shape)
        default: throw ReuseError.invalid("CNNの出力型が未対応です。")
        }
    }
}
