// SPDX-License-Identifier: Apache-2.0
import Foundation
import AVFoundation

public struct ReuseAudioInfo: Codable, Sendable {
    public let sourceSampleRate: Double
    public let sourceChannels: UInt32
    public let sourceFrames: Int64
    public let preparedSamples: Int
    public let resampled: Bool
    public var preparedSeconds: Double { Double(preparedSamples)/16000 }
}

public struct ReuseAudio: Sendable {
    public let samples: [Float]
    public let info: ReuseAudioInfo
}

public enum ReuseAudioFile {
    public static func read(_ url: URL, maximumSeconds: Double = 60) throws -> ReuseAudio {
        try autoreleasepool { try readPCM(url, maximumSeconds: maximumSeconds) }
    }

    private static func readPCM(_ url: URL, maximumSeconds: Double) throws -> ReuseAudio {
        let file = try AVAudioFile(forReading: url, commonFormat: .pcmFormatFloat32, interleaved: false)
        defer { file.close() }
        let source = file.processingFormat
        guard file.length > 320, Double(file.length)/source.sampleRate <= maximumSeconds,
              source.channelCount == 1 || source.channelCount == 2,
              file.length < Int64(UInt32.max) else {
            throw ReuseError.invalid(String(format: "%.0f秒以内のmono／stereo音声を選んでください。", maximumSeconds))
        }
        guard let input = AVAudioPCMBuffer(pcmFormat: source, frameCapacity: AVAudioFrameCount(file.length)) else {
            throw ReuseError.invalid("音声を読み込めません。")
        }
        // AVAudioFile can return a partial read even for a Float32 WAV.
        // Continue until every advertised frame is decoded, rather than
        // assuming one read fills the buffer.
        guard let readBuffer = AVAudioPCMBuffer(pcmFormat: source,
            frameCapacity: AVAudioFrameCount(min(file.length, 65536))) else {
            throw ReuseError.invalid("音声バッファを確保できません。")
        }
        var loaded = 0
        while loaded < Int(file.length) {
            let requested = min(Int(readBuffer.frameCapacity), Int(file.length)-loaded)
            try file.read(into: readBuffer, frameCount: AVAudioFrameCount(requested))
            let count = Int(readBuffer.frameLength)
            if count == 0 { break }
            for channel in 0..<Int(source.channelCount) {
                input.floatChannelData![channel].advanced(by: loaded)
                    .update(from: readBuffer.floatChannelData![channel], count: count)
            }
            loaded += count
        }
        guard loaded == Int(file.length) else { throw ReuseError.invalid("音声の末尾まで読み込めませんでした。") }
        input.frameLength = AVAudioFrameCount(loaded)
        let converted: AVAudioPCMBuffer
        let resampled = source.sampleRate != 16000
        if source.sampleRate == 16000 && source.channelCount == 1 { converted = input }
        else {
            guard let target = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1, interleaved: false),
                  let converter = AVAudioConverter(from: source, to: target),
                  let output = AVAudioPCMBuffer(pcmFormat: target,
                    frameCapacity: AVAudioFrameCount(ceil(Double(input.frameLength)*16000/source.sampleRate))+4096) else {
                throw ReuseError.invalid("16 kHzへの変換を準備できません。")
            }
            converter.downmix = true
            converter.sampleRateConverterQuality = AVAudioQuality.max.rawValue
            var supplied = false
            var error: NSError?
            let status = converter.convert(to: output, error: &error) { _, state in
                if supplied { state.pointee = .endOfStream; return nil }
                supplied = true; state.pointee = .haveData; return input
            }
            if let error { throw error }
            guard status != .error, output.frameLength > 320 else { throw ReuseError.invalid("音声の変換に失敗しました。") }
            converted = output
        }
        guard let channel = converted.floatChannelData?[0] else { throw ReuseError.invalid("Float音声へ変換できません。") }
        let samples = Array(UnsafeBufferPointer(start: channel, count: Int(converted.frameLength)))
        guard samples.allSatisfy(\.isFinite), samples.count > 320 else { throw ReuseError.invalid("音声が短すぎるか、非有限値があります。") }
        return ReuseAudio(samples: samples, info: ReuseAudioInfo(sourceSampleRate: source.sampleRate,
            sourceChannels: source.channelCount, sourceFrames: file.length,
            preparedSamples: samples.count, resampled: resampled))
    }

    public static func write(_ samples: [Float], to url: URL) throws {
        guard !samples.isEmpty, samples.allSatisfy(\.isFinite),
              let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1, interleaved: false),
              let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count)),
              let channel = buffer.floatChannelData?[0] else { throw ReuseError.invalid("音声を保存できません。") }
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        buffer.frameLength = AVAudioFrameCount(samples.count)
        samples.withUnsafeBufferPointer { channel.update(from: $0.baseAddress!, count: samples.count) }
        // AVAudioFile buffers its final partial block. Close it before
        // returning so immediate playback/sharing sees the complete WAV.
        let file = try AVAudioFile(forWriting: url, settings: format.settings, commonFormat: .pcmFormatFloat32, interleaved: false)
        defer { file.close() }
        try file.write(from: buffer)
    }

    public static func writeJSON<T: Encodable>(_ value: T, to url: URL) throws {
        let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try encoder.encode(value).write(to: url, options: .atomic)
    }
}
