// SPDX-License-Identifier: Apache-2.0
import AVFoundation
import CoreMedia
import Foundation
import ReuseKit

public struct CleanerOptions: Sendable {
    /// RE-USE accepts 8–48 kHz. Sources in this range are processed at their
    /// own rate; others are converted to the nearest limit for processing.
    public var minSampleRate = 8_000
    public var maxSampleRate = 48_000
    /// Analysis window and hop, as in NVIDIA's chunked inference (5 s / 50 %).
    public var windowSeconds = 5.0
    public var hopFraction = 0.5
    /// Keep every channel (each is enhanced separately); false mixes down to mono.
    public var keepChannels = true
    public var aacBitRatePerChannel = 128_000

    public init() {}

    public func processingRate(for sourceRate: Int) -> Int {
        min(max(sourceRate, minSampleRate), maxSampleRate)
    }
}

public struct CleanerProgress: Sendable {
    public enum Stage: String, Sendable { case decoding, enhancing, writing, done }
    public let stage: Stage
    /// Overall progress 0...1.
    public let fraction: Double
}

public struct CleanerReport: Sendable, Codable {
    public let input: String
    public let output: String
    public let hasVideo: Bool
    public let channels: Int
    /// Rate of the written audio.
    public let sampleRate: Int
    /// Rate the model ran at.
    public let processingSampleRate: Int
    /// For example "FLAC 24-bit".
    public let encoding: String
    public let audioSeconds: Double
    public let windows: Int
    /// Other audio tracks copied unchanged, and those that could not be copied.
    public let copiedAudioTracks: Int
    public let skippedAudioTracks: Int
    public let enhanceSeconds: Double
    public let totalSeconds: Double
    /// Enhancement time divided by audio duration (all channels included).
    public var realTimeFactor: Double { audioSeconds > 0 ? enhanceSeconds / audioSeconds : 0 }
}

public enum CleanerError: LocalizedError {
    case noAudio
    case unsupported(String)
    case failed(String)
    public var errorDescription: String? {
        switch self {
        case .noAudio: return "このファイルには音声トラックがありません。"
        case .unsupported(let message): return message
        case .failed(let message): return message
        }
    }
}

/// Removes noise from the audio of a video or audio file with RE-USE. Video
/// and other audio tracks are copied without re-encoding. The cleaned audio
/// keeps the source rate and channels; lossless sources stay lossless.
public final class MediaCleaner {
    private let engine: ReuseEngine

    public init(engine: ReuseEngine) {
        self.engine = engine
    }

    /// The output extension for an input: the same container where possible.
    public static func outputExtension(for input: URL, hasVideo: Bool) -> String {
        let ext = input.pathExtension.lowercased()
        if hasVideo { return ext == "mov" || ext == "m4v" ? ext : "mp4" }
        switch ext {
        case "wav", "caf", "flac": return ext
        case "aif", "aiff", "aifc": return "aiff"
        default: return "m4a"
        }
    }

    static func fileType(forExtension ext: String) -> AVFileType {
        switch ext {
        case "mov": return .mov
        case "m4v": return .m4v
        case "mp4": return .mp4
        default: return .m4a
        }
    }

    public static func hasVideo(_ input: URL) async -> Bool {
        let tracks = try? await AVURLAsset(url: input).loadTracks(withMediaType: .video)
        return !(tracks?.isEmpty ?? true)
    }

    public func clean(input: URL, output: URL, options: CleanerOptions = CleanerOptions(),
                      progress: @escaping @Sendable (CleanerProgress) -> Void = { _ in }) async throws -> CleanerReport {
        let began = Date()
        let asset = AVURLAsset(url: input)
        let audioTracks = try await asset.loadTracks(withMediaType: .audio)
        var mainTrack = audioTracks.first
        for track in audioTracks where try await track.load(.isEnabled) { mainTrack = track; break }
        guard let audioTrack = mainTrack else { throw CleanerError.noAudio }
        let otherAudioTracks = audioTracks.filter { $0 != audioTrack }
        let videoTracks = try await asset.loadTracks(withMediaType: .video)
        let duration = try await asset.load(.duration).seconds

        let ext = Self.outputExtension(for: input, hasVideo: !videoTracks.isEmpty)
        guard output.pathExtension.lowercased() == ext else {
            throw CleanerError.unsupported("出力ファイルの拡張子は .\(ext) にしてください。")
        }
        let source = try await SourceAudio(audioTrack)
        let layout = options.keepChannels ? source.layout : .standard(1)
        let rate = options.processingRate(for: source.sampleRate)
        let encoding = AudioEncoding.choose(outputExtension: ext, hasVideo: !videoTracks.isEmpty, source: source)

        let work = FileManager.default.temporaryDirectory
            .appendingPathComponent("VoiceCleaner-" + UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: work, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: work) }
        let decodedURL = work.appendingPathComponent("decoded.caf")
        let enhancedURL = work.appendingPathComponent("enhanced.caf")

        // 1. Decode the audio track to 32-bit float PCM at the processing rate.
        progress(CleanerProgress(stage: .decoding, fraction: 0))
        let audioStart = try await decode(asset: asset, track: audioTrack, layout: layout,
                                          sampleRate: rate, to: decodedURL) { fraction in
            progress(CleanerProgress(stage: .decoding, fraction: 0.05 * min(1, fraction / max(duration, 0.001))))
        }

        // 2. Enhance each channel window by window.
        let enhanceBegan = Date()
        let (frames, windows) = try await enhance(decodedURL, to: enhancedURL, layout: layout, sampleRate: rate,
                                                  options: options) { fraction in
            progress(CleanerProgress(stage: .enhancing, fraction: 0.05 + 0.9 * fraction))
        }
        let enhanceSeconds = Date().timeIntervalSince(enhanceBegan)

        // 3. Write the enhanced audio (and copy video and other audio tracks).
        if FileManager.default.fileExists(atPath: output.path) { try FileManager.default.removeItem(at: output) }
        let written: (rate: Int, encoding: AudioEncoding, copied: Int, skipped: Int)
        if videoTracks.isEmpty && encoding.isLossless {
            // Lossless audio files keep the source rate, so the frame count matches the input.
            try Self.writeFile(enhancedURL, to: output, encoding: encoding, layout: layout, sampleRate: source.sampleRate) { fraction in
                progress(CleanerProgress(stage: .writing, fraction: 0.95 + 0.05 * fraction))
            }
            written = (source.sampleRate, encoding, 0, otherAudioTracks.count)
        } else {
            let result = try await write(asset: asset, videoTracks: videoTracks, mainAudio: audioTrack,
                                         otherAudio: otherAudioTracks, enhanced: enhancedURL, audioStart: audioStart,
                                         layout: layout, sampleRate: rate, encoding: encoding,
                                         aacBitRatePerChannel: options.aacBitRatePerChannel,
                                         fileType: Self.fileType(forExtension: ext), to: output) { fraction in
                progress(CleanerProgress(stage: .writing, fraction: 0.95 + 0.05 * fraction))
            }
            written = (rate, result.encoding, result.copied, result.skipped)
        }
        progress(CleanerProgress(stage: .done, fraction: 1))
        return CleanerReport(input: input.path, output: output.path, hasVideo: !videoTracks.isEmpty, channels: layout.channels,
                             sampleRate: written.rate, processingSampleRate: rate, encoding: written.encoding.label,
                             audioSeconds: Double(frames) / Double(rate), windows: windows,
                             copiedAudioTracks: written.copied, skippedAudioTracks: written.skipped,
                             enhanceSeconds: enhanceSeconds, totalSeconds: Date().timeIntervalSince(began))
    }

    // MARK: - Decoding

    static func pcmSettings(layout: ChannelLayout, sampleRate: Int, interleaved: Bool) -> [String: Any] {
        [AVFormatIDKey: kAudioFormatLinearPCM,
         AVSampleRateKey: sampleRate,
         AVNumberOfChannelsKey: layout.channels,
         AVChannelLayoutKey: layout.data,
         AVLinearPCMBitDepthKey: 32,
         AVLinearPCMIsFloatKey: true,
         AVLinearPCMIsBigEndianKey: false,
         AVLinearPCMIsNonInterleaved: !interleaved]
    }

    /// Returns the presentation time of the first decoded audio sample.
    private func decode(asset: AVURLAsset, track: AVAssetTrack, layout: ChannelLayout, sampleRate: Int, to url: URL,
                        progress: (Double) -> Void) async throws -> CMTime {
        let channels = layout.channels
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderAudioMixOutput(audioTracks: [track],
            audioSettings: Self.pcmSettings(layout: layout, sampleRate: sampleRate, interleaved: true))
        output.alwaysCopiesSampleData = false
        guard reader.canAdd(output) else { throw CleanerError.unsupported("この音声形式は読み込めません。") }
        reader.add(output)
        guard reader.startReading() else {
            throw CleanerError.failed("音声を読み込めませんでした: \(reader.error?.localizedDescription ?? "")")
        }
        let file = try AVAudioFile(forWriting: url,
            settings: Self.pcmSettings(layout: layout, sampleRate: sampleRate, interleaved: false),
            commonFormat: .pcmFormatFloat32, interleaved: false)
        var start: CMTime?
        var interleavedBuffer = [Float]()
        while let sample = output.copyNextSampleBuffer() {
            try Task.checkCancellation()
            if start == nil { start = CMSampleBufferGetPresentationTimeStamp(sample) }
            guard let block = CMSampleBufferGetDataBuffer(sample) else { continue }
            let length = CMBlockBufferGetDataLength(block)
            let count = length / MemoryLayout<Float>.size
            let frames = count / channels
            guard frames > 0 else { continue }
            if interleavedBuffer.count < count { interleavedBuffer = [Float](repeating: 0, count: count) }
            let status = interleavedBuffer.withUnsafeMutableBytes {
                CMBlockBufferCopyDataBytes(block, atOffset: 0, dataLength: length, destination: $0.baseAddress!)
            }
            guard status == kCMBlockBufferNoErr else { throw CleanerError.failed("音声データを取り出せませんでした。") }
            guard let pcm = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(frames)),
                  let data = pcm.floatChannelData else { throw CleanerError.failed("メモリを確保できませんでした。") }
            for c in 0..<channels {
                let destination = data[c]
                for i in 0..<frames { destination[i] = interleavedBuffer[i * channels + c] }
            }
            pcm.frameLength = AVAudioFrameCount(frames)
            try file.write(from: pcm)
            progress(CMSampleBufferGetPresentationTimeStamp(sample).seconds)
        }
        file.close()
        if reader.status == .failed {
            throw CleanerError.failed("音声の読み込みに失敗しました: \(reader.error?.localizedDescription ?? "")")
        }
        guard let start else { throw CleanerError.noAudio }
        return start
    }

    // MARK: - Enhancement

    /// Returns the number of frames and the number of windows per channel.
    private func enhance(_ input: URL, to output: URL, layout: ChannelLayout, sampleRate: Int, options: CleanerOptions,
                         progress: (Double) -> Void) async throws -> (Int, Int) {
        let source = try AVAudioFile(forReading: input, commonFormat: .pcmFormatFloat32, interleaved: false)
        let format = source.processingFormat
        let channels = Int(format.channelCount)
        let total = Int(source.length)
        guard total > 0 else { throw CleanerError.noAudio }
        let destination = try AVAudioFile(forWriting: output,
            settings: Self.pcmSettings(layout: layout, sampleRate: sampleRate, interleaved: false),
            commonFormat: .pcmFormatFloat32, interleaved: false)
        var closed = false
        defer { if !closed { destination.close() } }
        var written = 0

        let windowLength = Int(options.windowSeconds * Double(sampleRate))
        let hop = max(1, Int(Double(windowLength) * options.hopFraction))
        let minimum = STFTConfig.forSampleRate(sampleRate).nFFT + 1
        let plan = WindowPlan(total: total, windowLength: windowLength, hop: hop)
        let streams = (0..<channels).map { _ in OverlapAddStream(plan: plan) }
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(plan.windows[0].length)) else {
            throw CleanerError.failed("メモリを確保できませんでした。")
        }
        for (index, window) in plan.windows.enumerated() {
            try Task.checkCancellation()
            let input = try Self.read(source, start: window.start, length: window.length, using: buffer)
            var finalized = [[Float]]()
            var finalStart = 0
            for c in 0..<channels {
                let samples = input[c]
                // Too short for one STFT frame: keep the original samples.
                let enhanced = window.length < minimum
                    ? samples
                    : try await engine.enhanceWindow(samples, sampleRate: sampleRate)
                let (start, values) = streams[c].add(enhanced)
                finalStart = start
                finalized.append(values)
            }
            // AVAudioFile.length can lag behind while writing, so the position
            // is tracked here instead of read back from the file.
            guard finalStart == written else {
                throw CleanerError.failed("出力位置が一致しません（\(finalStart) / \(written)）。")
            }
            written += try Self.write(finalized, to: destination)
            progress(Double(index + 1) / Double(plan.windows.count))
        }
        destination.close()
        closed = true
        guard written == total else { throw CleanerError.failed("出力の長さが一致しません（\(written) / \(total)）。") }
        let check = try AVAudioFile(forReading: output, commonFormat: .pcmFormatFloat32, interleaved: false)
        guard Int(check.length) == total else {
            throw CleanerError.failed("一時ファイルの長さが一致しません（\(check.length) / \(total)）。")
        }
        return (total, plan.windows.count)
    }

    /// Reads `length` frames starting at `start`, one array per channel.
    /// `AVAudioFile.read` may return fewer frames than requested (it does for
    /// mono CAF files), so this keeps reading until the window is filled.
    static func read(_ file: AVAudioFile, start: Int, length: Int, using buffer: AVAudioPCMBuffer) throws -> [[Float]] {
        let channels = Int(buffer.format.channelCount)
        var result = (0..<channels).map { _ in [Float]() }
        for c in 0..<channels { result[c].reserveCapacity(length) }
        file.framePosition = AVAudioFramePosition(start)
        while result[0].count < length {
            let wanted = min(length - result[0].count, Int(buffer.frameCapacity))
            try file.read(into: buffer, frameCount: AVAudioFrameCount(wanted))
            let got = Int(buffer.frameLength)
            guard got > 0, let data = buffer.floatChannelData else {
                throw CleanerError.failed("一時ファイルを読み込めませんでした。")
            }
            for c in 0..<channels { result[c] += UnsafeBufferPointer(start: data[c], count: got) }
        }
        return result
    }

    /// Returns the number of frames written.
    private static func write(_ channels: [[Float]], to file: AVAudioFile) throws -> Int {
        let frames = channels.first?.count ?? 0
        guard frames > 0 else { return 0 }
        guard let pcm = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(frames)),
              let data = pcm.floatChannelData else { throw CleanerError.failed("メモリを確保できませんでした。") }
        for (c, values) in channels.enumerated() {
            values.withUnsafeBufferPointer { data[c].update(from: $0.baseAddress!, count: frames) }
        }
        pcm.frameLength = AVAudioFrameCount(frames)
        try file.write(from: pcm)
        return frames
    }

    // MARK: - Writing audio files

    /// Writes a lossless audio file (the container follows the extension) at
    /// `sampleRate`, converting from the processing rate when they differ.
    static func writeFile(_ enhanced: URL, to output: URL, encoding: AudioEncoding, layout: ChannelLayout,
                          sampleRate: Int, progress: (Double) -> Void) throws {
        let source = try AVAudioFile(forReading: enhanced, commonFormat: .pcmFormatFloat32, interleaved: false)
        var settings: [String: Any] = [AVSampleRateKey: sampleRate, AVNumberOfChannelsKey: layout.channels,
                                       AVChannelLayoutKey: layout.data]
        switch encoding {
        case .pcm(let bits, let float, let bigEndian):
            settings[AVFormatIDKey] = kAudioFormatLinearPCM
            settings[AVLinearPCMBitDepthKey] = bits
            settings[AVLinearPCMIsFloatKey] = float
            settings[AVLinearPCMIsBigEndianKey] = bigEndian
        case .alac(let bits):
            settings[AVFormatIDKey] = kAudioFormatAppleLossless
            settings[AVEncoderBitDepthHintKey] = bits
        case .flac(let bits):
            settings[AVFormatIDKey] = kAudioFormatFLAC
            settings[AVEncoderBitDepthHintKey] = bits
        case .aac:
            throw CleanerError.failed("AAC はファイルとして直接書き出せません。")
        }
        var destination: AVAudioFile? = try AVAudioFile(forWriting: output, settings: settings,
                                                        commonFormat: .pcmFormatFloat32, interleaved: false)
        let outFormat = destination!.processingFormat
        let block: AVAudioFrameCount = 8192
        guard let input = AVAudioPCMBuffer(pcmFormat: source.processingFormat, frameCapacity: block) else {
            throw CleanerError.failed("メモリを確保できませんでした。")
        }
        let total = source.length
        let clampsToUnit = encoding != .pcm(bits: 32, float: true, bigEndian: false)
        func append(_ buffer: AVAudioPCMBuffer) throws {
            guard buffer.frameLength > 0, let data = buffer.floatChannelData else { return }
            if clampsToUnit {
                // Integer formats would wrap; clip instead.
                for c in 0..<layout.channels {
                    for i in 0..<Int(buffer.frameLength) { data[c][i] = min(1, max(-1, data[c][i])) }
                }
            }
            try destination!.write(from: buffer)
        }

        if Int(source.processingFormat.sampleRate) == sampleRate {
            while source.framePosition < total {
                try source.read(into: input, frameCount: min(block, AVAudioFrameCount(total - source.framePosition)))
                guard input.frameLength > 0 else { break }
                try append(input)
                progress(Double(source.framePosition) / Double(total))
            }
        } else {
            guard let converter = AVAudioConverter(from: source.processingFormat, to: outFormat),
                  let output = AVAudioPCMBuffer(pcmFormat: outFormat,
                      frameCapacity: AVAudioFrameCount(Double(block) * outFormat.sampleRate / source.processingFormat.sampleRate) + 64)
            else { throw CleanerError.failed("サンプルレートを変換できません。") }
            converter.sampleRateConverterQuality = AVAudioQuality.max.rawValue
            var finished = false
            while true {
                var conversionError: NSError?
                let status = converter.convert(to: output, error: &conversionError) { _, inputStatus in
                    if finished || source.framePosition >= total {
                        finished = true
                        inputStatus.pointee = .endOfStream
                        return nil
                    }
                    do {
                        try source.read(into: input, frameCount: min(block, AVAudioFrameCount(total - source.framePosition)))
                    } catch {
                        finished = true
                        inputStatus.pointee = .endOfStream
                        return nil
                    }
                    inputStatus.pointee = .haveData
                    return input
                }
                if let conversionError { throw CleanerError.failed("サンプルレートを変換できません: \(conversionError.localizedDescription)") }
                try append(output)
                progress(Double(source.framePosition) / Double(max(total, 1)))
                if status == .endOfStream || status == .error { break }
            }
        }
        destination = nil  // Closes the file.
        let check = try AVAudioFile(forReading: output)
        let expected = Int((Double(total) * Double(sampleRate) / source.processingFormat.sampleRate).rounded())
        guard abs(Int(check.length) - expected) <= 1 else {
            throw CleanerError.failed("出力の長さが一致しません（\(check.length) / \(expected)）。")
        }
    }

    // MARK: - Writing media

    private final class Cancellation: @unchecked Sendable {
        private let lock = NSLock()
        private var value = false
        var isCancelled: Bool { lock.lock(); defer { lock.unlock() }; return value }
        func cancel() { lock.lock(); value = true; lock.unlock() }
    }

    /// Output settings to try for the enhanced audio, best first. AAC is not
    /// accepted for every multichannel layout or bit rate, so it steps down
    /// and finally falls back to ALAC.
    static func audioSettingsCandidates(encoding: AudioEncoding, layout: ChannelLayout, sampleRate: Int,
                                        aacBitRatePerChannel: Int) -> [(AudioEncoding, [String: Any])] {
        let common: [String: Any] = [AVSampleRateKey: sampleRate, AVNumberOfChannelsKey: layout.channels,
                                     AVChannelLayoutKey: layout.data]
        var candidates = [(AudioEncoding, [String: Any])]()
        switch encoding {
        case .pcm(let bits, let float, let bigEndian):
            candidates.append((encoding, common.merging([
                AVFormatIDKey: kAudioFormatLinearPCM, AVLinearPCMBitDepthKey: bits, AVLinearPCMIsFloatKey: float,
                AVLinearPCMIsBigEndianKey: bigEndian, AVLinearPCMIsNonInterleaved: false]) { $1 }))
        case .alac(let bits), .flac(let bits):
            candidates.append((.alac(bits: bits), common.merging([
                AVFormatIDKey: kAudioFormatAppleLossless, AVEncoderBitDepthHintKey: bits]) { $1 }))
        case .aac:
            break
        }
        // The AAC encoder accepts settings it then fails to encode when the bit
        // rate is high for the sample rate (e.g. 128 kbps per channel at 24 kHz).
        let ceiling = min(aacBitRatePerChannel, sampleRate * 3)
        for perChannel in [aacBitRatePerChannel, 96_000, 64_000, 48_000, 32_000, 24_000, 16_000] where perChannel <= ceiling {
            candidates.append((.aac, common.merging([
                AVFormatIDKey: kAudioFormatMPEG4AAC, AVEncoderBitRateKey: perChannel * layout.channels]) { $1 }))
        }
        candidates.append((.alac(bits: 24), common.merging([
            AVFormatIDKey: kAudioFormatAppleLossless, AVEncoderBitDepthHintKey: 24]) { $1 }))
        return candidates
    }

    private func write(asset: AVURLAsset, videoTracks: [AVAssetTrack], mainAudio: AVAssetTrack, otherAudio: [AVAssetTrack],
                       enhanced: URL, audioStart: CMTime, layout: ChannelLayout, sampleRate: Int, encoding: AudioEncoding,
                       aacBitRatePerChannel: Int, fileType: AVFileType, to output: URL,
                       progress: @escaping @Sendable (Double) -> Void) async throws
                       -> (encoding: AudioEncoding, copied: Int, skipped: Int) {
        let writer = try AVAssetWriter(outputURL: output, fileType: fileType)
        writer.shouldOptimizeForNetworkUse = true

        // Video and other audio tracks: passthrough (no re-encoding).
        let reader = try AVAssetReader(asset: asset)
        var passthrough = [(AVAssetReaderTrackOutput, AVAssetWriterInput)]()
        for track in videoTracks {
            let trackOutput = AVAssetReaderTrackOutput(track: track, outputSettings: nil)
            trackOutput.alwaysCopiesSampleData = false
            guard reader.canAdd(trackOutput) else { continue }
            let hint = try await track.load(.formatDescriptions).first
            let input = AVAssetWriterInput(mediaType: .video, outputSettings: nil, sourceFormatHint: hint)
            input.transform = try await track.load(.preferredTransform)
            input.expectsMediaDataInRealTime = false
            guard writer.canAdd(input) else {
                throw CleanerError.unsupported("この映像形式は\(fileType == .mp4 ? "MP4" : "このファイル形式")のまま書き出せません。")
            }
            reader.add(trackOutput)
            writer.add(input)
            passthrough.append((trackOutput, input))
        }

        // Enhanced audio.
        var chosen: (AudioEncoding, [String: Any])?
        for candidate in Self.audioSettingsCandidates(encoding: encoding, layout: layout, sampleRate: sampleRate,
                                                      aacBitRatePerChannel: aacBitRatePerChannel)
        where writer.canApply(outputSettings: candidate.1, forMediaType: .audio) {
            chosen = candidate
            break
        }
        guard let (usedEncoding, settings) = chosen else {
            throw CleanerError.unsupported("この音声（\(layout.channels)ch・\(sampleRate) Hz）を書き出せる形式がありません。")
        }
        let audioInput = AVAssetWriterInput(mediaType: .audio, outputSettings: settings)
        audioInput.expectsMediaDataInRealTime = false
        audioInput.languageCode = try await mainAudio.load(.languageCode)
        audioInput.extendedLanguageTag = try await mainAudio.load(.extendedLanguageTag)
        guard writer.canAdd(audioInput) else { throw CleanerError.unsupported("音声トラックを追加できません。") }
        writer.add(audioInput)

        // Other audio tracks (commentary, Spatial Audio, …) are kept unchanged
        // but disabled, so players pick the cleaned track by default.
        var others = [AVAssetWriterInput]()
        var skipped = 0
        for track in otherAudio {
            let trackOutput = AVAssetReaderTrackOutput(track: track, outputSettings: nil)
            trackOutput.alwaysCopiesSampleData = false
            let hint = try await track.load(.formatDescriptions).first
            let input = AVAssetWriterInput(mediaType: .audio, outputSettings: nil, sourceFormatHint: hint)
            input.expectsMediaDataInRealTime = false
            input.marksOutputTrackAsEnabled = false
            input.languageCode = try await track.load(.languageCode)
            input.extendedLanguageTag = try await track.load(.extendedLanguageTag)
            guard reader.canAdd(trackOutput), writer.canAdd(input) else { skipped += 1; continue }
            reader.add(trackOutput)
            writer.add(input)
            passthrough.append((trackOutput, input))
            others.append(input)
        }
        if !others.isEmpty {
            let group = AVAssetWriterInputGroup(inputs: [audioInput] + others, defaultInput: audioInput)
            if writer.canAdd(group) { writer.add(group) }
        }

        guard writer.startWriting() else {
            throw CleanerError.failed("書き出しを開始できませんでした: \(writer.error?.localizedDescription ?? "")")
        }
        if !passthrough.isEmpty && !reader.startReading() {
            writer.cancelWriting()
            throw CleanerError.failed("映像を読み込めませんでした: \(reader.error?.localizedDescription ?? "")")
        }

        // Peek at the first sample of each passthrough track to find the session start.
        var pending = passthrough.map { $0.0.copyNextSampleBuffer() }
        var sessionStart = CMTimeMinimum(.zero, audioStart)
        for case let sample? in pending {
            sessionStart = CMTimeMinimum(sessionStart, CMSampleBufferGetPresentationTimeStamp(sample))
        }
        writer.startSession(atSourceTime: sessionStart)

        let audioSource = try AudioSampleSource(url: enhanced, layout: layout, sampleRate: sampleRate, start: audioStart)
        let cancellation = Cancellation()
        do {
            try await withTaskCancellationHandler {
                try await withThrowingTaskGroup(of: Void.self) { group in
                    for (index, pair) in passthrough.enumerated() {
                        let first = pending[index]
                        pending[index] = nil
                        let trackOutput = pair.0
                        let held = Box<CMSampleBuffer?>(first)
                        group.addTask {
                            try await Self.pump(pair.1, writer: writer, cancellation: cancellation, label: "track-\(index)") {
                                if let sample = held.value { held.value = nil; return sample }
                                return trackOutput.copyNextSampleBuffer()
                            }
                        }
                    }
                    group.addTask {
                        try await Self.pump(audioInput, writer: writer, cancellation: cancellation, label: "audio") {
                            let sample = try audioSource.next()
                            progress(audioSource.fraction)
                            return sample
                        }
                    }
                    try await group.waitForAll()
                }
            } onCancel: {
                cancellation.cancel()
            }
        } catch {
            writer.cancelWriting()
            reader.cancelReading()
            throw error
        }
        await writer.finishWriting()
        if writer.status != .completed {
            throw CleanerError.failed("書き出しに失敗しました: \(writer.error?.localizedDescription ?? "")")
        }
        if reader.status == .failed {
            throw CleanerError.failed("映像の読み込みに失敗しました: \(reader.error?.localizedDescription ?? "")")
        }
        return (usedEncoding, others.count, skipped)
    }

    private static func pump(_ input: AVAssetWriterInput, writer: AVAssetWriter, cancellation: Cancellation,
                             label: String, next: @escaping () throws -> CMSampleBuffer?) async throws {
        let queue = DispatchQueue(label: "voicecleaner." + label)
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            // Only touched on `queue`.
            let finished = Box(false)
            let finish: (Error?) -> Void = { error in
                guard !finished.value else { return }
                finished.value = true
                input.markAsFinished()
                if let error { continuation.resume(throwing: error) } else { continuation.resume() }
            }
            input.requestMediaDataWhenReady(on: queue) {
                while !finished.value && input.isReadyForMoreMediaData {
                    if cancellation.isCancelled { finish(CancellationError()); return }
                    do {
                        guard let sample = try next() else { finish(nil); return }
                        if !input.append(sample) {
                            finish(CleanerError.failed("書き出しに失敗しました: \(writer.error?.localizedDescription ?? "")"))
                            return
                        }
                    } catch {
                        finish(error)
                        return
                    }
                }
            }
        }
    }
}

/// A mutable reference used from a single serial queue at a time.
private final class Box<Value>: @unchecked Sendable {
    var value: Value
    init(_ value: Value) { self.value = value }
}

/// Reads the enhanced PCM file and produces interleaved float sample buffers.
private final class AudioSampleSource: @unchecked Sendable {
    private let file: AVAudioFile
    private let channels: Int
    private let sampleRate: Int
    private let start: CMTime
    private let buffer: AVAudioPCMBuffer
    private let format: CMAudioFormatDescription
    private var position: Int64 = 0
    private let total: Int64
    private static let blockFrames: AVAudioFrameCount = 8192

    init(url: URL, layout: ChannelLayout, sampleRate: Int, start: CMTime) throws {
        file = try AVAudioFile(forReading: url, commonFormat: .pcmFormatFloat32, interleaved: false)
        channels = layout.channels
        self.sampleRate = sampleRate
        self.start = start
        total = file.length
        guard let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: Self.blockFrames) else {
            throw CleanerError.failed("メモリを確保できませんでした。")
        }
        self.buffer = buffer
        var asbd = AudioStreamBasicDescription(
            mSampleRate: Float64(sampleRate), mFormatID: kAudioFormatLinearPCM,
            mFormatFlags: kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked,
            mBytesPerPacket: UInt32(4 * channels), mFramesPerPacket: 1, mBytesPerFrame: UInt32(4 * channels),
            mChannelsPerFrame: UInt32(channels), mBitsPerChannel: 32, mReserved: 0)
        var description: CMAudioFormatDescription?
        let status = layout.data.withUnsafeBytes { bytes in
            CMAudioFormatDescriptionCreate(allocator: kCFAllocatorDefault, asbd: &asbd,
                layoutSize: bytes.count, layout: bytes.baseAddress!.assumingMemoryBound(to: AudioChannelLayout.self),
                magicCookieSize: 0, magicCookie: nil, extensions: nil, formatDescriptionOut: &description)
        }
        guard status == noErr, let description else { throw CleanerError.failed("音声形式を作成できませんでした。") }
        format = description
    }

    var fraction: Double { total > 0 ? Double(position) / Double(total) : 1 }

    func next() throws -> CMSampleBuffer? {
        guard position < total else { return nil }
        let frames = AVAudioFrameCount(min(Int64(Self.blockFrames), total - position))
        try file.read(into: buffer, frameCount: frames)
        let count = Int(buffer.frameLength)
        guard count > 0, let data = buffer.floatChannelData else { return nil }
        var interleaved = [Float](repeating: 0, count: count * channels)
        for c in 0..<channels {
            let channel = data[c]
            for i in 0..<count { interleaved[i * channels + c] = channel[i] }
        }
        let byteCount = interleaved.count * MemoryLayout<Float>.size
        var block: CMBlockBuffer?
        var status = CMBlockBufferCreateWithMemoryBlock(allocator: kCFAllocatorDefault, memoryBlock: nil,
            blockLength: byteCount, blockAllocator: kCFAllocatorDefault, customBlockSource: nil, offsetToData: 0,
            dataLength: byteCount, flags: kCMBlockBufferAssureMemoryNowFlag, blockBufferOut: &block)
        guard status == kCMBlockBufferNoErr, let block else { throw CleanerError.failed("音声バッファを作成できませんでした。") }
        status = interleaved.withUnsafeBytes {
            CMBlockBufferReplaceDataBytes(with: $0.baseAddress!, blockBuffer: block, offsetIntoDestination: 0, dataLength: byteCount)
        }
        guard status == kCMBlockBufferNoErr else { throw CleanerError.failed("音声バッファを作成できませんでした。") }
        let time = CMTimeAdd(start, CMTime(value: position, timescale: CMTimeScale(sampleRate)))
        var sample: CMSampleBuffer?
        let created = CMAudioSampleBufferCreateReadyWithPacketDescriptions(allocator: kCFAllocatorDefault,
            dataBuffer: block, formatDescription: format, sampleCount: count, presentationTimeStamp: time,
            packetDescriptions: nil, sampleBufferOut: &sample)
        guard created == noErr, let sample else { throw CleanerError.failed("音声サンプルを作成できませんでした。") }
        position += Int64(count)
        return sample
    }
}
