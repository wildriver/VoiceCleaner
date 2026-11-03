// SPDX-License-Identifier: Apache-2.0
import AVFoundation
import CoreMedia
import Foundation
import ReuseKit

public struct CleanerOptions: Sendable {
    /// Processing and output sample rate. RE-USE accepts 8–48 kHz.
    public var sampleRate = 48_000
    /// Analysis window and hop, as in NVIDIA's chunked inference (5 s / 50 %).
    public var windowSeconds = 5.0
    public var hopFraction = 0.5
    /// Keep stereo sources as stereo (each channel is enhanced separately).
    public var keepStereo = true
    public var aacBitRatePerChannel = 128_000

    public init() {}
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
    public let sampleRate: Int
    public let audioSeconds: Double
    public let windows: Int
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
/// tracks are copied without re-encoding; audio is re-encoded as AAC.
public final class MediaCleaner {
    private let engine: ReuseEngine

    public init(engine: ReuseEngine) {
        self.engine = engine
    }

    /// The output file type and extension chosen for an input.
    public static func outputType(for input: URL, hasVideo: Bool) -> (AVFileType, String) {
        switch input.pathExtension.lowercased() {
        case "mov": return (.mov, "mov")
        case "m4v": return (.m4v, "m4v")
        default: return hasVideo ? (.mp4, "mp4") : (.m4a, "m4a")
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
        guard let audioTrack = try await asset.loadTracks(withMediaType: .audio).first else { throw CleanerError.noAudio }
        let videoTracks = try await asset.loadTracks(withMediaType: .video)
        let duration = try await asset.load(.duration).seconds
        let sourceChannels = try await Self.channelCount(of: audioTrack)
        let channels = options.keepStereo && sourceChannels >= 2 ? 2 : 1

        let work = FileManager.default.temporaryDirectory
            .appendingPathComponent("VoiceCleaner-" + UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: work, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: work) }
        let decodedURL = work.appendingPathComponent("decoded.caf")
        let enhancedURL = work.appendingPathComponent("enhanced.caf")

        // 1. Decode the audio track to 32-bit float PCM at the processing rate.
        progress(CleanerProgress(stage: .decoding, fraction: 0))
        let audioStart = try await decode(asset: asset, track: audioTrack, channels: channels,
                                          sampleRate: options.sampleRate, to: decodedURL) { fraction in
            progress(CleanerProgress(stage: .decoding, fraction: 0.05 * min(1, fraction / max(duration, 0.001))))
        }

        // 2. Enhance each channel window by window.
        let enhanceBegan = Date()
        let (frames, windows) = try await enhance(decodedURL, to: enhancedURL, options: options) { fraction in
            progress(CleanerProgress(stage: .enhancing, fraction: 0.05 + 0.9 * fraction))
        }
        let enhanceSeconds = Date().timeIntervalSince(enhanceBegan)

        // 3. Copy video and encode the enhanced audio.
        let (fileType, _) = Self.outputType(for: input, hasVideo: !videoTracks.isEmpty)
        if FileManager.default.fileExists(atPath: output.path) { try FileManager.default.removeItem(at: output) }
        try await write(asset: asset, videoTracks: videoTracks, enhanced: enhancedURL, audioStart: audioStart,
                        channels: channels, sampleRate: options.sampleRate, bitRate: options.aacBitRatePerChannel * channels,
                        fileType: fileType, to: output) { fraction in
            progress(CleanerProgress(stage: .writing, fraction: 0.95 + 0.05 * fraction))
        }
        progress(CleanerProgress(stage: .done, fraction: 1))
        return CleanerReport(input: input.path, output: output.path, hasVideo: !videoTracks.isEmpty, channels: channels,
                             sampleRate: options.sampleRate, audioSeconds: Double(frames) / Double(options.sampleRate),
                             windows: windows, enhanceSeconds: enhanceSeconds,
                             totalSeconds: Date().timeIntervalSince(began))
    }

    // MARK: - Decoding

    static func channelCount(of track: AVAssetTrack) async throws -> Int {
        let descriptions = try await track.load(.formatDescriptions)
        guard let description = descriptions.first,
              let asbd = CMAudioFormatDescriptionGetStreamBasicDescription(description)?.pointee else { return 1 }
        return Int(asbd.mChannelsPerFrame)
    }

    static func layoutData(channels: Int) -> Data {
        var layout = AudioChannelLayout()
        layout.mChannelLayoutTag = channels == 2 ? kAudioChannelLayoutTag_Stereo : kAudioChannelLayoutTag_Mono
        return Data(bytes: &layout, count: MemoryLayout<AudioChannelLayout>.size)
    }

    static func pcmSettings(channels: Int, sampleRate: Int, interleaved: Bool) -> [String: Any] {
        [AVFormatIDKey: kAudioFormatLinearPCM,
         AVSampleRateKey: sampleRate,
         AVNumberOfChannelsKey: channels,
         AVChannelLayoutKey: layoutData(channels: channels),
         AVLinearPCMBitDepthKey: 32,
         AVLinearPCMIsFloatKey: true,
         AVLinearPCMIsBigEndianKey: false,
         AVLinearPCMIsNonInterleaved: !interleaved]
    }

    /// Returns the presentation time of the first decoded audio sample.
    private func decode(asset: AVURLAsset, track: AVAssetTrack, channels: Int, sampleRate: Int, to url: URL,
                        progress: (Double) -> Void) async throws -> CMTime {
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderAudioMixOutput(audioTracks: [track],
            audioSettings: Self.pcmSettings(channels: channels, sampleRate: sampleRate, interleaved: true))
        output.alwaysCopiesSampleData = false
        guard reader.canAdd(output) else { throw CleanerError.unsupported("この音声形式は読み込めません。") }
        reader.add(output)
        guard reader.startReading() else {
            throw CleanerError.failed("音声を読み込めませんでした: \(reader.error?.localizedDescription ?? "")")
        }
        let file = try AVAudioFile(forWriting: url,
            settings: Self.pcmSettings(channels: channels, sampleRate: sampleRate, interleaved: false),
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
    private func enhance(_ input: URL, to output: URL, options: CleanerOptions,
                         progress: (Double) -> Void) async throws -> (Int, Int) {
        let source = try AVAudioFile(forReading: input, commonFormat: .pcmFormatFloat32, interleaved: false)
        let format = source.processingFormat
        let channels = Int(format.channelCount)
        let total = Int(source.length)
        guard total > 0 else { throw CleanerError.noAudio }
        let destination = try AVAudioFile(forWriting: output,
            settings: Self.pcmSettings(channels: channels, sampleRate: options.sampleRate, interleaved: false),
            commonFormat: .pcmFormatFloat32, interleaved: false)
        var closed = false
        defer { if !closed { destination.close() } }
        var written = 0

        let windowLength = Int(options.windowSeconds * Double(options.sampleRate))
        let hop = max(1, Int(Double(windowLength) * options.hopFraction))
        let minimum = STFTConfig.forSampleRate(options.sampleRate).nFFT + 1
        let plan = WindowPlan(total: total, windowLength: windowLength, hop: hop)
        let streams = (0..<channels).map { _ in OverlapAddStream(plan: plan) }
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(plan.windows[0].length)) else {
            throw CleanerError.failed("メモリを確保できませんでした。")
        }
        for (index, window) in plan.windows.enumerated() {
            try Task.checkCancellation()
            source.framePosition = AVAudioFramePosition(window.start)
            try source.read(into: buffer, frameCount: AVAudioFrameCount(window.length))
            guard Int(buffer.frameLength) == window.length, let data = buffer.floatChannelData else {
                throw CleanerError.failed("一時ファイルを読み込めませんでした。")
            }
            var finalized = [[Float]]()
            var finalStart = 0
            for c in 0..<channels {
                let samples = Array(UnsafeBufferPointer(start: data[c], count: window.length))
                // Too short for one STFT frame: keep the original samples.
                let enhanced = window.length < minimum
                    ? samples
                    : try await engine.enhanceWindow(samples, sampleRate: options.sampleRate)
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

    // MARK: - Writing

    private final class Cancellation: @unchecked Sendable {
        private let lock = NSLock()
        private var value = false
        var isCancelled: Bool { lock.lock(); defer { lock.unlock() }; return value }
        func cancel() { lock.lock(); value = true; lock.unlock() }
    }

    private func write(asset: AVURLAsset, videoTracks: [AVAssetTrack], enhanced: URL, audioStart: CMTime,
                       channels: Int, sampleRate: Int, bitRate: Int, fileType: AVFileType, to output: URL,
                       progress: @escaping @Sendable (Double) -> Void) async throws {
        let writer = try AVAssetWriter(outputURL: output, fileType: fileType)
        writer.shouldOptimizeForNetworkUse = true

        // Video: passthrough (no re-encoding).
        var videoReader: AVAssetReader?
        var videoPairs = [(AVAssetReaderTrackOutput, AVAssetWriterInput)]()
        if !videoTracks.isEmpty {
            let reader = try AVAssetReader(asset: asset)
            for track in videoTracks {
                let trackOutput = AVAssetReaderTrackOutput(track: track, outputSettings: nil)
                trackOutput.alwaysCopiesSampleData = false
                guard reader.canAdd(trackOutput) else { continue }
                reader.add(trackOutput)
                let hint = try await track.load(.formatDescriptions).first
                let input = AVAssetWriterInput(mediaType: .video, outputSettings: nil, sourceFormatHint: hint)
                input.transform = try await track.load(.preferredTransform)
                input.expectsMediaDataInRealTime = false
                guard writer.canAdd(input) else {
                    throw CleanerError.unsupported("この映像形式は\(fileType == .mp4 ? "MP4" : "このファイル形式")のまま書き出せません。")
                }
                writer.add(input)
                videoPairs.append((trackOutput, input))
            }
            videoReader = reader
        }

        // Audio: AAC from the enhanced PCM.
        let audioInput = AVAssetWriterInput(mediaType: .audio, outputSettings: [
            AVFormatIDKey: kAudioFormatMPEG4AAC,
            AVSampleRateKey: sampleRate,
            AVNumberOfChannelsKey: channels,
            AVChannelLayoutKey: Self.layoutData(channels: channels),
            AVEncoderBitRateKey: bitRate,
        ])
        audioInput.expectsMediaDataInRealTime = false
        guard writer.canAdd(audioInput) else { throw CleanerError.unsupported("AAC音声を追加できません。") }
        writer.add(audioInput)

        guard writer.startWriting() else {
            throw CleanerError.failed("書き出しを開始できませんでした: \(writer.error?.localizedDescription ?? "")")
        }
        if let videoReader, !videoReader.startReading() {
            writer.cancelWriting()
            throw CleanerError.failed("映像を読み込めませんでした: \(videoReader.error?.localizedDescription ?? "")")
        }

        // Peek at the first video sample of each track to find the session start.
        var pending = videoPairs.map { $0.0.copyNextSampleBuffer() }
        var sessionStart = CMTimeMinimum(.zero, audioStart)
        for case let sample? in pending {
            sessionStart = CMTimeMinimum(sessionStart, CMSampleBufferGetPresentationTimeStamp(sample))
        }
        writer.startSession(atSourceTime: sessionStart)

        let audioSource = try AudioSampleSource(url: enhanced, channels: channels, sampleRate: sampleRate, start: audioStart)
        let cancellation = Cancellation()
        do {
            try await withTaskCancellationHandler {
                try await withThrowingTaskGroup(of: Void.self) { group in
                    for (index, pair) in videoPairs.enumerated() {
                        let first = pending[index]
                        pending[index] = nil
                        let trackOutput = pair.0
                        let held = Box<CMSampleBuffer?>(first)
                        group.addTask {
                            try await Self.pump(pair.1, writer: writer, cancellation: cancellation, label: "video-\(index)") {
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
            videoReader?.cancelReading()
            throw error
        }
        await writer.finishWriting()
        if writer.status != .completed {
            throw CleanerError.failed("書き出しに失敗しました: \(writer.error?.localizedDescription ?? "")")
        }
        if let videoReader, videoReader.status == .failed {
            throw CleanerError.failed("映像の読み込みに失敗しました: \(videoReader.error?.localizedDescription ?? "")")
        }
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

    init(url: URL, channels: Int, sampleRate: Int, start: CMTime) throws {
        file = try AVAudioFile(forReading: url, commonFormat: .pcmFormatFloat32, interleaved: false)
        self.channels = channels
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
        var layout = AudioChannelLayout()
        layout.mChannelLayoutTag = channels == 2 ? kAudioChannelLayoutTag_Stereo : kAudioChannelLayoutTag_Mono
        var description: CMAudioFormatDescription?
        let status = CMAudioFormatDescriptionCreate(allocator: kCFAllocatorDefault, asbd: &asbd,
            layoutSize: MemoryLayout<AudioChannelLayout>.size, layout: &layout, magicCookieSize: 0, magicCookie: nil,
            extensions: nil, formatDescriptionOut: &description)
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
