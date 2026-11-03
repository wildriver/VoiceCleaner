// SPDX-License-Identifier: Apache-2.0
import AVFoundation
import XCTest
@testable import CleanerCore

final class MediaCleanerTests: XCTestCase {
    func testOutputKeepsContainer() {
        let cases: [(String, Bool, String)] = [
            ("a.mov", true, "mov"), ("a.M4V", true, "m4v"), ("a.avi", true, "mp4"), ("a.mp4", true, "mp4"),
            ("a.wav", false, "wav"), ("a.flac", false, "flac"), ("a.caf", false, "caf"), ("a.aif", false, "aiff"),
            ("a.m4a", false, "m4a"), ("a.mp3", false, "m4a"), ("a.ogg", false, "m4a"),
        ]
        for (name, hasVideo, ext) in cases {
            XCTAssertEqual(MediaCleaner.outputExtension(for: URL(fileURLWithPath: name), hasVideo: hasVideo), ext, name)
        }
        XCTAssertEqual(CleanerOptions().processingRate(for: 44_100), 44_100)
        XCTAssertEqual(CleanerOptions().processingRate(for: 96_000), 48_000)
        XCTAssertEqual(CleanerOptions().processingRate(for: 6_000), 8_000)
    }

    func testReadFillsWholeWindow() throws {
        // AVAudioFile.read returns short reads for mono CAF files; windows must still be filled exactly.
        for channels in [1, 2] {
            let url = FileManager.default.temporaryDirectory.appendingPathComponent("read-\(UUID().uuidString).caf")
            defer { try? FileManager.default.removeItem(at: url) }
            let total = 480_000
            let signal = (0..<total).map { Float(sin(Double($0) * 0.001)) }
            let writer = try AVAudioFile(forWriting: url,
                settings: MediaCleaner.pcmSettings(layout: .standard(channels), sampleRate: 48_000, interleaved: false),
                commonFormat: .pcmFormatFloat32, interleaved: false)
            let pcm = try XCTUnwrap(AVAudioPCMBuffer(pcmFormat: writer.processingFormat, frameCapacity: AVAudioFrameCount(total)))
            pcm.frameLength = AVAudioFrameCount(total)
            for c in 0..<channels { signal.withUnsafeBufferPointer { pcm.floatChannelData![c].update(from: $0.baseAddress!, count: total) } }
            try writer.write(from: pcm)
            writer.close()

            let reader = try AVAudioFile(forReading: url, commonFormat: .pcmFormatFloat32, interleaved: false)
            let buffer = try XCTUnwrap(AVAudioPCMBuffer(pcmFormat: reader.processingFormat, frameCapacity: 240_000))
            for start in [0, 120_000, 240_000] {
                let window = try MediaCleaner.read(reader, start: start, length: 240_000, using: buffer)
                XCTAssertEqual(window.count, channels)
                for values in window {
                    XCTAssertEqual(values.count, 240_000, "channels \(channels), start \(start)")
                    XCTAssertEqual(values, Array(signal[start..<(start + 240_000)]), "channels \(channels), start \(start)")
                }
            }
        }
    }
}
