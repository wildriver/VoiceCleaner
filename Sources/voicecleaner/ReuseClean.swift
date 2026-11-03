// SPDX-License-Identifier: Apache-2.0
// Command-line front end used for automated checks.
//
//   voicecleaner download
//   voicecleaner clean <input> <output> [--mono]
//   voicecleaner window <input.wav> <output.wav> [--sample-rate 48000]
import AVFoundation
import CleanerCore
import Foundation

func fail(_ message: String) -> Never {
    FileHandle.standardError.write((message + "\n").data(using: .utf8)!)
    exit(1)
}

func readMono(_ url: URL, sampleRate: Int) throws -> [Float] {
    let file = try AVAudioFile(forReading: url, commonFormat: .pcmFormatFloat32, interleaved: false)
    guard Int(file.processingFormat.sampleRate) == sampleRate, file.processingFormat.channelCount == 1 else {
        fail("window: input must be mono \(sampleRate) Hz")
    }
    guard let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length)) else {
        fail("window: cannot allocate")
    }
    try file.read(into: buffer)
    return Array(UnsafeBufferPointer(start: buffer.floatChannelData![0], count: Int(buffer.frameLength)))
}

func writeMono(_ samples: [Float], to url: URL, sampleRate: Int) throws {
    let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: Double(sampleRate), channels: 1, interleaved: false)!
    let file = try AVAudioFile(forWriting: url, settings: format.settings, commonFormat: .pcmFormatFloat32, interleaved: false)
    let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count))!
    samples.withUnsafeBufferPointer { buffer.floatChannelData![0].update(from: $0.baseAddress!, count: samples.count) }
    buffer.frameLength = AVAudioFrameCount(samples.count)
    try file.write(from: buffer)
}

final class Ref<T>: @unchecked Sendable {
    var value: T
    init(_ value: T) { self.value = value }
}

@main
struct ReuseClean {
    static func main() async {
        let arguments = Array(CommandLine.arguments.dropFirst())
        guard let command = arguments.first else {
            fail("usage: voicecleaner download | clean <input> <output> [--mono] | window <in.wav> <out.wav> [--sample-rate N]")
        }
        do {
            let provider = try ModelProvider()
            switch command {
            case "download":
                if await provider.isInstalled() { print("model already installed"); return }
                let last = Ref(-1)
                try await provider.download { fraction in
                    let percent = Int(fraction * 100)
                    if percent / 10 != last.value / 10 { last.value = percent; print("download \(percent)%") }
                }
                print("model installed")
            case "clean":
                guard arguments.count >= 3 else { fail("usage: voicecleaner clean <input> <output> [--mono]") }
                var options = CleanerOptions()
                options.keepChannels = !arguments.contains("--mono")
                let engine = provider.makeEngine()
                try await engine.prepare()
                let cleaner = MediaCleaner(engine: engine)
                let lastStage = Ref("")
                let report = try await cleaner.clean(input: URL(fileURLWithPath: arguments[1]),
                                                     output: URL(fileURLWithPath: arguments[2]), options: options) { progress in
                    if progress.stage.rawValue != lastStage.value {
                        lastStage.value = progress.stage.rawValue
                        FileHandle.standardError.write("stage \(progress.stage.rawValue)\n".data(using: .utf8)!)
                    }
                }
                let encoder = JSONEncoder()
                encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                print(String(data: try encoder.encode(report), encoding: .utf8)!)
                print(String(format: "realTimeFactor %.3f", report.realTimeFactor))
            case "window":
                guard arguments.count >= 3 else { fail("usage: voicecleaner window <in.wav> <out.wav> [--sample-rate N]") }
                var sampleRate = 48_000
                if let index = arguments.firstIndex(of: "--sample-rate"), index + 1 < arguments.count,
                   let value = Int(arguments[index + 1]) { sampleRate = value }
                let samples = try readMono(URL(fileURLWithPath: arguments[1]), sampleRate: sampleRate)
                var optimization = ReuseOptimization.recommended
                if let index = arguments.firstIndex(of: "--optimization"), index + 1 < arguments.count {
                    guard let value = ReuseOptimization(rawValue: arguments[index + 1]) else { fail("unknown optimization") }
                    optimization = value
                }
                let engine = provider.makeEngine(optimization: optimization)
                if arguments.contains("--trace") {
                    for (name, value) in try await engine.traceWindow(samples, sampleRate: sampleRate) {
                        print("trace \(name) maxabs=\(value)")
                    }
                    return
                }
                let began = Date()
                let output = try await engine.enhanceWindow(samples, sampleRate: sampleRate)
                print(String(format: "window %d samples in %.3f s", samples.count, Date().timeIntervalSince(began)))
                try writeMono(output, to: URL(fileURLWithPath: arguments[2]), sampleRate: sampleRate)
            default:
                fail("unknown command \(command)")
            }
        } catch {
            fail("error: \(error.localizedDescription)")
        }
    }
}
