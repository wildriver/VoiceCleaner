// SPDX-License-Identifier: Apache-2.0
import AVFoundation
import CoreMedia
import Foundation

/// An `AudioChannelLayout` as the bytes AVFoundation settings expect.
struct ChannelLayout: Sendable, Equatable {
    let channels: Int
    let data: Data

    static func standard(_ channels: Int) -> ChannelLayout {
        var layout = AudioChannelLayout()
        switch channels {
        case 1: layout.mChannelLayoutTag = kAudioChannelLayoutTag_Mono
        case 2: layout.mChannelLayoutTag = kAudioChannelLayoutTag_Stereo
        default: layout.mChannelLayoutTag = kAudioChannelLayoutTag_DiscreteInOrder | UInt32(channels)
        }
        return ChannelLayout(channels: channels, data: Data(bytes: &layout, count: MemoryLayout<AudioChannelLayout>.size))
    }

    /// The layout stored in a track's format description, or a standard one.
    static func of(_ description: CMAudioFormatDescription, channels: Int) -> ChannelLayout {
        var size = 0
        guard channels > 2, let pointer = CMAudioFormatDescriptionGetChannelLayout(description, sizeOut: &size),
              size >= MemoryLayout<AudioChannelLayout>.size - MemoryLayout<AudioChannelDescription>.size else {
            return standard(channels)
        }
        if pointer.pointee.mChannelLayoutTag == kAudioChannelLayoutTag_UseChannelDescriptions {
            return ChannelLayout(channels: channels, data: Data(bytes: pointer, count: size))
        }
        // Tag or bitmap layouts are often stored without the description
        // array; AVAssetWriter only accepts the full-size struct.
        var layout = AudioChannelLayout()
        layout.mChannelLayoutTag = pointer.pointee.mChannelLayoutTag
        layout.mChannelBitmap = pointer.pointee.mChannelBitmap
        return ChannelLayout(channels: channels, data: Data(bytes: &layout, count: MemoryLayout<AudioChannelLayout>.size))
    }
}

/// The format of the audio track being cleaned.
struct SourceAudio: Sendable {
    let formatID: AudioFormatID
    let sampleRate: Int
    let layout: ChannelLayout
    /// Bit depth of PCM, ALAC and FLAC sources; nil for lossy codecs.
    let bitDepth: Int?
    let isFloat: Bool

    init(_ track: AVAssetTrack) async throws {
        guard let description = try await track.load(.formatDescriptions).first,
              var asbd = CMAudioFormatDescriptionGetStreamBasicDescription(description)?.pointee else {
            throw CleanerError.unsupported("この音声形式は読み込めません。")
        }
        // HE-AAC reports its 24 kHz core; the richest decodable format has the real rate.
        if let richest = CMAudioFormatDescriptionGetRichestDecodableFormat(description)?.pointee.mASBD,
           richest.mSampleRate > asbd.mSampleRate {
            asbd = richest
        }
        formatID = asbd.mFormatID
        sampleRate = Int(asbd.mSampleRate.rounded())
        layout = ChannelLayout.of(description, channels: max(1, Int(asbd.mChannelsPerFrame)))
        switch asbd.mFormatID {
        case kAudioFormatLinearPCM:
            bitDepth = Int(asbd.mBitsPerChannel)
            isFloat = asbd.mFormatFlags & kAudioFormatFlagIsFloat != 0
        case kAudioFormatAppleLossless, kAudioFormatFLAC:
            // Both codecs store the source bit depth in the format flags.
            bitDepth = [1: 16, 2: 20, 3: 24, 4: 32][asbd.mFormatFlags] ?? 24
            isFloat = false
        default:
            bitDepth = nil
            isFloat = false
        }
    }
}

/// How the cleaned audio is stored.
enum AudioEncoding: Equatable, Sendable {
    case aac
    case pcm(bits: Int, float: Bool, bigEndian: Bool)
    case alac(bits: Int)
    case flac(bits: Int)

    /// Audio-only lossless outputs are written with AVAudioFile at the source rate.
    var isLossless: Bool { self != .aac }

    var label: String {
        switch self {
        case .aac: return "AAC"
        case .pcm(let bits, let float, _): return "PCM \(bits)-bit\(float ? " float" : "")"
        case .alac(let bits): return "ALAC \(bits)-bit"
        case .flac(let bits): return "FLAC \(bits)-bit"
        }
    }

    /// The encoding for an output extension, keeping lossless sources lossless.
    static func choose(outputExtension: String, hasVideo: Bool, source: SourceAudio) -> AudioEncoding {
        let bits = source.bitDepth.map { $0 <= 16 ? 16 : $0 <= 24 ? 24 : 32 } ?? 24
        let float = source.isFloat && bits == 32
        if hasVideo {
            // PCM stays PCM in QuickTime; everything else becomes AAC.
            return outputExtension == "mov" && source.formatID == kAudioFormatLinearPCM
                ? .pcm(bits: bits, float: float, bigEndian: false) : .aac
        }
        switch outputExtension {
        case "wav", "caf": return .pcm(bits: bits, float: float, bigEndian: false)
        case "aiff": return .pcm(bits: min(bits, 24), float: false, bigEndian: true)
        case "flac": return .flac(bits: min(bits, 24))
        default: return source.formatID == kAudioFormatAppleLossless ? .alac(bits: min(bits, 24)) : .aac
        }
    }
}
