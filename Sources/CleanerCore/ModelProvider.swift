// SPDX-License-Identifier: Apache-2.0
import Foundation
@_exported import ReuseKit
import ReuseModelAssets

/// Locates, downloads and opens the RE-USE model pack (MLX weights converted
/// by the RE-USE-MLX-CoreML project from NVIDIA's OpenMDW-1.1 checkpoint).
public final class ModelProvider: Sendable {
    public let store: ReuseModelStore
    public let source: ReuseModelSource

    public init(applicationSupport: URL? = nil, caches: URL? = nil) throws {
        let fm = FileManager.default
        let support = applicationSupport
            ?? fm.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0].appendingPathComponent("VoiceCleaner")
        let cache = caches
            ?? fm.urls(for: .cachesDirectory, in: .userDomainMask)[0].appendingPathComponent("VoiceCleaner")
        // Earlier builds were named "RE-USE Cleaner": reuse an already downloaded model.
        if applicationSupport == nil {
            let legacy = support.deletingLastPathComponent().appendingPathComponent("RE-USE Cleaner")
            if !fm.fileExists(atPath: support.path), fm.fileExists(atPath: legacy.path) {
                try? fm.moveItem(at: legacy, to: support)
            }
        }
        try fm.createDirectory(at: support, withIntermediateDirectories: true)
        try fm.createDirectory(at: cache, withIntermediateDirectories: true)
        guard let catalogURL = Bundle.module.url(forResource: "ModelCatalog", withExtension: "json"),
              let sourceURL = Bundle.module.url(forResource: "ModelSource", withExtension: "json") else {
            throw ReuseError.invalid("モデル情報がアプリに含まれていません。")
        }
        let catalog = try JSONDecoder().decode(ReuseModelCatalog.self, from: Data(contentsOf: catalogURL))
        struct Source: Decodable { let repository: String; let revision: String }
        let config = try JSONDecoder().decode(Source.self, from: Data(contentsOf: sourceURL))
        store = try ReuseModelStore(catalog: catalog, applicationSupportDirectory: support, cachesDirectory: cache)
        source = try ReuseModelSource(huggingFaceRepository: config.repository, revision: config.revision)
    }

    public var downloadBytes: Int64 { store.catalog.totalBytes }

    public func isInstalled() async -> Bool {
        (try? await store.inspect().isInstalled) ?? false
    }

    public func download(progress: @escaping @Sendable (Double) -> Void = { _ in }) async throws {
        try await store.download(from: source) { progress($0.fraction) }
    }

    /// Creates an engine for the installed model. The engine loads lazily.
    public func makeEngine(optimization: ReuseOptimization = .recommended) -> ReuseEngine {
        ReuseEngine(modelPack: ReuseModelPack(root: store.installedModelDirectory),
                    cacheDirectory: store.compilationCacheDirectory, optimization: optimization)
    }
}
