// SPDX-License-Identifier: Apache-2.0
import Foundation

public struct ReuseModelTransferProgress: Sendable {
    public enum Phase: Sendable, Equatable { case downloading, verifying }
    public let phase: Phase
    public let completedBytes: Int64
    public let totalBytes: Int64
    public let completedFiles: Int
    public let totalFiles: Int
    public var fraction: Double { min(1, Double(completedBytes) / Double(max(totalBytes, 1))) }
}

public struct ReuseModelStorageStatus: Sendable {
    public let isInstalled: Bool
    public let storedBytes: Int64
    public let compiledBytes: Int64
    public var totalBytes: Int64 { storedBytes + compiledBytes }
}

/// Owns only REUSEModels subdirectories; recordings and exported audio are outside its scope.
public actor ReuseModelStore {
    public nonisolated let catalog: ReuseModelCatalog
    public nonisolated let installedModelDirectory: URL
    public nonisolated let compilationCacheDirectory: URL
    private let storageRoot: URL
    private let cacheRoot: URL
    private let staging: URL
    private var installing = false
    private let fm = FileManager.default

    public init(catalog: ReuseModelCatalog, applicationSupportDirectory: URL, cachesDirectory: URL) throws {
        try catalog.validate()
        self.catalog = catalog
        storageRoot = applicationSupportDirectory.appendingPathComponent("REUSEModels", isDirectory: true)
        cacheRoot = cachesDirectory.appendingPathComponent("REUSEModels", isDirectory: true)
        installedModelDirectory = storageRoot.appendingPathComponent(catalog.identifier, isDirectory: true)
        staging = storageRoot.appendingPathComponent("incoming-" + catalog.identifier, isDirectory: true)
        compilationCacheDirectory = cacheRoot.appendingPathComponent(catalog.identifier, isDirectory: true)
    }

    public func inspect() throws -> ReuseModelStorageStatus {
        let ready = try catalog.files.allSatisfy { try matches($0, in: installedModelDirectory) }
        return ReuseModelStorageStatus(isInstalled: ready, storedBytes: Self.bytes(in: storageRoot),
                                       compiledBytes: Self.bytes(in: cacheRoot))
    }

    public func download(from source: ReuseModelSource,
                         progress: @escaping @Sendable (ReuseModelTransferProgress) -> Void = { _ in }) async throws {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = 60
        configuration.timeoutIntervalForResource = 30 * 60
        let client = FileDownloadClient(configuration: configuration)
        defer { client.invalidate() }
        try await install(progress: progress) { file, destination, update in
            do {
                try await client.download(source.request(for: file.path), to: destination, maximum: file.bytes, update: update)
            } catch {
                if Task.isCancelled { throw CancellationError() }
                throw error
            }
        }
    }

    /// Useful for offline development or importing an already downloaded model pack.
    public func importLocalPack(from root: URL,
                                progress: @escaping @Sendable (ReuseModelTransferProgress) -> Void = { _ in }) async throws {
        let base = root.resolvingSymlinksInPath().standardizedFileURL
        try await install(progress: progress) { file, destination, update in
            let input = base.appendingPathComponent(file.path).resolvingSymlinksInPath().standardizedFileURL
            guard input.path.hasPrefix(base.path + "/") else {
                throw ReuseAssetError.invalid("モデルの参照先が不正です。")
            }
            try Task.checkCancellation()
            try FileManager.default.copyItem(at: input, to: destination)
            update(file.bytes)
        }
    }

    public func remove() throws {
        guard !installing else { throw ReuseAssetError.invalid("ダウンロードを停止してから削除してください。") }
        for root in [storageRoot, cacheRoot] where fm.fileExists(atPath: root.path) {
            try fm.removeItem(at: root)
        }
    }

    private func install(progress: @escaping @Sendable (ReuseModelTransferProgress) -> Void,
                         fetch: @escaping @Sendable (ReuseModelCatalog.File, URL, @escaping @Sendable (Int64) -> Void) async throws -> Void) async throws {
        guard !installing else { throw ReuseAssetError.invalid("モデルはダウンロード中です。") }
        installing = true
        defer { installing = false }
        if try inspect().isInstalled { return }
        try fm.createDirectory(at: staging, withIntermediateDirectories: true)
        var excludedRoot = storageRoot
        var resourceValues = URLResourceValues(); resourceValues.isExcludedFromBackup = true
        try excludedRoot.setResourceValues(resourceValues)
        let partials = staging.appendingPathComponent(".transfers", isDirectory: true)
        if fm.fileExists(atPath: partials.path) { try fm.removeItem(at: partials) }
        try fm.createDirectory(at: partials, withIntermediateDirectories: true)
        var completed: Int64 = 0, completedFiles = 0
        var valid = Set<String>()
        for file in catalog.files where try matches(file, in: staging) {
            completed += file.bytes; completedFiles += 1; valid.insert(file.path)
        }
        let free = (try fm.attributesOfFileSystem(forPath: staging.path)[.systemFreeSize] as? NSNumber)?.int64Value ?? 0
        guard free >= catalog.totalBytes - completed + 32 * 1024 * 1024 else {
            throw ReuseAssetError.invalid("モデルを保存する空き容量が不足しています。")
        }
        for file in catalog.files where !valid.contains(file.path) {
            try Task.checkCancellation()
            let final = staging.appendingPathComponent(file.path)
            let temporary = partials.appendingPathComponent(UUID().uuidString)
            defer { try? fm.removeItem(at: temporary) }
            let before = completed, count = completedFiles, catalog = self.catalog
            progress(ReuseModelTransferProgress(phase: .downloading, completedBytes: before, totalBytes: catalog.totalBytes,
                completedFiles: count, totalFiles: catalog.files.count))
            try await fetch(file, temporary) { bytes in
                progress(ReuseModelTransferProgress(phase: .downloading, completedBytes: before + min(bytes, file.bytes),
                    totalBytes: catalog.totalBytes, completedFiles: count, totalFiles: catalog.files.count))
            }
            try Task.checkCancellation()
            progress(ReuseModelTransferProgress(phase: .verifying, completedBytes: before + file.bytes,
                totalBytes: catalog.totalBytes, completedFiles: count, totalFiles: catalog.files.count))
            guard try matches(file, at: temporary) else { throw ReuseAssetError.invalid("モデルの検証に失敗しました。再試行してください。") }
            try fm.createDirectory(at: final.deletingLastPathComponent(), withIntermediateDirectories: true)
            if fm.fileExists(atPath: final.path) { try fm.removeItem(at: final) }
            try fm.moveItem(at: temporary, to: final)
            completed += file.bytes; completedFiles += 1
        }
        try Task.checkCancellation()
        try fm.removeItem(at: partials)
        // A complete, verified directory becomes visible in one rename on the same volume.
        if fm.fileExists(atPath: installedModelDirectory.path) { try fm.removeItem(at: installedModelDirectory) }
        try fm.moveItem(at: staging, to: installedModelDirectory)
        progress(ReuseModelTransferProgress(phase: .verifying, completedBytes: catalog.totalBytes,
            totalBytes: catalog.totalBytes, completedFiles: catalog.files.count, totalFiles: catalog.files.count))
    }

    private func matches(_ file: ReuseModelCatalog.File, in directory: URL) throws -> Bool {
        try matches(file, at: directory.appendingPathComponent(file.path))
    }
    private func matches(_ file: ReuseModelCatalog.File, at url: URL) throws -> Bool {
        guard fm.fileExists(atPath: url.path),
              (try url.resourceValues(forKeys: [.isRegularFileKey])).isRegularFile == true,
              (try fm.attributesOfItem(atPath: url.path)[.size] as? NSNumber)?.int64Value == file.bytes else { return false }
        return try ReuseModelCatalog.digest(file: url) == file.sha256
    }
    private static func bytes(in root: URL) -> Int64 {
        guard let values = FileManager.default.enumerator(at: root, includingPropertiesForKeys: [.isRegularFileKey, .fileSizeKey]) else { return 0 }
        return values.reduce(Int64(0)) { total, item in
            guard let url = item as? URL, let value = try? url.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey]),
                  value.isRegularFile == true else { return total }
            return total + Int64(value.fileSize ?? 0)
        }
    }
}

enum ReuseDownloadRedirect {
    static func request(_ proposed: URLRequest, original: URLRequest) -> URLRequest? {
        guard let target = proposed.url, let origin = original.url,
              target.user == nil, target.password == nil,
              target.scheme == "https" || (origin.scheme == "http" && target.scheme == "http") else { return nil }
        var result = proposed
        result.setValue(nil, forHTTPHeaderField: "Authorization")
        let port: (URL) -> Int = { $0.port ?? ($0.scheme == "https" ? 443 : 80) }
        if target.scheme == origin.scheme, target.host?.lowercased() == origin.host?.lowercased(), port(target) == port(origin) {
            result.setValue(original.value(forHTTPHeaderField: "Authorization"), forHTTPHeaderField: "Authorization")
        }
        return result
    }
}

private final class FileDownloadClient: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    private final class Pending {
        let destination: URL
        let maximum: Int64
        let update: @Sendable (Int64) -> Void
        let continuation: CheckedContinuation<Void, Error>
        // Delegate callbacks are serialized on one operation queue.
        var failure: Error?
        var moved = false
        init(destination: URL, maximum: Int64, update: @escaping @Sendable (Int64) -> Void,
             continuation: CheckedContinuation<Void, Error>) {
            self.destination = destination; self.maximum = maximum; self.update = update; self.continuation = continuation
        }
    }
    private final class Cancellation: @unchecked Sendable {
        let lock = NSLock()
        var task: URLSessionTask?
        var cancelled = false
        func attach(_ task: URLSessionTask) {
            lock.lock(); self.task = task; let stop = cancelled; lock.unlock()
            if stop { task.cancel() }
        }
        func cancel() {
            lock.lock(); cancelled = true; let task = task; lock.unlock()
            task?.cancel()
        }
    }
    private let lock = NSLock()
    private var pending: [Int: Pending] = [:]
    private var session: URLSession!
    init(configuration: URLSessionConfiguration) {
        super.init()
        let queue = OperationQueue(); queue.maxConcurrentOperationCount = 1
        session = URLSession(configuration: configuration, delegate: self, delegateQueue: queue)
    }
    func invalidate() { session.invalidateAndCancel() }
    private func value(for task: URLSessionTask) -> Pending? {
        lock.lock(); defer { lock.unlock() }; return pending[task.taskIdentifier]
    }
    func download(_ request: URLRequest, to destination: URL, maximum: Int64,
                  update: @escaping @Sendable (Int64) -> Void) async throws {
        try Task.checkCancellation()
        let cancellation = Cancellation()
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
                let task = session.downloadTask(with: request)
                let value = Pending(destination: destination, maximum: maximum, update: update, continuation: continuation)
                lock.lock(); pending[task.taskIdentifier] = value; lock.unlock()
                cancellation.attach(task); task.resume()
            }
        } onCancel: { cancellation.cancel() }
    }
    func urlSession(_ session: URLSession, task: URLSessionTask,
                    willPerformHTTPRedirection response: HTTPURLResponse, newRequest request: URLRequest,
                    completionHandler: @escaping (URLRequest?) -> Void) {
        guard let original = task.originalRequest else { completionHandler(nil); return }
        completionHandler(ReuseDownloadRedirect.request(request, original: original))
    }
    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {
        guard let value = value(for: downloadTask) else { return }
        if value.failure != nil { return }
        guard let response = downloadTask.response as? HTTPURLResponse, (200...299).contains(response.statusCode) else {
            value.failure = ReuseAssetError.invalid("モデルを取得できませんでした（HTTP \((downloadTask.response as? HTTPURLResponse)?.statusCode ?? 0)）。")
            return
        }
        do {
            try FileManager.default.moveItem(at: location, to: value.destination)
            value.moved = true
        } catch { value.failure = error }
    }
    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didWriteData bytesWritten: Int64,
                    totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64) {
        guard let value = value(for: downloadTask) else { return }
        if let response = downloadTask.response as? HTTPURLResponse, !(200...299).contains(response.statusCode) {
            value.failure = ReuseAssetError.invalid("モデルを取得できませんでした（HTTP \(response.statusCode)）。")
            downloadTask.cancel(); return
        }
        if totalBytesWritten > value.maximum || totalBytesExpectedToWrite > value.maximum {
            value.failure = ReuseAssetError.invalid("モデルのファイルサイズが配布情報と一致しません。")
            downloadTask.cancel()
        } else { value.update(totalBytesWritten) }
    }
    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        lock.lock(); let value = pending.removeValue(forKey: task.taskIdentifier); lock.unlock()
        guard let value else { return }
        if let error = value.failure ?? error { value.continuation.resume(throwing: error) }
        else if value.moved { value.continuation.resume() }
        else { value.continuation.resume(throwing: ReuseAssetError.invalid("ダウンロードを完了できませんでした。")) }
    }
}
