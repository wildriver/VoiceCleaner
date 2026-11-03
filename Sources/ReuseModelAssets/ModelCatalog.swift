// SPDX-License-Identifier: Apache-2.0
import Foundation
import CryptoKit

public enum ReuseAssetError: LocalizedError {
    case invalid(String)
    public var errorDescription: String? {
        switch self { case .invalid(let message): return message }
    }
}

public struct ReuseModelCatalog: Codable, Sendable {
    public struct File: Codable, Sendable {
        public let path: String
        public let bytes: Int64
        public let sha256: String
        public init(path: String, bytes: Int64, sha256: String) {
            self.path = path; self.bytes = bytes; self.sha256 = sha256
        }
    }
    public let version: Int
    public let name: String
    public let files: [File]
    public init(version: Int = 1, name: String, files: [File]) throws {
        self.version = version; self.name = name; self.files = files
        try validate()
    }
    public func validate() throws {
        guard version == 1, !name.isEmpty, !files.isEmpty, files.count <= 512 else {
            throw ReuseAssetError.invalid("モデル配布情報の形式が不正です。")
        }
        var paths = Set<String>(), size: Int64 = 0
        for file in files {
            let parts = file.path.split(separator: "/", omittingEmptySubsequences: false)
            let validPath = !parts.isEmpty && parts.first != ".transfers" && parts.allSatisfy {
                !$0.isEmpty && $0 != "." && $0 != ".." && !$0.contains("\\") && !$0.contains(":") && !$0.contains("\0")
            }
            let sum = size.addingReportingOverflow(file.bytes)
            guard validPath, paths.insert(file.path).inserted, file.bytes > 0,
                  file.sha256.count == 64, file.sha256.allSatisfy({ "0123456789abcdef".contains($0) }),
                  !sum.overflow, sum.partialValue <= 4 * 1024 * 1024 * 1024 else {
                throw ReuseAssetError.invalid("モデルのファイル一覧が不正です。")
            }
            size = sum.partialValue
        }
        // Prevent a file and a directory from sharing the same path.
        for path in paths {
            var parent = (path as NSString).deletingLastPathComponent
            while !parent.isEmpty {
                guard !paths.contains(parent) else { throw ReuseAssetError.invalid("モデルのパスが重複しています。") }
                parent = (parent as NSString).deletingLastPathComponent
            }
        }
    }
    public var totalBytes: Int64 { files.reduce(0) { $0 + $1.bytes } }
    public var identifier: String {
        let value = files.sorted { $0.path < $1.path }.map { "\($0.path)\t\($0.bytes)\t\($0.sha256)" }.joined(separator: "\n")
        return String(Self.digest(Data(value.utf8)).prefix(24))
    }
    static func digest(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
    static func digest(file url: URL) throws -> String {
        let stream = try FileHandle(forReadingFrom: url)
        defer { try? stream.close() }
        var digest = SHA256()
        while let bytes = try stream.read(upToCount: 1024 * 1024), !bytes.isEmpty {
            try Task.checkCancellation()
            digest.update(data: bytes)
        }
        return digest.finalize().map { String(format: "%02x", $0) }.joined()
    }
}

/// The caller chooses the repository. No credentials or placeholder repo are embedded.
public struct ReuseModelSource: Sendable {
    public let repository: String
    public let revision: String
    let baseURL: URL
    private let accessToken: String?
    public init(huggingFaceRepository: String, revision: String, accessToken: String? = nil) throws {
        let components = huggingFaceRepository.split(separator: "/", omittingEmptySubsequences: false)
        let allowed = Set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
        guard components.count == 2, components.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." && $0.allSatisfy(allowed.contains) }),
              !revision.isEmpty, revision != ".", revision != "..", revision.allSatisfy(allowed.contains) else {
            throw ReuseAssetError.invalid("Hugging Faceの配布先が不正です。")
        }
        repository = huggingFaceRepository; self.revision = revision
        let token = accessToken?.trimmingCharacters(in: .whitespacesAndNewlines)
        guard token?.utf8.allSatisfy({ (33...126).contains($0) }) ?? true else {
            throw ReuseAssetError.invalid("Hugging Faceトークンの形式が不正です。")
        }
        self.accessToken = token?.isEmpty == false ? token : nil
        baseURL = URL(string: "https://huggingface.co")!
            .appendingPathComponent(repository).appendingPathComponent("resolve").appendingPathComponent(revision)
    }
    // Internal initializer for local HTTP integration tests.
    init(baseURL: URL, accessToken: String? = nil) {
        self.baseURL = baseURL; self.accessToken = accessToken; repository = "test/local"; revision = "test"
    }
    func url(for path: String) -> URL {
        path.split(separator: "/").reduce(baseURL) { $0.appendingPathComponent(String($1)) }
    }
    func request(for path: String) -> URLRequest {
        var request = URLRequest(url: url(for: path))
        if let accessToken { request.setValue("Bearer " + accessToken, forHTTPHeaderField: "Authorization") }
        return request
    }
}
