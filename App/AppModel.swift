// SPDX-License-Identifier: Apache-2.0
import AppKit
import CleanerCore
import Foundation
import UniformTypeIdentifiers

@MainActor
final class Job: ObservableObject, Identifiable {
    enum State: Equatable {
        case waiting, running, finished(URL), failed(String), cancelled
    }

    let id = UUID()
    let input: URL
    @Published var state: State = .waiting
    @Published var stageText = "待機中"
    @Published var fraction = 0.0
    @Published var remainingText = ""
    @Published var detailText = ""
    var task: Task<Void, Never>?

    init(input: URL) { self.input = input }

    var isActive: Bool { state == .waiting || state == .running }
}

@MainActor
final class AppModel: ObservableObject {
    enum ModelState: Equatable {
        case checking, downloading(Double), ready, failed(String)
    }

    @Published var modelState: ModelState = .checking
    @Published var jobs: [Job] = []
    @Published var keepStereo = true

    private var provider: ModelProvider?
    private var engine: ReuseEngine?
    private var running = false

    static let acceptedTypes: [UTType] = [.movie, .audio, .mpeg4Movie, .quickTimeMovie, .mpeg4Audio]

    init() {
        Task { await prepareModel() }
    }

    func prepareModel() async {
        modelState = .checking
        do {
            let provider = try self.provider ?? ModelProvider()
            self.provider = provider
            if !(await provider.isInstalled()) {
                modelState = .downloading(0)
                try await provider.download { fraction in
                    Task { @MainActor in self.modelState = .downloading(fraction) }
                }
            }
            let engine = provider.makeEngine()
            try await engine.prepare()
            self.engine = engine
            modelState = .ready
            startNext()
        } catch {
            modelState = .failed(error.localizedDescription)
        }
    }

    var modelDownloadMB: Int {
        Int((provider?.downloadBytes ?? 54_206_689) / 1_000_000)
    }

    // MARK: Queue

    func add(_ urls: [URL]) {
        for url in urls where !jobs.contains(where: { $0.input == url && $0.isActive }) {
            jobs.append(Job(input: url))
        }
        startNext()
    }

    func chooseFiles() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = true
        panel.canChooseDirectories = false
        panel.allowedContentTypes = Self.acceptedTypes
        panel.message = "ノイズを除去する動画または音声ファイルを選んでください"
        if panel.runModal() == .OK { add(panel.urls) }
    }

    func cancel(_ job: Job) {
        if job.state == .waiting {
            job.state = .cancelled
            job.stageText = "キャンセルしました"
        } else {
            job.task?.cancel()
        }
    }

    func clearFinished() {
        jobs.removeAll { !$0.isActive }
    }

    func reveal(_ url: URL) {
        NSWorkspace.shared.activateFileViewerSelecting([url])
    }

    private func startNext() {
        guard !running, let engine, modelState == .ready,
              let job = jobs.first(where: { $0.state == .waiting }) else { return }
        running = true
        job.state = .running
        var options = CleanerOptions()
        options.keepStereo = keepStereo
        job.task = Task { [weak self] in
            await self?.run(job, engine: engine, options: options)
            guard let self else { return }
            self.running = false
            self.startNext()
        }
    }

    private func run(_ job: Job, engine: ReuseEngine, options: CleanerOptions) async {
        let activity = ProcessInfo.processInfo.beginActivity(
            options: [.userInitiated, .idleSystemSleepDisabled], reason: "ノイズ除去の処理中")
        defer { ProcessInfo.processInfo.endActivity(activity) }
        let input = job.input
        let accessing = input.startAccessingSecurityScopedResource()
        defer { if accessing { input.stopAccessingSecurityScopedResource() } }
        do {
            let hasVideo = await MediaCleaner.hasVideo(input)
            let output = try Self.outputURL(for: input, hasVideo: hasVideo)
            let cleaner = MediaCleaner(engine: engine)
            let began = Date()
            let report = try await cleaner.clean(input: input, output: output, options: options) { progress in
                Task { @MainActor in
                    job.fraction = progress.fraction
                    switch progress.stage {
                    case .decoding: job.stageText = "音声を読み込み中"
                    case .enhancing: job.stageText = "ノイズを除去中"
                    case .writing: job.stageText = "ファイルを書き出し中"
                    case .done: job.stageText = "完了"
                    }
                    let elapsed = Date().timeIntervalSince(began)
                    if progress.stage == .enhancing, progress.fraction > 0.08 {
                        let remaining = elapsed / progress.fraction * (1 - progress.fraction)
                        job.remainingText = "残り約" + Self.format(remaining)
                    } else if progress.stage != .enhancing {
                        job.remainingText = ""
                    }
                }
            }
            job.state = .finished(output)
            job.stageText = "完了"
            job.fraction = 1
            job.remainingText = ""
            job.detailText = String(format: "%@ · %@ · 処理時間 %@",
                                    report.channels == 2 ? "ステレオ" : "モノラル",
                                    "\(report.sampleRate / 1000) kHz",
                                    Self.format(report.totalSeconds))
        } catch is CancellationError {
            job.state = .cancelled
            job.stageText = "キャンセルしました"
            job.remainingText = ""
        } catch {
            job.state = .failed(error.localizedDescription)
            job.stageText = "エラー"
            job.remainingText = ""
        }
    }

    /// `<name>_clean.<ext>` next to the input, or in ~/Movies/RE-USE Cleaner
    /// when the input folder is not writable.
    static func outputURL(for input: URL, hasVideo: Bool) throws -> URL {
        let (_, ext) = MediaCleaner.outputType(for: input, hasVideo: hasVideo)
        var folder = input.deletingLastPathComponent()
        if !FileManager.default.isWritableFile(atPath: folder.path) {
            folder = FileManager.default.urls(for: .moviesDirectory, in: .userDomainMask)[0]
                .appendingPathComponent("RE-USE Cleaner", isDirectory: true)
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        }
        let base = input.deletingPathExtension().lastPathComponent + "_clean"
        var candidate = folder.appendingPathComponent(base).appendingPathExtension(ext)
        var counter = 2
        while FileManager.default.fileExists(atPath: candidate.path) {
            candidate = folder.appendingPathComponent("\(base) \(counter)").appendingPathExtension(ext)
            counter += 1
        }
        return candidate
    }

    static func format(_ seconds: Double) -> String {
        let value = max(0, Int(seconds.rounded()))
        if value >= 3600 { return "\(value / 3600)時間\((value % 3600) / 60)分" }
        if value >= 60 { return "\(value / 60)分\(value % 60)秒" }
        return "\(value)秒"
    }
}
