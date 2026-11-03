// SPDX-License-Identifier: Apache-2.0
import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @EnvironmentObject var model: AppModel
    @State private var targeted = false
    @State private var showingNotices = false

    var body: some View {
        VStack(spacing: 16) {
            modelBanner
            dropZone
            if !model.jobs.isEmpty { jobList }
            footer
        }
        .padding(20)
        .frame(minWidth: 560, minHeight: 460)
        .sheet(isPresented: $showingNotices) { NoticesView() }
    }

    @ViewBuilder private var modelBanner: some View {
        switch model.modelState {
        case .checking:
            Label("モデルを確認しています…", systemImage: "hourglass").foregroundStyle(.secondary)
        case .downloading(let fraction):
            VStack(alignment: .leading, spacing: 6) {
                Text("初回のみ、ノイズ除去モデル（約\(model.modelDownloadMB) MB）をダウンロードしています…")
                ProgressView(value: fraction)
            }
        case .ready:
            EmptyView()
        case .failed(let message):
            VStack(alignment: .leading, spacing: 6) {
                Label("モデルを準備できませんでした", systemImage: "exclamationmark.triangle").foregroundStyle(.red)
                Text(message).font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
                Button("再試行") { Task { await model.prepareModel() } }
            }
        }
    }

    private var dropZone: some View {
        RoundedRectangle(cornerRadius: 14)
            .strokeBorder(style: StrokeStyle(lineWidth: 2, dash: [8, 6]))
            .foregroundStyle(targeted ? Color.accentColor : Color.secondary.opacity(0.5))
            .background(RoundedRectangle(cornerRadius: 14).fill(targeted ? Color.accentColor.opacity(0.08) : Color.clear))
            .overlay {
                VStack(spacing: 10) {
                    Image(systemName: "waveform.badge.minus").font(.system(size: 40)).foregroundStyle(.secondary)
                    Text("動画・音声ファイルをここにドロップ").font(.title3)
                    Text("MP4・MOV・M4A・WAV など。映像はそのまま、音声だけノイズを除去します。")
                        .font(.callout).foregroundStyle(.secondary)
                    Button("ファイルを選択…") { model.chooseFiles() }
                }
                .padding()
            }
            .frame(minHeight: 180)
            .onDrop(of: [.fileURL], isTargeted: $targeted) { providers in
                for provider in providers {
                    _ = provider.loadObject(ofClass: URL.self) { url, _ in
                        guard let url else { return }
                        Task { @MainActor in model.add([url]) }
                    }
                }
                return true
            }
    }

    private var jobList: some View {
        List {
            ForEach(model.jobs) { job in JobRow(job: job) }
        }
        .listStyle(.inset)
        .frame(minHeight: 140)
    }

    private var footer: some View {
        HStack {
            Toggle("ステレオを保持（左右を別々に処理）", isOn: $model.keepStereo)
                .help("オフにするとモノラルで書き出します。処理時間は約半分になります。")
            Spacer()
            if model.jobs.contains(where: { !$0.isActive }) {
                Button("完了した項目を消去") { model.clearFinished() }
            }
            Button("ライセンス") { showingNotices = true }
        }
        .font(.callout)
    }
}

struct JobRow: View {
    @EnvironmentObject var model: AppModel
    @ObservedObject var job: Job

    var body: some View {
        HStack(alignment: .center, spacing: 12) {
            Image(systemName: icon).foregroundStyle(color).frame(width: 20)
            VStack(alignment: .leading, spacing: 4) {
                Text(job.input.lastPathComponent).lineLimit(1).truncationMode(.middle)
                if job.state == .running {
                    ProgressView(value: job.fraction)
                }
                HStack(spacing: 8) {
                    Text(job.stageText)
                    if !job.remainingText.isEmpty { Text(job.remainingText) }
                    if !job.detailText.isEmpty { Text(job.detailText) }
                }
                .font(.caption).foregroundStyle(.secondary)
                if case .failed(let message) = job.state {
                    Text(message).font(.caption).foregroundStyle(.red).textSelection(.enabled)
                }
            }
            Spacer()
            switch job.state {
            case .finished(let url):
                Button("Finderで表示") { model.reveal(url) }
            case .waiting, .running:
                Button("キャンセル") { model.cancel(job) }
            default:
                EmptyView()
            }
        }
        .padding(.vertical, 4)
    }

    private var icon: String {
        switch job.state {
        case .waiting: return "clock"
        case .running: return "gearshape.2"
        case .finished: return "checkmark.circle.fill"
        case .failed: return "xmark.octagon.fill"
        case .cancelled: return "minus.circle"
        }
    }

    private var color: Color {
        switch job.state {
        case .finished: return .green
        case .failed: return .red
        default: return .secondary
        }
    }
}

struct NoticesView: View {
    @Environment(\.dismiss) private var dismiss
    private let text: String = {
        guard let url = Bundle.main.url(forResource: "Notices", withExtension: "txt"),
              let value = try? String(contentsOf: url, encoding: .utf8) else { return "Notices.txt が見つかりません。" }
        return value
    }()

    var body: some View {
        VStack(alignment: .trailing) {
            ScrollView {
                Text(text).font(.system(.caption, design: .monospaced)).textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            Button("閉じる") { dismiss() }.keyboardShortcut(.defaultAction)
        }
        .padding()
        .frame(width: 640, height: 480)
    }
}
