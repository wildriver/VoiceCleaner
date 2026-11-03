// SPDX-License-Identifier: Apache-2.0
import AppKit
import Foundation

/// Folders the cleaned files may be written to. A sandboxed app (the App Store
/// build) can only write where the user allowed it, so the folder is asked for
/// once and remembered as a security-scoped bookmark.
@MainActor
final class OutputFolders {
    private static let key = "OutputFolderBookmarks"
    private static let limit = 30
    private var bookmarks: [Data]

    init() {
        bookmarks = UserDefaults.standard.array(forKey: Self.key) as? [Data] ?? []
        // Keep access for the life of the app; the folders are written to again and again.
        bookmarks = bookmarks.filter { data in
            var stale = false
            guard let url = try? URL(resolvingBookmarkData: data, options: .withSecurityScope,
                                     bookmarkDataIsStale: &stale), !stale else { return false }
            return url.startAccessingSecurityScopedResource()
        }
        UserDefaults.standard.set(bookmarks, forKey: Self.key)
    }

    /// The input's own folder when it is writable, otherwise a folder the user picks.
    func folder(for input: URL) -> URL? {
        let folder = input.deletingLastPathComponent()
        if FileManager.default.isWritableFile(atPath: folder.path) { return folder }
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.canCreateDirectories = true
        panel.directoryURL = folder
        panel.prompt = "このフォルダに保存"
        panel.message = "「\(input.lastPathComponent)」のノイズ除去後のファイルを保存するフォルダを選んでください。"
            + "選んだフォルダには、次からは確認なしで保存します。"
        guard panel.runModal() == .OK, let chosen = panel.url else { return nil }
        remember(chosen)
        return chosen
    }

    private func remember(_ folder: URL) {
        guard let data = try? folder.bookmarkData(options: .withSecurityScope),
              folder.startAccessingSecurityScopedResource() else { return }
        bookmarks = Array(([data] + bookmarks).prefix(Self.limit))
        UserDefaults.standard.set(bookmarks, forKey: Self.key)
    }
}
