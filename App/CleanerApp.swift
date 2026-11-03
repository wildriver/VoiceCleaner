// SPDX-License-Identifier: Apache-2.0
import AppKit
import SwiftUI

/// Receives files opened from Finder; `onOpenURL` would only pass the first of several.
final class AppDelegate: NSObject, NSApplicationDelegate {
    @MainActor let model = AppModel()

    @MainActor func application(_ application: NSApplication, open urls: [URL]) {
        model.add(urls)
    }
}

@main
struct CleanerApp: App {
    @NSApplicationDelegateAdaptor private var delegate: AppDelegate

    var body: some Scene {
        // One window: files opened from Finder join its queue instead of opening another window.
        Window("VoiceCleaner", id: "main") {
            ContentView()
                .environmentObject(delegate.model)
        }
        .windowResizability(.contentMinSize)
        .commands {
            CommandGroup(replacing: .newItem) {
                Button("ファイルを開く…") { delegate.model.chooseFiles() }.keyboardShortcut("o")
            }
        }
    }
}
