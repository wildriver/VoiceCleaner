// SPDX-License-Identifier: Apache-2.0
import SwiftUI

@main
struct CleanerApp: App {
    @StateObject private var model = AppModel()

    var body: some Scene {
        // One window: files opened from Finder join its queue instead of opening another window.
        Window("VoiceCleaner", id: "main") {
            ContentView()
                .environmentObject(model)
                .onOpenURL { model.add([$0]) }
        }
        .windowResizability(.contentMinSize)
        .commands {
            CommandGroup(replacing: .newItem) {
                Button("ファイルを開く…") { model.chooseFiles() }.keyboardShortcut("o")
            }
        }
    }
}
