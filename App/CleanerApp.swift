// SPDX-License-Identifier: Apache-2.0
import SwiftUI

@main
struct CleanerApp: App {
    @StateObject private var model = AppModel()

    var body: some Scene {
        WindowGroup("RE-USE Cleaner") {
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
