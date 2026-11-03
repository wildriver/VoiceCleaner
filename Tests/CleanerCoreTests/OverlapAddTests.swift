// SPDX-License-Identifier: Apache-2.0
import XCTest
@testable import CleanerCore

final class OverlapAddTests: XCTestCase {
    func testPlanMatchesReferenceScript() {
        // 12 s at 1 kHz, 5 s window, 2.5 s hop: starts 0, 2.5, 5 and the tail aligned to the end.
        let plan = WindowPlan(total: 12_000, windowLength: 5_000, hop: 2_500)
        XCTAssertEqual(plan.windows.map(\.start), [0, 2_500, 5_000, 7_000])
        XCTAssertTrue(plan.windows.allSatisfy { $0.length == 5_000 })
        XCTAssertEqual(WindowPlan(total: 3_000, windowLength: 5_000, hop: 2_500).windows,
                       [WindowPlan.Window(start: 0, length: 3_000)])
        XCTAssertEqual(WindowPlan(total: 5_000, windowLength: 5_000, hop: 2_500).windows.count, 1)
    }

    func testIdentityReconstruction() {
        for total in [1, 999, 5_000, 5_001, 12_000, 17_777] {
            let signal = (0..<total).map { Float(sin(Double($0) * 0.01)) }
            let plan = WindowPlan(total: total, windowLength: 5_000, hop: 2_500)
            let stream = OverlapAddStream(plan: plan)
            var output = [Float]()
            for window in plan.windows {
                let (start, values) = stream.add(Array(signal[window.start..<(window.start + window.length)]))
                XCTAssertEqual(start, output.count)
                output += values
            }
            XCTAssertTrue(stream.isComplete)
            XCTAssertEqual(output.count, total)
            let error = zip(output, signal).map { abs($0 - $1) }.max() ?? 0
            XCTAssertLessThan(error, 1e-4, "total \(total)")
        }
    }
}
