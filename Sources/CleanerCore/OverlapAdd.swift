// SPDX-License-Identifier: Apache-2.0
import Foundation

/// Window placement identical to NVIDIA's `inference_chunk.py`:
/// windows start every `hop` samples, and the last window is aligned to the
/// end of the signal. A signal shorter than one window is a single window.
public struct WindowPlan: Sendable, Equatable {
    public struct Window: Sendable, Equatable {
        public let start: Int
        public let length: Int
    }

    public let total: Int
    public let windowLength: Int
    public let hop: Int
    public let windows: [Window]

    public init(total: Int, windowLength: Int, hop: Int) {
        precondition(total > 0 && windowLength > 0 && hop > 0 && hop <= windowLength)
        self.total = total
        self.windowLength = windowLength
        self.hop = hop
        if total <= windowLength {
            windows = [Window(start: 0, length: total)]
        } else {
            let count = (total - windowLength + hop - 1) / hop + 1
            var result = (0..<(count - 1)).map { Window(start: $0 * hop, length: windowLength) }
            result.append(Window(start: total - windowLength, length: windowLength))
            windows = result
        }
    }
}

/// Streaming weighted overlap-add with a Hann window (clamped to 1e-7, as in
/// the reference script). Only about one window of state is kept, so very
/// long recordings can be processed with bounded memory.
public final class OverlapAddStream {
    public let plan: WindowPlan
    private let weights: [Float]
    private var base = 0
    private var accumulation: [Float] = []
    private var normalization: [Float] = []
    private var nextIndex = 0

    public init(plan: WindowPlan) {
        self.plan = plan
        let n = plan.windowLength
        weights = (0..<n).map { i in
            max(Float(0.5 - 0.5 * cos(2 * Double.pi * Double(i) / Double(n))), 1e-7)
        }
    }

    /// Adds the processed samples of the next window in plan order and returns
    /// the samples that are now final (they start at `start`).
    public func add(_ values: [Float]) -> (start: Int, samples: [Float]) {
        precondition(nextIndex < plan.windows.count, "all windows were already added")
        let window = plan.windows[nextIndex]
        precondition(values.count == window.length, "window length mismatch")
        let end = window.start + window.length
        let needed = end - base
        if accumulation.count < needed {
            accumulation += [Float](repeating: 0, count: needed - accumulation.count)
            normalization += [Float](repeating: 0, count: needed - normalization.count)
        }
        let offset = window.start - base
        for i in 0..<window.length {
            let w = weights[i]
            accumulation[offset + i] += values[i] * w
            normalization[offset + i] += w
        }
        nextIndex += 1
        let finalUpTo = nextIndex < plan.windows.count ? plan.windows[nextIndex].start : plan.total
        let count = finalUpTo - base
        guard count > 0 else { return (base, []) }
        var output = [Float](repeating: 0, count: count)
        for i in 0..<count { output[i] = accumulation[i] / max(normalization[i], 1e-12) }
        let start = base
        accumulation.removeFirst(count)
        normalization.removeFirst(count)
        base = finalUpTo
        return (start, output)
    }

    public var isComplete: Bool { nextIndex == plan.windows.count }
}
