// SPDX-License-Identifier: Apache-2.0
import Foundation
import MLX
import MLXNN

public struct ReuseDiagnosticRow: Encodable, Sendable {
    public let part: String
    public let variant: String
    public let seconds: [Double]
    public let relativeRMSE: Double
    public let maximumError: Double
}

public struct ReuseDiagnostics: Encodable, Sendable {
    public let date: String
    public let arguments: [String]
    public let gpu: String
    public let thermalBefore: Int
    public let thermalAfter: Int
    public let lowPower: Bool
    public let encodedShape: [Int]
    public let rows: [ReuseDiagnosticRow]
}

extension ReuseEngine {
    public func diagnose(_ audio: [Float], progressURL: URL? = nil,
                         variants: [ReuseOptimization]? = nil, kernelsOnly: Bool = false) throws -> ReuseDiagnostics {
        try diagnoseLoaded(audio, progressURL: progressURL, requestedVariants: variants, kernelsOnly: kernelsOnly)
    }
}
