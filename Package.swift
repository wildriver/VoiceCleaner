// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "REUSECleaner",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "CleanerCore", targets: ["CleanerCore"]),
        .executable(name: "reuse-clean", targets: ["reuse-clean"]),
    ],
    dependencies: [.package(url: "https://github.com/ml-explore/mlx-swift", exact: "0.33.0")],
    targets: [
        // Vendored from Corvelis/RE-USE-MLX-CoreML v0.1.0 (Apache-2.0), modified.
        .target(name: "ReuseKit", dependencies: [
            .product(name: "MLX", package: "mlx-swift"),
            .product(name: "MLXNN", package: "mlx-swift"),
            .product(name: "MLXFast", package: "mlx-swift"),
            .product(name: "MLXFFT", package: "mlx-swift"),
        ]),
        .target(name: "ReuseModelAssets"),
        .target(name: "CleanerCore", dependencies: ["ReuseKit", "ReuseModelAssets"],
                resources: [.copy("Resources/ModelCatalog.json"), .copy("Resources/ModelSource.json")]),
        .executableTarget(name: "reuse-clean", dependencies: ["CleanerCore"]),
        .testTarget(name: "CleanerCoreTests", dependencies: ["CleanerCore"]),
    ],
    swiftLanguageModes: [.v5]
)
