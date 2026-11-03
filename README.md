# RE-USE Cleaner

動画や音声ファイルを入れると、音声のノイズを除去したファイルを書き出す macOS アプリです。ノイズ除去には NVIDIA の音声強調モデル [RE-USE](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE) を使い、処理はすべて Mac の中で行います。

- MP4・MOV・M4V・M4A・WAV などに対応
- 映像は再エンコードせずにそのままコピーし、音声だけを差し替え（AAC）
- 48 kHz で処理し、ステレオはステレオのまま出力（左右を別々に処理）
- 長い動画も、メモリ使用量を抑えて処理

## 使い方

1. [Actions](../../actions/workflows/build.yml) の最新の成功したビルドの「Artifacts」から `RE-USE-Cleaner` をダウンロードして展開し（GitHub へのログインが必要）、`RE-USE-Cleaner.dmg` を開いて、アプリを「アプリケーション」フォルダへドラッグします。
2. アプリを起動します。初回だけ、ノイズ除去モデル（約54 MB）を自動でダウンロードします。
3. 動画または音声ファイルをウィンドウにドロップします（「ファイルを選択…」からも選べます）。
4. 処理が終わると、元のファイルと同じフォルダに `元の名前_clean.mp4` が作られます。「Finderで表示」で開けます。

元のフォルダに書き込めない場合は、「ムービー」フォルダ内の `RE-USE Cleaner` に保存します。

## 動作環境

- Apple Silicon（M1 以降）の Mac
- macOS 15 以降
- メモリ 16 GB 以上を推奨（48 kHz・5 秒の処理窓で数 GB を使います）

## 処理の内容

| 項目 | 内容 |
| --- | --- |
| モデル | NVIDIA NemotronLabs RE-USE（9.6M パラメータ、全30段）。MLX 版の重みは [AILogDev/RE-USE-MLX-CoreML](https://huggingface.co/AILogDev/RE-USE-MLX-CoreML) から取得 |
| 推論 | MLX（Metal GPU） |
| サンプルレート | 48 kHz（STFT 1920 点・ホップ 240。NVIDIA の推論スクリプトと同じ換算） |
| 処理窓 | 5 秒窓・50% 重なり・Hann 窓による重ね合わせ（NVIDIA の `inference_chunk.py` と同じ） |
| チャンネル | モノラル入力はモノラル、ステレオ以上はステレオ（左右を個別に処理）。設定でモノラル出力も選択可 |
| 出力 | 映像はそのままコピー。音声は AAC（モノラル 128 kbps、ステレオ 256 kbps） |

RE-USE は前後の音声を参照するモデル（非因果的）のため、リアルタイム処理には向きません。このアプリはファイルをまとめて処理する用途に限定しています。

## 開発者向け

ビルドは GitHub Actions（`.github/workflows/build.yml`）で行います。手元でビルドする場合は Xcode 26.4 以降と [XcodeGen](https://github.com/yonaskolb/XcodeGen) が必要です。

```bash
xcodegen generate
```

```bash
open REUSECleaner.xcodeproj
```

コマンドライン版 `reuse-clean`（`swift package` のプロダクト）は、CI での動作確認に使います。

### Developer ID 署名と公証

リポジトリの Secrets に次の値を設定すると、CI が Developer ID で署名して公証した dmg を作ります。未設定の場合はアドホック署名になり、初回起動時に「システム設定 → プライバシーとセキュリティ」で許可が必要です。

| Secret | 内容 |
| --- | --- |
| `MACOS_CERTIFICATE_P12` | 「Developer ID Application」証明書（.p12）の Base64 |
| `MACOS_CERTIFICATE_PASSWORD` | .p12 のパスワード |
| `APPLE_API_KEY_P8` | App Store Connect API キー（.p8）の中身 |
| `APPLE_API_KEY_ID` | API キーの ID |
| `APPLE_API_ISSUER_ID` | Issuer ID |

## ライセンス

- アプリのコード：Apache-2.0（[LICENSE](LICENSE)、[NOTICE](NOTICE)）
- `Sources/ReuseKit`・`Sources/ReuseModelAssets`：[Corvelis/RE-USE-MLX-CoreML](https://github.com/Corvelis/RE-USE-MLX-CoreML) v0.1.0（Apache-2.0）を改変
- モデル：NVIDIA RE-USE、OpenMDW-1.1（[licenses/NVIDIA-RE-USE.txt](licenses/NVIDIA-RE-USE.txt)）。リポジトリには含まず、初回起動時にダウンロード
- 依存ライブラリ：[licenses/](licenses/) を参照
