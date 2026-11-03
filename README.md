<p align="center">
  <img src="docs/assets/banner.svg" alt="VoiceCleaner — 動画の声を、ワンドロップでクリアに。" width="100%">
</p>

<p align="center">
  <b>日本語</b> · <a href="README.en.md">English</a>
</p>

<p align="center">
  <img alt="macOS 15+" src="https://img.shields.io/badge/macOS-15%2B-111827?logo=apple&logoColor=white">
  <img alt="Apple Silicon" src="https://img.shields.io/badge/Apple%20Silicon-M1%E2%80%93M4-4F46E5">
  <img alt="48 kHz Stereo" src="https://img.shields.io/badge/audio-48%20kHz%20stereo-0E7490">
  <img alt="On-device" src="https://img.shields.io/badge/processing-on--device-059669">
  <img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-6B7280">
  <a href="../../actions/workflows/build.yml"><img alt="Build" src="../../actions/workflows/build.yml/badge.svg"></a>
</p>

<p align="center">
  <b>VoiceCleaner</b> は、動画や音声ファイルをドロップするだけで、声の背景ノイズを取り除く macOS アプリです。<br>
  NVIDIA の音声強調モデル <a href="https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE">RE-USE</a> を Mac の GPU で動かし、映像には一切手を加えません。
</p>

---

## ✨ 特長

<table>
<tr>
<td width="50%" valign="top">

### 🎬 映像はそのまま
映像は再エンコードせずにコピーし、音声だけを差し替えます。画質は元のファイルと同じです。

### 🎧 48 kHz・ステレオ
48 kHz で処理し、ステレオは左右を別々に処理してステレオのまま書き出します。

</td>
<td width="50%" valign="top">

### 🔒 すべて Mac の中で
ファイルを外部に送信しません。通信は初回のモデル取得（約54 MB）だけです。

### 🧲 ドラッグ＆ドロップ
ファイルをウィンドウに落とすだけ。複数ファイルは順番に処理し、残り時間も表示します。

</td>
</tr>
</table>

## 🚀 使い方

1. [**Releases**](../../releases/latest) から `VoiceCleaner.dmg` をダウンロードします。
2. `VoiceCleaner.dmg` を開き、アプリを「アプリケーション」フォルダへドラッグします。
3. VoiceCleaner を起動します。初回だけ、ノイズ除去モデルを自動でダウンロードします。
4. 動画（MP4・MOV・M4V）や音声（M4A・WAV など）をウィンドウにドロップします。
5. 完了すると、元のファイルと同じフォルダに **`元の名前_clean.mp4`** ができます。「Finderで表示」ですぐ開けます。

## 🛠 しくみ

```mermaid
flowchart LR
    A["🎞️ 動画 / 音声"] --> B["音声を取り出す<br/>48 kHz・32bit float"]
    A --> V["映像トラック<br/>（再エンコードなし）"]
    B --> C["RE-USE で<br/>ノイズ除去<br/>5秒窓・50%重なり"]
    C --> D["AAC で<br/>書き出し"]
    V --> E["🎞️ _clean.mp4"]
    D --> E
```

| 項目 | 内容 |
| :-- | :-- |
| モデル | NVIDIA NemotronLabs RE-USE（9.6M パラメータ・全30段）。MLX 版の重みは [AILogDev/RE-USE-MLX-CoreML](https://huggingface.co/AILogDev/RE-USE-MLX-CoreML) から取得 |
| 推論 | [MLX](https://github.com/ml-explore/mlx-swift)（Metal GPU） |
| サンプルレート | 48 kHz（STFT 1920 点・ホップ 240。NVIDIA の推論スクリプトと同じ換算） |
| 処理窓 | 5 秒窓・50% 重なり・Hann 窓で重ね合わせ（NVIDIA `inference_chunk.py` と同じ方式） |
| 精度 | 公式 PyTorch 実装との比較で SNR 55.5 dB（48 kHz・2.5 秒の窓、CI で毎回検証） |
| 出力 | 映像はそのままコピー。音声は AAC（モノラル 128 kbps／ステレオ 256 kbps） |

> [!TIP]
> RE-USE は前後の音声を参照するモデルのため、リアルタイム処理（通話用の仮想マイクなど）には向きません。VoiceCleaner はファイルをまとめて処理する用途に特化しています。

## 💻 動作環境

- Apple Silicon（M1 以降）の Mac、macOS 15 以降
- メモリ 16 GB 以上を推奨（48 kHz・5 秒窓の処理で数 GB を使います）
- 処理時間は、Mac の性能と動画の長さによって変わります

## 📄 ライセンス

- アプリのコード：Apache-2.0（[LICENSE](LICENSE)・[NOTICE](NOTICE)）
- `Sources/ReuseKit`・`Sources/ReuseModelAssets`：[Corvelis/RE-USE-MLX-CoreML](https://github.com/Corvelis/RE-USE-MLX-CoreML) v0.1.0（Apache-2.0）を改変
- モデル：NVIDIA RE-USE、OpenMDW-1.1（[licenses/NVIDIA-RE-USE.txt](licenses/NVIDIA-RE-USE.txt)）。リポジトリには含まず、初回起動時にダウンロードします
- 依存ライブラリ：[licenses/](licenses/) を参照
