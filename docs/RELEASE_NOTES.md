VoiceCleaner: 動画・音声ファイルのノイズを NVIDIA RE-USE で除去する macOS アプリ（Apple Silicon、macOS 15 以降）。

- サンプルレートとチャンネル数を保持（モノラル・ステレオ・5.1ch などを各チャンネル別々に処理）
- WAV・AIFF・CAF・FLAC・ALAC は同じ形式で可逆のまま書き出し、それ以外の音声は AAC
- 映像と他の音声トラックは再エンコードせずそのままコピー
- 初回起動時にモデル（約54 MB）をダウンロード
