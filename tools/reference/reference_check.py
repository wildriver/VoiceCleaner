# SPDX-License-Identifier: Apache-2.0
"""Compare one RE-USE window processed by the official PyTorch code (CPU,
pure-PyTorch Mamba stand-in) with the MLX output of `voicecleaner window`.

usage: python reference_check.py <input.wav> <mlx_output.wav> <reference_output.wav>
The input must be mono float WAV at 48 kHz (or the rate in --sample-rate).
"""
import argparse
import json
import os
import sys

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

parser = argparse.ArgumentParser()
parser.add_argument("input")
parser.add_argument("mlx")
parser.add_argument("reference")
parser.add_argument("--sample-rate", type=int, default=48000)
args = parser.parse_args()

here = os.path.dirname(os.path.abspath(__file__))
root = snapshot_download("nvidia/NVIDIA-NemotronLabs-RE-USE",
                         allow_patterns=["*.py", "config.json", "model.safetensors", "LICENSE"])
sys.path.insert(0, here)   # mamba_ssm stand-in
sys.path.insert(0, root)   # official models/ and utils/
from models.generator_SEMamba_time_d4 import SEMamba  # noqa: E402
from models.stfts import mag_phase_stft, mag_phase_istft  # noqa: E402

cfg = json.load(open(os.path.join(root, "config.json")))
model = SEMamba(cfg).eval()
model.load_state_dict(load_file(os.path.join(root, "model.safetensors")), strict=True)

def make_even(value):
    value = int(round(value))
    return value if value % 2 == 0 else value + 1

sr = args.sample_rate
base = cfg["stft_cfg"]["sampling_rate"]
n_fft = make_even(cfg["stft_cfg"]["n_fft"] * sr // base)
hop = make_even(cfg["stft_cfg"]["hop_size"] * sr // base)
win = make_even(cfg["stft_cfg"]["win_size"] * sr // base)

audio, file_sr = sf.read(args.input, dtype="float32")
assert file_sr == sr and audio.ndim == 1, "input must be mono at the given rate"
# AVAudioFile may report a slightly shorter length for ffmpeg-written WAVs;
# process exactly the samples the MLX side processed.
mlx_length = sf.info(args.mlx).frames
if mlx_length < audio.shape[0]:
    print(f"note: trimming input {audio.shape[0]} -> {mlx_length} samples to match the MLX input", file=sys.stderr)
    audio = audio[:mlx_length]
noisy = torch.from_numpy(audio).unsqueeze(0)
torch.set_num_threads(os.cpu_count() or 4)
with torch.no_grad():
    mag, pha, _ = mag_phase_stft(noisy, n_fft=n_fft, hop_size=hop, win_size=win,
                                 compress_factor="relu_log1p", center=True, addeps=False)
    amp_g, pha_g, _ = model(mag, pha)
    m = torch.expm1(torch.relu(amp_g))
    zero_portion = torch.sum(m == 0, 1) / m.shape[1]
    amp_g[:, :, (zero_portion > 0.5)[0]] = 0
    out = mag_phase_istft(amp_g, pha_g, n_fft, hop, win, "relu_log1p")[0].numpy()
if out.shape[0] < audio.shape[0]:
    out = np.pad(out, (0, audio.shape[0] - out.shape[0]), constant_values=1e-8)
out = out[: audio.shape[0]]
sf.write(args.reference, out, sr, subtype="FLOAT")

mlx, mlx_sr = sf.read(args.mlx, dtype="float32")
assert mlx_sr == sr and mlx.shape == out.shape, (mlx.shape, out.shape)

def snr(reference, estimate):
    return float(10 * np.log10(np.sum(reference ** 2) / max(np.sum((reference - estimate) ** 2), 1e-20)))

report = {
    "sample_rate": sr, "n_fft": n_fft, "hop": hop, "samples": int(audio.shape[0]),
    "snr_mlx_vs_reference_db": snr(out, mlx),
    "snr_input_vs_reference_db": snr(out, audio),
    "correlation": float(np.corrcoef(out, mlx)[0, 1]),
}
print(json.dumps(report, indent=2))
if report["snr_mlx_vs_reference_db"] < 15:
    print("MLX output deviates from the official reference", file=sys.stderr)
    sys.exit(1)
