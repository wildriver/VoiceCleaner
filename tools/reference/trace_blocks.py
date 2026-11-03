# SPDX-License-Identifier: Apache-2.0
"""Max |activation| per RE-USE block with the official FP32 PyTorch model.
usage: python trace_blocks.py <input.wav> <sample_rate> [blocks]"""
import json, os, sys
import soundfile as sf, torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

here = os.path.dirname(os.path.abspath(__file__))
root = snapshot_download("nvidia/NVIDIA-NemotronLabs-RE-USE", allow_patterns=["*.py", "config.json", "model.safetensors"])
sys.path.insert(0, here); sys.path.insert(0, root)
import mamba_ssm  # stand-in
from models.generator_SEMamba_time_d4 import SEMamba
from models.stfts import mag_phase_stft

cfg = json.load(open(os.path.join(root, "config.json")))
model = SEMamba(cfg).eval()
model.load_state_dict(load_file(os.path.join(root, "model.safetensors")), strict=True)
sr = int(sys.argv[2]); blocks = int(sys.argv[3]) if len(sys.argv) > 3 else 8
def even(v): v = int(round(v)); return v if v % 2 == 0 else v + 1
n_fft = even(320 * sr // 8000); hop = even(40 * sr // 8000)
audio, _ = sf.read(sys.argv[1], dtype="float32")
mag, pha, _ = mag_phase_stft(torch.from_numpy(audio)[None], n_fft, hop, n_fft, "relu_log1p")
mamba_ssm.TRACE = {}
torch.set_num_threads(os.cpu_count() or 4)
with torch.no_grad():
    from einops import rearrange
    m = rearrange(mag, 'b f t -> b t f').unsqueeze(1); p = rearrange(pha, 'b f t -> b t f').unsqueeze(1)
    x = torch.cat((m, p), 1); B, C, T, Fq = x.shape
    x = torch.cat((x, torch.zeros(B, C, T, 2)), -1); x = torch.cat((x, torch.zeros(B, C, 2, Fq + 2)), -2)
    x = model.dense_encoder(x)
    print(f"encoder {tuple(x.shape)} max {x.abs().max():.3f}")
    for i, block in enumerate(model.TSMamba[:blocks]):
        b, c, t, f = x.size()
        y = x.permute(0, 3, 2, 1).contiguous().view(b * f, t, c)
        mamba_ssm.CURRENT[0] = f"b{i}.time"; y = block.time_mamba(y) + y
        y = y.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b * t, f, c)
        mamba_ssm.CURRENT[0] = f"b{i}.freq"; y = block.freq_mamba(y) + y
        x = y.view(b, t, f, c).permute(0, 3, 1, 2)
        print(f"block{i} out max {x.abs().max():.3f} | " + ", ".join(f"{k}={v:.1f}" for k, v in sorted(mamba_ssm.TRACE.items()) if k.startswith(f"b{i}.")))
