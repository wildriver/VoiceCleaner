# SPDX-License-Identifier: Apache-2.0
# Pure-PyTorch stand-in for mamba_ssm.Mamba (v1 semantics, sequential scan) so the
# official RE-USE code can run on CPU for reference comparisons. Slow; tests only.
import math, torch, torch.nn as nn, torch.nn.functional as F
class Mamba(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_inner = expand*d_model; self.d_state=d_state
        self.dt_rank = math.ceil(d_model/16)
        self.in_proj = nn.Linear(d_model, 2*self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv, groups=self.d_inner, padding=d_conv-1)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank+2*d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner)
        self.A_log = nn.Parameter(torch.log(torch.arange(1,d_state+1).float().repeat(self.d_inner,1)))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
    def forward(self, x):
        B,L,_ = x.shape
        xz = self.in_proj(x); x_, z = xz.chunk(2, -1)
        x_ = F.silu(self.conv1d(x_.transpose(1,2))[..., :L]).transpose(1,2)
        dbc = self.x_proj(x_); dt, Bm, Cm = dbc.split([self.dt_rank, self.d_state, self.d_state], -1)
        dt = F.softplus(self.dt_proj(dt))               # B L D
        A = -torch.exp(self.A_log)                       # D N
        dA = torch.exp(dt.unsqueeze(-1)*A)               # B L D N
        dBx = dt.unsqueeze(-1)*Bm.unsqueeze(2)*x_.unsqueeze(-1)
        h = torch.zeros(B, self.d_inner, self.d_state); ys=[]
        for t in range(L):
            h = dA[:,t]*h + dBx[:,t]; ys.append((h*Cm[:,t].unsqueeze(1)).sum(-1))
        y = torch.stack(ys,1) + x_*self.D
        return self.out_proj(y*F.silu(z))
