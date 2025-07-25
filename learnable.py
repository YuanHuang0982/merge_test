import math
from typing import Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def do_nothing(x, mode=None):
    return x

class LearnableBipartiteMerge(nn.Module):
    def __init__(self, layer_idx, total_layers, final_threshold):
        super().__init__()
        base = final_threshold * layer_idx / total_layers
        self.register_buffer("base_threshold", torch.tensor(base))
        self.delta = nn.Parameter(torch.zeros_like(self.base_threshold))
        # 이 모듈이 channel‐first 구조를 안다면 여기에 flag를 넣어두세요.
        self.channel_first = True  

    def forward(self, metric: torch.Tensor, class_token=False, distill_token=False):
        B, T, C = metric.shape
        # 1) normalize & split
        m = metric / metric.norm(dim=-1, keepdim=True)  # (B,T,C)
        a = m[:, ::2, :]   # (B,P,C)
        b = m[:, 1::2, :]  # (B,P,C)
        P = a.size(1)

        # 2) dot → mask
        scores = (a * b).sum(dim=-1)                     # (B,P)
        threshold = (self.base_threshold + self.delta).clamp(min=0.0)
        mask = scores >= threshold.view(1,1)             # (B,P)
        if class_token:  mask[:,0] = False
        if distill_token: mask[:,1] = False

        # 3) 아무 것도 merge 대상 없으면 no‐op
        if not mask.any():
            return do_nothing, do_nothing

        # 클로저에 캡처할 변수들
        mask_cl = mask
        P_cl = P
        channel_first = self.channel_first

        def merge(x: torch.Tensor, mode="sum") -> torch.Tensor:
            # x: (B,T,C)
            a_x = x[:, ::2, :]       # (B,P,C)
            b_x = x[:, 1::2, :]      # (B,P,C)
            mask_e = mask_cl.unsqueeze(-1).to(a_x.dtype)  # (B,P,1)
            src_x = a_x * mask_e     # (B,P,C)
            # **scatter 대신** 덧셈 한 줄로 병합
            out_x = b_x + src_x      # (B,P,C)
            return out_x

        def unmerge(x: torch.Tensor) -> torch.Tensor:
            # x: (B,P,C) → spatial reshape → (B,C,H',W') or (B,H',W',C)
            B_, P_, C_ = x.shape
            H = int(math.sqrt(P_)); W = P_ // H
            x_sp = x.reshape(B_, H, W, C_)  # (B,H,W,C)
            if channel_first:
                return x_sp.permute(0,3,1,2).contiguous()  # (B,C,H,W)
            else:
                return x_sp

        return merge, unmerge

