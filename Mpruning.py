import torch
import torch.nn as nn


class MambaAbsoluteThresholdPruner(nn.Module):
    """
    Layer-wise absolute threshold token pruning for Mamba.
    Tokens with importance < (base + learnable) are pruned.
    """
    def __init__(self,
                 layer_idx: int,
                 total_layers: int,
                 final_token_threshold: float,
                 method: str = "delta"):
        """
        Args:
            layer_idx: index of current block (1-based)
            total_layers: total number of SSM blocks
            final_token_threshold: final pruning strength (0~1)
            method: one of {"l2", "mean", "var", "delta", "abc"}
        """
        super().__init__()
        # base threshold scales linearly by layer index
        base_val = final_token_threshold * layer_idx / total_layers
        # register buffer to ensure device consistency
        self.register_buffer("keep_threshold_base", torch.tensor(base_val, dtype=torch.float32))
        # learnable delta around base
        self.keep_threshold = nn.Parameter(torch.zeros_like(self.keep_threshold_base))
        self.method = method

    @staticmethod
    def compute_importance(x, method="l2", prev_x=None, B_mat=None, C_mat=None):
        """
        Compute token importance scores:
          - "l2": Euclidean norm of features
          - "mean": mean over feature dim
          - "var": variance over feature dim
          - "delta": change from previous features
          - "abc": sum of norms of B_mat and C_mat projections
        """
        if method == "l2":
            return x.norm(p=2, dim=-1)
        elif method == "mean":
            return x.mean(dim=-1)
        elif method == "var":
            return x.var(dim=-1)
        elif method == "delta":
            if prev_x is None:
                raise ValueError("prev_x is required for 'delta' method")
            return (x - prev_x).norm(p=2, dim=-1)
        elif method == "abc":
            if B_mat is None or C_mat is None:
                raise ValueError("B_mat and C_mat required for 'abc' method")
            b_score = B_mat.flatten(2).norm(p=2, dim=-1)
            c_score = C_mat.flatten(2).norm(p=2, dim=-1)
            return b_score + c_score
        else:
            raise ValueError(f"Unknown importance method: {method}")

    def forward(self, x, prev_x=None, B_mat=None, C_mat=None):
        """
        Args:
            x: (B, N, C) current tokens
            prev_x: (B, N, C) previous tokens for delta
            B_mat, C_mat: (B, N, d, L) selective_scan outputs for abc
        Returns:
            out: (B, M, C) pruned tokens (M = max kept per batch)
            info: dict with keys {
                'threshold': float threshold,
                'importance': (B, N) importance scores,
                'mask': (B, N) binary keep mask,
                'indices': list of kept indices per batch
            }
        """
        B, N, C = x.shape
        # 1) compute importance
        imp = self.compute_importance(x, method=self.method,
                                      prev_x=prev_x, B_mat=B_mat, C_mat=C_mat)
        # 2) compute threshold & clamp
        threshold = (self.keep_threshold_base + self.keep_threshold).clamp(min=0.0)
        # 3) create mask: keep if importance >= threshold
        mask = (imp >= threshold).float()
        # 4) ensure at least one token per sample
        for b in range(B):
            if mask[b].sum() < 1:
                topk_idx = imp[b].topk(1)[1]
                mask[b, topk_idx] = 1.0
        # 5) gather kept tokens and record indices
        keep_counts = mask.sum(dim=1).long()        # (B,)
        max_keep = keep_counts.max().item()
        out = torch.zeros(B, max_keep, C, device=x.device, dtype=x.dtype)
        indices = []
        for b in range(B):
            idx = mask[b].nonzero(as_tuple=False).squeeze(1)
            indices.append(idx)
            out[b, :idx.numel()] = x[b, idx]

        info = {
            "threshold": threshold,
            "importance": imp,
            "mask": mask,
            "indices": indices
        }
        return out, info
