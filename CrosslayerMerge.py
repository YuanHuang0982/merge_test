import torch, torch.nn as nn, math
from typing import Callable, Tuple
#from learnable import LearnableBipartiteMerge, do_nothing
from learnMerge import LearnableThresholdSoftMatching, do_nothing

class CrossLayerMerge(LearnableThresholdSoftMatching):
    """
    이전 레이어 metric과 현재 metric을 α로 가중합해
    fused_metric으로 merge 판단 → learnable threshold 적용.
    """
    def __init__(self, layer_idx, total_layers, final_threshold):
        super().__init__(layer_idx, total_layers, final_threshold)
        # α: 이전 vs 현재 중요도 비율 (0~1)
        self.alpha = nn.Parameter(torch.tensor(0.5))

        # 이전 metric 저장용 (첫 레이어는 None)
        self.prev_metric = None

    def forward(self,
                metric: torch.Tensor,
                class_token: bool = False,
                distill_token: bool = False
    ) -> Tuple[Callable, Callable]:
        # 1) 이전 metric이 있으면 가중합, 없으면 그대로
        if self.prev_metric is None or self.prev_metric.shape[0] != metric.shape[0]:
            fused = metric
        else:
            fused = self.alpha * self.prev_metric + (1 - self.alpha) * metric
        # 2) 이번 레이어에서도 fused metric 저장 (detach!)
        self.prev_metric = fused.detach()

        # 3) base + delta threshold 계산은 부모 클래스에 맡기기
        return super().forward(fused, class_token, distill_token)
