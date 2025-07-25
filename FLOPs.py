import torch
import os
import argparse
from fvcore.nn import FlopCountAnalysis
from models.vmambaDense import VSSM  # 정확한 경로에 맞게 조정 필요
from config import get_config
from models import build_model

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
def _numel(sizes):
    n = 1
    for d in sizes:
        n *= d
    return n

'''
# 모델 구성 설정 (예시는 기본 설정)
model = VSSM(
    patch_size=4,
    in_chans=3,
    num_classes=160,
    imgsize=224,
    forward_type="v05_noz",  # 사용 중인 타입으로 설정
    ssm_ratio=1.0,
    ssm_d_state=1,
    ssm_dt_rank="auto",
    patchembed_version="v2",  # 실제 사용한 버전으로 바꾸세요
    downsample_version="v3",  # 실제 사용한 버전으로 바꾸세요
).eval()
'''
def parse_args():
    parser = argparse.ArgumentParser(description='Test Person ReID Model with VMamba')
    parser.add_argument('--model_path', type=str, default="./output_M/7_25_base_market_Cross_layer_merge_learnable_all/vssm1_base_0229s/baseline/vssm1_base_0229s.pth")
    return parser.parse_args()

args = parse_args()
model_path = args.model_path

# Load model
class Args:
    #cfg = './configs/vssm/vmambav0_tiny_224.yaml'
    #cfg = './configs/vssm/vmambav2v_tiny_224.yaml'
    cfg = './configs/vssm/vmambav2v_base_224.yaml'
    opts = None

args_cfg = Args()
config = get_config(args_cfg)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = build_model(config).to(device)
ckpt = torch.load(model_path, map_location=device)
state_dict = ckpt['model'] if 'model' in ckpt else ckpt
model.load_state_dict(state_dict, strict=False)
model.eval()

# 입력 텐서 설정 (배치 크기 1, 채널 3, 224x224)
input_tensor = torch.randn(1, 3, 224, 224)

# FLOPs 분석 전에 모델과 입력 텐서를 모두 GPU로 올려야 합니다
model = model.to("cuda")  # 모델 GPU로 이동
input_tensor = input_tensor.to("cuda")  # 입력도 GPU로 이동


# FLOPs 분석
flops = FlopCountAnalysis(model, input_tensor)

 # ———— 여기에 핸들러 등록 ————
 # 1) 각 연산별 근사 FLOP 계산 함수 정의
def flop_gelu(inputs, outputs):
    # inputs[0]는 torch._C.Value → 실제 shape 정보는 .type().sizes()에 담겨 있습니다.
    shape = inputs[0].type().sizes()
    numel = 1
    for s in shape:
        numel *= s
    # GELU 한 번당 대략 4 FLOPs 근사
    return numel * 4

def flop_silu(inputs, outputs):
    shape = inputs[0].type().sizes()
    numel = 1
    for s in shape:
        numel *= s
    # SiLU(x)=x*sigmoid(x) 근사로 3 FLOPs
    return numel * 3

# 2) 누락된 연산 핸들러 등록
flops.set_op_handle("aten::gelu", flop_gelu)
flops.set_op_handle("aten::silu", flop_silu)
flops.set_op_handle("aten::mul", lambda ins, outs: int(_numel(ins[0].type().sizes())))
flops.set_op_handle("aten::add", lambda ins, outs: int(_numel(ins[0].type().sizes())))
flops.set_op_handle("aten::exp", lambda ins, outs: int(_numel(ins[0].type().sizes())))

# 추가 등록해야 할 핸들러들
flops.set_op_handle("aten::neg",
    lambda ins, outs: _numel(ins[0].type().sizes()))

flops.set_op_handle("aten::div",
    lambda ins, outs: _numel(ins[0].type().sizes()))

flops.set_op_handle("aten::sum",
    lambda ins, outs: _numel(ins[0].type().sizes()))

flops.set_op_handle("aten::sub",
    lambda ins, outs: _numel(ins[0].type().sizes()))

flops.set_op_handle("aten::pad",
    lambda ins, outs: 0)  # 패딩은 실제 연산량이 미미하니 0으로 처리해도 무방


def flop_cross_op(ins, outs):
    # ins[0]이 (B, N, D) 토큰 스캔 입력이라고 가정
    sizes = ins[0].type().sizes()
    B, N, D = sizes[0], sizes[1], sizes[2]
    return B * N * D * 10  # 한 토큰당 10 FLOPs로 근사

for op in ["prim::PythonOp.CrossScanTritonF",
           "prim::PythonOp.SelectiveScanCuda",
           "prim::PythonOp.CrossMergeTritonF"]:
    flops.set_op_handle(op, flop_cross_op)

flops.set_op_handle(
    "aten::linalg_vector_norm",
    lambda ins, outs: _numel(ins[0].type().sizes()) * 2
)
# 필요에 따라 aten::neg, aten::div, aten::sum 등도 추가 등록 가능
# ————————————————————————
# ==============================
# Unsupported 연산 핸들러 등록
# ==============================

# 비교 연산 (lt) → FLOPs 1로 근사
flops.set_op_handle(
    "aten::lt", lambda ins, outs: _numel(ins[0].type().sizes())
)

# topk 연산 → 정렬 연산 복잡도 O(N log K)로 근사
def flop_topk(ins, outs):
    sizes = ins[0].type().sizes()
    N = sizes[-1] if len(sizes) > 0 else 1
    # K 값 추출 (PyTorch 버전에 따라 meta 또는 value에서 가져옴)
    try:
        K = ins[1].meta["val"].item()
    except Exception:
        K = 1  # 안전하게 기본값 처리
    # 근사 FLOPs: N log K + N
    return int(N * (K + (K * (K.bit_length() if K > 0 else 1))))

flops.set_op_handle("aten::topk", flop_topk)

# clone 연산 → 단순 메모리 복사로 간주 (FLOPs 0)
flops.set_op_handle(
    "aten::clone", lambda ins, outs: 0
)

# scatter_ 연산 → 인덱스마다 1 FLOP으로 근사
def flop_scatter(ins, outs):
    sizes = ins[0].type().sizes()
    return int(_numel(sizes))
flops.set_op_handle("aten::scatter_", flop_scatter)



 # 최종 FLOPs & 파라미터 출력
print(f"총 FLOPs: {flops.total() / 1e9:.2f} GFLOPs")
print(f"총 파라미터 수: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
# 어떤 연산이 아직 지원되지 않는지도 확인
#print("Unsupported ops:", flops.unsupported_ops())