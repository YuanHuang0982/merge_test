#!/usr/bin/env python
# heatm.py
# -*- coding: utf-8 -*-

import os
import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
from captum.attr import LayerIntegratedGradients

from config import get_config
from models import build_model


# --------------------- CLI ---------------------
def get_args():
    p = argparse.ArgumentParser(
        description="VMamba Token Merge Attribution + SS2D Activation Visualization"
    )
    p.add_argument("--cfg", default="./configs/vssm/vmambav2v_base_224.yaml",
                   help="YAML config path ")
    p.add_argument("--opts", nargs="*", default=[],
                   help="Override config options, e.g. MODEL.TOKEN_MERGE True")
    p.add_argument("--checkpoint", default="./output_m/merge_base_market/vssm1_base_0229s/baseline/vssm1_base_0229s.pth",
                   help="Model checkpoint (.pth) path")
    p.add_argument("--image", default="/home/user/Project/Test2/VMamba_Person/dataset/market_1501/test/query/0001/0001_c1s1_001051_00.jpg",
                   help="Input image path")
    p.add_argument("--out-dir", default="./vis_out/test9",
                   help="Output directory")
    p.add_argument("--fusion-layer", type=int, default=0,
                   help="Flattened block index where merge occurred")
    p.add_argument("--ig-steps", type=int, default=50,
                   help="IG integration steps")
    p.add_argument("--device", choices=["cpu","cuda"], default="cuda",
                   help="Compute device")
    return p.parse_args()

# ImageNet 표준화/역변환
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
invTrans = transforms.Compose([
    transforms.Normalize([0,0,0], [1/s for s in IMAGENET_STD]),
    transforms.Normalize([-m for m in IMAGENET_MEAN], [1,1,1]),
])

def show_cam_on_image(img, mask):
    """img: H×W×3 float[0,1], mask: H×W float[0,1]
    heat = cv2.applyColorMap((mask*255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = heat.astype(np.float32)/255.0
    cam  = heat + img
    #cam  = cam / cam.max()
    cam = cam / np.percentile(cam,99)
    return (cam*255).astype(np.uint8)"""
    # 1) 컬러맵 적용 → uint8 [0~255]
    heat = cv2.applyColorMap((mask*255).astype(np.uint8), cv2.COLORMAP_JET)

    # 2) 원본 img (float[0,1]) → uint8 [0~255]
    img_uint8 = (img * 255).astype(np.uint8)

    # 3) alpha-blend (두 입력 모두 uint8)
    overlay = cv2.addWeighted(img_uint8, 0.6, heat, 0.4, 0)
    return overlay


def find_block(model, fusion_layer):
    idx = 0
    for layer in model.layers:
        blks = getattr(layer, "blocks", layer)
        for blk in blks:
            if idx == fusion_layer:
                return blk
            idx += 1
    raise ValueError(f"fusion-layer={fusion_layer} not found")

def register_hooks(model, block):
    holder = {}

    # forward 후, block 내부에 저장된 병합 결과 가져오기
    def hook_after_forward(_, __, ___):
        # block.forward가 실행되면 아래 값들이 채워짐
        holder['orig_indices'] = getattr(block, 'orig_indices', [])
        holder['dst_indices'] = getattr(block, 'merged_indices', [])
        holder['x_merged'] = getattr(block, 'x_merged', None)

    # block의 forward 끝난 뒤 호출되도록 hook 추가
    block.register_forward_hook(hook_after_forward)

    # head token도 그대로 추적 (기존 코드 유지)
    model.head.register_forward_pre_hook(
        lambda mod, inp: holder.__setitem__('feat_tok', inp[0])
    )
    return holder

import random   # 파일 최상단에 추가

def draw_merge_arrows_with_outlines(img_bgr, src_coords, dst_coords, init_grid, max_arrows=20):
    h, w = img_bgr.shape[:2]
    ph, pw = h//init_grid, w//init_grid
    out = img_bgr.copy()

    # 1) (옵션) 전체 그리드 라인 그리기
    grid_color = (200,200,200)  # 연한 회색
    for i in range(init_grid+1):
        # 수평선
        y = i * ph
        cv2.line(out, (0,y), (w,y), grid_color, 1)
        # 수직선
        x = i * pw
        cv2.line(out, (x,0), (x,h), grid_color, 1)

    # 2) src_coords, dst_coords 쌍 샘플링
    pairs = list(zip(src_coords, dst_coords))
    if len(pairs) > max_arrows:
        import random
        pairs = random.sample(pairs, max_arrows)

    # 3) src/dst 아웃라인 그리기
    for (r_s,c_s), (r_d,c_d) in pairs:
        # 자기 자신 병합 skip
        if (r_s,c_s) == (r_d,c_d):
            continue
        # src 패치 빨간 테두리
        x0_s, y0_s = c_s*pw, r_s*ph
        x1_s, y1_s = x0_s+pw, y0_s+ph
        cv2.rectangle(out, (x0_s,y0_s), (x1_s,y1_s), (0,0,255), 2)
        # dst 패치 초록 테두리
        x0_d, y0_d = c_d*pw, r_d*ph
        x1_d, y1_d = x0_d+pw, y0_d+ph
        cv2.rectangle(out, (x0_d,y0_d), (x1_d,y1_d), (0,255,0), 2)

    # 4) 화살표 그리기 (랜덤 컬러 버전)
    total = len(pairs)
    for idx, ((r_s,c_s), (r_d,c_d)) in enumerate(pairs):
        # 자기 자신 병합 skip
        if (r_s,c_s) == (r_d,c_d):
            continue
        # Hue 기반 컬러
        hue = int(179 * idx / total)
        hsv_pixel = np.uint8([[[hue,255,200]]])
        bgr = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0,0].tolist()

        y0 = r_s*ph + ph//2
        x0 = c_s*pw + pw//2
        y1 = r_d*ph + ph//2
        x1 = c_d*pw + pw//2
        cv2.arrowedLine(out, (x0,y0), (x1,y1), color=bgr,
                        thickness=1, tipLength=0.3)

    return out


def get_merged_patch_scores_and_mask(model, x, fusion_layer, ig_steps, config):
    device = x.device

    # 1) config 로부터 depths, patch_size, img_size
    depths     = config.MODEL.VSSM.DEPTHS
    patch_size = config.MODEL.VSSM.PATCH_SIZE
    img_size   = config.DATA.IMG_SIZE

    # 2) init grid
    init_grid = img_size // patch_size      # e.g. 56
    total0    = init_grid * init_grid       # e.g. 56*56=3136

    # 3) fusion_layer → stage 계산
    cum = 0
    for stage, d in enumerate(depths):
        if fusion_layer < cum + d:
            fusion_stage = stage
            break
        cum += d
    else:
        raise ValueError(f"fusion-layer {fusion_layer} out of range")

    # 4) 이 stage 에서의 grid 크기
    gh = gw = init_grid // (2 ** fusion_stage)
    print(f"[debug] fusion-layer={fusion_layer} → stage={fusion_stage}, grid={gh}×{gw}")

    # 5) hook 등록
    block  = find_block(model, fusion_layer)
    holder = register_hooks(model, block)

    # 6) attribution 함수 정의
    def fwd(_x):
        _ = model(_x)
        tok   = holder['feat_tok']
        preds = model.head(tok)
        tgt   = preds.argmax(dim=-1)
        return preds.gather(1, tgt[:,None]).squeeze(-1)

    lig  = LayerIntegratedGradients(fwd, model.patch_embed)
    attr = lig.attribute(
        inputs    = x,
        baselines = torch.zeros_like(x),
        n_steps   = ig_steps
    )

    # 7) scr 계산 — 항상 init_grid×init_grid 로 만들기
    #    patch_embed 의 출력이 3D(B, N, C)이면 dim==3, 4D(B,C,H,W)이면 dim==4
    if attr.dim() == 3:
        # (B, N, C) → 채널합 → (B, N)
        scr = attr.sum(dim=2).clamp(min=0)
    else:
        # (B, C, H, W) → 채널합 → (B, H, W)
        scr = attr.sum(dim=1).clamp(min=0)

    # 정규화
    scr = (scr - scr.min())/(scr.max()-scr.min()+1e-6)

    # numpy flat vector (길이 = init_grid²)
    scr_np = scr.squeeze(0).cpu().numpy().reshape(init_grid*init_grid)

    # 8) 최초 그리드(56×56) → 실제 stage 그리드(gh×gw) 로 다운샘플 (average pooling)
    scr_grid = scr_np.reshape(init_grid, init_grid)
    if fusion_stage > 0:
        factor = 2 ** fusion_stage
        # (gh, factor, gw, factor) 으로 보고 블록별 평균
        heatmap = scr_grid.reshape(
            gh, factor,
            gw, factor
        ).mean(axis=(1,3))
    else:
        heatmap = scr_grid  # (init_grid, init_grid)

    # 9) mask_map 생성 (hook 으로 저장된 dst_indices)
    mask_map = np.zeros((gh, gw), np.uint8)
    for coord in holder.get('dst_indices', []):
        if isinstance(coord, (list, tuple)) and len(coord)==2:
            r, c = coord
        else:
            idx = int(coord)
            r, c = divmod(idx, gw)
            # 위에서 다운샘플할 거니 여기서는 init_grid 기준
            #r //= factor; c //= factor
        if 0 <= r < gh and 0 <= c < gw:
            mask_map[r, c] = 1

    # 10) outline 용 그룹
    merged_groups = []
    for coord in holder.get('orig_indices', []):
        if isinstance(coord, (list, tuple)) and len(coord)==2:
            merged_groups.append([coord])
        else:
            idx = int(coord)
            r, c = divmod(idx, init_grid)
            r //= factor; c //= factor
            merged_groups.append([(r, c)])

    return heatmap, mask_map, merged_groups, gh, gw, scr_np, holder.get('dst_indices', []), holder.get('orig_indices', [])






if __name__ == "__main__":
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 모델 & config 로드
    CFG    = type("", (), {"cfg":args.cfg, "opts":args.opts})
    config = get_config(CFG)
    model  = build_model(config).to(args.device)
    ck     = torch.load(args.checkpoint, map_location="cpu")
    sd     = ck.get("model", ck)
    model.load_state_dict(sd, strict=False)
    model.eval()

    # 이미지 전처리
    input_size = config.DATA.IMG_SIZE
    interp     = getattr(transforms.InterpolationMode, config.DATA.INTERPOLATION.upper())
    tfm = transforms.Compose([
        transforms.Resize((input_size, input_size), interpolation=interp),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    img    = Image.open(args.image).convert("RGB")
    tensor = tfm(img).unsqueeze(0).to(args.device)
    img_np = invTrans(tensor.squeeze().cpu()).permute(1,2,0).numpy()

    init_grid = config.DATA.IMG_SIZE // config.MODEL.VSSM.PATCH_SIZE

    # 0~5 fusion-layer 순회
    for fl in range(25):
        print(f"\n>>> fusion-layer = {fl}")
        outd = Path(args.out_dir)/f"fl_{fl}"
        outd.mkdir(exist_ok=True, parents=True)

        heat_stage, mask_stage, _, gh, gw, scr_flat, dst_coords, src_coords = \
            get_merged_patch_scores_and_mask(
                model, tensor,
                fusion_layer=fl, ig_steps=args.ig_steps, config=config
            )

        # -- (1) Heatmap overlay
        h_up = F.interpolate(
            torch.tensor(heat_stage[None,None]),
            size=(input_size,input_size),
            mode="nearest"#, align_corners=False
        ).squeeze().numpy()
        vis = show_cam_on_image(img_np, 1.0 - h_up)
        cv2.imwrite(str(outd/f"{Path(args.image).stem}_heat_fl{fl}.png"),
                    vis[...,::-1])

        # -- (2) Mask overlay
        m_up = cv2.resize(mask_stage,
                        (input_size,input_size),
                        interpolation=cv2.INTER_NEAREST)
        overlay = img_np.copy()
        overlay[m_up==1] = [0,1,0]
        cv2.imwrite(str(outd/f"{Path(args.image).stem}_mask_fl{fl}.png"),
                    (overlay*255).astype(np.uint8)[...,::-1])

        # -- (2') Removed-token overlay (red)
        # src_coords 에서 orig_indices → src_map 생성
        src_map = np.zeros((gh, gw), np.uint8)
        for (r, c) in src_coords:
            if 0 <= r < gh and 0 <= c < gw:
                src_map[r, c] = 1
        # 업샘플해서 원본 크기와 맞추기
        src_up = cv2.resize(
            src_map,
            (input_size, input_size),
            interpolation=cv2.INTER_NEAREST
        )
        red_overlay = img_np.copy()
        red_overlay[src_up == 1] = [1, 0, 0]
        cv2.imwrite(
            str(outd / f"{Path(args.image).stem}_removed_fl{fl}.png"),
            (red_overlay * 255).astype(np.uint8)[..., ::-1]
        )



        # -- (3) Outline
        img_bgr = cv2.imread(str(args.image))
        img_bgr = cv2.resize(img_bgr,(input_size,input_size))
        outline = img_bgr.copy()
        ph, pw = input_size//gh, input_size//gw
        for (r,c) in dst_coords:
            y0, x0 = r*ph, c*pw
            y1, x1 = (r+1)*ph, (c+1)*pw
            cv2.rectangle(outline,(x0,y0),(x1,y1),(0,255,0),2)
        cv2.imwrite(str(outd/f"{Path(args.image).stem}_outline_fl{fl}.png"),
                    outline)

        # main() 안, 각 fusion-layer 처리 직후에

        # → (4) 실제 stage 크기(gh×gw) 마스크 저장
        # mask_stage는 get_merged_patch_scores_and_mask에서 반환된 배열입니다.
        txt_stage = outd / f"{Path(args.image).stem}_maskstage_fl{fl}.txt"
        np.savetxt(txt_stage, mask_stage, fmt="%d", delimiter=" ")
        print(f"→ stage 그리드({gh}×{gw}) 저장: {txt_stage}")

        # gh, gw = 현재 stage 그리드 너비 높이
        keep_map = np.ones((gh, gw), np.uint8)   # 기본 모두 1(남음)
        # 제거된 토큰(src_coords) 자리에 0 표시
        for (r,c) in src_coords:
            if 0 <= r < gh and 0 <= c < gw:
                keep_map[r, c] = 0

        # --- 녹색: keep_map==1, 붉은색: keep_map==0 으로 통합 시각화 예시 ---
        overlay = img_np.copy()
        # upsample 해서 전체 해상도에 맞추고
        keep_up = cv2.resize(keep_map, (input_size,input_size),
                            interpolation=cv2.INTER_NEAREST)
        # 붉은색 먼저 덮기
        overlay[keep_up==0] = [1, 0, 0]
        # 녹색 덮기
        overlay[keep_up==1] = [0, 1, 0]
        cv2.imwrite(str(outd/f"{Path(args.image).stem}_merged_all_fl{fl}.png"),
                    (overlay*255).astype(np.uint8)[...,::-1])

        img_bgr = cv2.imread(str(args.image))
        img_bgr = cv2.resize(img_bgr, (input_size,input_size))
        arrow_img = draw_merge_arrows_with_outlines(img_bgr, src_coords, dst_coords, init_grid)
        save_path = Path(args.out_dir)/f"{Path(args.image).stem}_merge_arrows_fl{fl}.png"
        cv2.imwrite(str(save_path), arrow_img)
        print(f"→ 병합 경로 시각화 저장: {save_path}")




    print("\n✅ 모두 완료했습니다!")

    