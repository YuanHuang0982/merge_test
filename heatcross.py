#!/usr/bin/env python
# heatm_crosslayer.py
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
from CrosslayerMerge import CrossLayerMerge

# --------------------- CLI ---------------------
def get_args():
    p = argparse.ArgumentParser(
        description="VMamba Cross-Layer Merge Attribution + SS2D Activation Visualization"
    )
    p.add_argument("--cfg", default="./configs/vssm/vmambav2v_base_224.yaml",
                   help="YAML config path ")
    p.add_argument("--opts", nargs="*", default=[],
                   help="Override config options, e.g. MODEL.MERGE.TYPE CrossLayerMerge")
    p.add_argument("--checkpoint", default="./output_M/7_14_base_market_Cross_layer_merge_learnable_all/vssm1_base_0229s/baseline/vssm1_base_0229s.pth",
                   help="Model checkpoint (.pth) path")
    p.add_argument("--image", default="./dataset/market_1501/test/query/0001/0001_c1s1_001051_00.jpg",
                   help="Input image path")
    p.add_argument("--out-dir", default="./vis_out/test3",
                   help="Output directory")
    p.add_argument("--fusion-layer", type=int, default=25,
                   help="Index among CrossLayerMerge modules to visualize (will loop all if not used)")
    p.add_argument("--ig-steps", type=int, default=50,
                   help="IG integration steps")
    p.add_argument("--device", choices=["cpu","cuda"], default="cuda",
                   help="Compute device")
    return p.parse_args()


# ImageNet normalize/inverse
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
invTrans = transforms.Compose([
    transforms.Normalize([0,0,0], [1/s for s in IMAGENET_STD]),
    transforms.Normalize([-m for m in IMAGENET_MEAN], [1,1,1]),
])

# show heatmap overlay
def show_cam_on_image(img, mask):
    heat = cv2.applyColorMap((mask*255).astype(np.uint8), cv2.COLORMAP_JET)
    img_uint8 = (img * 255).astype(np.uint8)
    return cv2.addWeighted(img_uint8, 0.6, heat, 0.4, 0)

# draw merge arrows with outlines
def draw_merge_arrows_with_outlines(img_bgr, src_coords, dst_coords, init_grid, max_arrows=20):
    h, w = img_bgr.shape[:2]
    ph, pw = h//init_grid, w//init_grid
    out = img_bgr.copy()
    # grid lines
    for i in range(init_grid+1):
        y = i * ph
        x = i * pw
        cv2.line(out, (0,y), (w,y), (200,200,200), 1)
        cv2.line(out, (x,0), (x,h), (200,200,200), 1)
    pairs = list(zip(src_coords, dst_coords))
    if len(pairs) > max_arrows:
        pairs = __import__('random').sample(pairs, max_arrows)
    # outlines & arrows
    total = len(pairs)
    for idx, ((r_s,c_s),(r_d,c_d)) in enumerate(pairs):
        if (r_s,c_s)==(r_d,c_d): continue
        # outlines
        cv2.rectangle(out, (c_s*pw, r_s*ph), ((c_s+1)*pw,(r_s+1)*ph), (0,0,255), 2)
        cv2.rectangle(out, (c_d*pw, r_d*ph), ((c_d+1)*pw,(r_d+1)*ph), (0,255,0), 2)
        # arrow
        hue = int(179 * idx/(total+1))
        hsv = np.uint8([[[hue,255,200]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0,0].tolist()
        y0,x0 = r_s*ph+ph//2, c_s*pw+pw//2
        y1,x1 = r_d*ph+ph//2, c_d*pw+pw//2
        cv2.arrowedLine(out, (x0,y0),(x1,y1), color=bgr, thickness=1, tipLength=0.3)
    return out

# find CrossLayerMerge by index
def find_merge_module(model, fusion_layer):
    idx = 0
    for m in model.modules():
        if isinstance(m, CrossLayerMerge):
            if idx == fusion_layer:
                return m
            idx += 1
    raise ValueError(f"fusion-layer {fusion_layer} not found")

# register hooks capturing merge info + prev_metric
def register_hooks(model, module):
    holder = {}
    def after_forward(_, __, ___):
        holder['orig_indices'] = getattr(module, 'orig_indices', [])
        holder['dst_indices']  = getattr(module, 'merged_indices', [])
        holder['x_merged']     = getattr(module, 'x_merged', None)
        holder['prev_metric']  = getattr(module, 'prev_metric', None)
    module.register_forward_hook(after_forward)
    model.head.register_forward_pre_hook(lambda mod, inp: holder.__setitem__('feat_tok', inp[0]))
    return holder

# compute attribution + merge mask per fusion layer
def get_merged_patch_scores_and_mask(model, x, fusion_layer, ig_steps, config):
    depths    = config.MODEL.VSSM.DEPTHS
    patch_sz  = config.MODEL.VSSM.PATCH_SIZE
    img_sz    = config.DATA.IMG_SIZE
    init_grid = img_sz // patch_sz
    # determine stage
    cum=0
    for stage,d in enumerate(depths):
        if fusion_layer < cum+d:
            fusion_stage=stage; break
        cum+=d
    else:
        raise ValueError(f"fusion-layer {fusion_layer} out of range")
    gh = gw = init_grid // (2**fusion_stage)
    print(f"[debug] fusion-layer={fusion_layer}→stage={fusion_stage}, grid={gh}×{gw}")
    # hook CrossLayerMerge
    module = find_merge_module(model, fusion_layer)
    module.prev_metric=None
    holder=register_hooks(model,module)
    # attribution
    def fwd(inp):
        _=model(inp)
        tok=holder['feat_tok']; pred=model.head(tok)
        tgt=pred.argmax(dim=-1)
        return pred.gather(1,tgt[:,None]).squeeze(-1)
    lig = LayerIntegratedGradients(fwd, model.patch_embed)
    attr=lig.attribute(inputs=x, baselines=torch.zeros_like(x), n_steps=ig_steps)
    # score map flat
    scr = attr.sum(dim=2).clamp(min=0) if attr.dim()==3 else attr.sum(dim=1).clamp(min=0)
    scr=(scr-scr.min())/(scr.max()-scr.min()+1e-6)
    scr_np=scr.squeeze(0).cpu().numpy().reshape(init_grid*init_grid)
    # downsample
    grid2d = scr_np.reshape(init_grid, init_grid)
    if fusion_stage>0:
        factor=2**fusion_stage
        heat=grid2d.reshape(gh,factor,gw,factor).mean(axis=(1,3))
    else:
        heat=grid2d
    # mask
    mask=np.zeros((gh,gw),np.uint8)
    for coord in holder.get('dst_indices',[]):
        r,c=coord if isinstance(coord,(list,tuple)) else divmod(int(coord),gw)
        if 0<=r<gh and 0<=c<gw: mask[r,c]=1
    # orig groups (not used later but returned)
    groups=[]
    for coord in holder.get('orig_indices',[]):
        if isinstance(coord,(list,tuple)):
            groups.append([coord])
        else:
            r,c=divmod(int(coord),init_grid)
            groups.append([(r//(2**fusion_stage), c//(2**fusion_stage))])
    return heat, mask, groups, gh, gw, scr_np, holder.get('dst_indices',[]), holder.get('orig_indices',[])

if __name__=="__main__":
    args=get_args()
    os.makedirs(args.out_dir,exist_ok=True)
    CFG=type("",(),{"cfg":args.cfg,"opts":args.opts})
    config=get_config(CFG)
    model=build_model(config).to(args.device)
    ck=torch.load(args.checkpoint,map_location="cpu")
    sd=ck.get("model",ck)
    model.load_state_dict(sd,strict=False)
    model.eval()
    # preprocess
    interp=getattr(transforms.InterpolationMode,config.DATA.INTERPOLATION.upper())
    tfm=transforms.Compose([
        transforms.Resize((config.DATA.IMG_SIZE,config.DATA.IMG_SIZE),interpolation=interp),
        transforms.ToTensor(),transforms.Normalize(IMAGENET_MEAN,IMAGENET_STD)
    ])
    img=Image.open(args.image).convert("RGB")
    tensor=tfm(img).unsqueeze(0).to(args.device)
    img_np=invTrans(tensor.squeeze().cpu()).permute(1,2,0).numpy()
    init_grid=config.DATA.IMG_SIZE//config.MODEL.VSSM.PATCH_SIZE
    # determine loops
    merges=[m for m in model.modules() if isinstance(m,CrossLayerMerge)]
    print("[info] num CrossLayerMerge modules =", len(merges))
    layers=range(len(merges)) if args.fusion_layer is None else [args.fusion_layer]
    for fl in layers:
        print(f"\n>>> fusion-layer={fl}")
        outd=Path(args.out_dir)/f"fl_{fl}"; outd.mkdir(exist_ok=True,parents=True)
        heat,mask,_,gh,gw,scr, dst, src = get_merged_patch_scores_and_mask(model,tensor,fl,args.ig_steps,config)
        # heatmap
        h_up=F.interpolate(torch.tensor(heat[None,None]),size=(config.DATA.IMG_SIZE,config.DATA.IMG_SIZE),mode="nearest").squeeze().numpy()
        vis=show_cam_on_image(img_np,1.0-h_up)
        cv2.imwrite(str(outd/f"{Path(args.image).stem}_heat_fl{fl}.png"),vis[...,::-1])
        # mask overlay
        m_up=cv2.resize(mask,(config.DATA.IMG_SIZE,config.DATA.IMG_SIZE),interpolation=cv2.INTER_NEAREST)
        ov=img_np.copy(); ov[m_up==1]=[0,1,0]
        cv2.imwrite(str(outd/f"{Path(args.image).stem}_mask_fl{fl}.png"),(ov*255).astype(np.uint8)[...,::-1])
        # removed tokens
        src_map=np.zeros((gh,gw),np.uint8)
        for r,c in src: 
            if 0<=r<gh and 0<=c<gw: src_map[r,c]=1
        src_up=cv2.resize(src_map,(config.DATA.IMG_SIZE,config.DATA.IMG_SIZE),interpolation=cv2.INTER_NEAREST)
        rd=img_np.copy(); rd[src_up==1]=[1,0,0]
        cv2.imwrite(str(outd/f"{Path(args.image).stem}_removed_fl{fl}.png"),(rd*255).astype(np.uint8)[...,::-1])
        # outline
        img_b=cv2.imread(str(args.image)); img_b=cv2.resize(img_b,(config.DATA.IMG_SIZE,config.DATA.IMG_SIZE))
        outl=img_b.copy(); ph,pw=config.DATA.IMG_SIZE//gh,config.DATA.IMG_SIZE//gw
        for r,c in dst: cv2.rectangle(outl,(c*pw,r*ph),((c+1)*pw,(r+1)*ph),(0,255,0),2)
        cv2.imwrite(str(outd/f"{Path(args.image).stem}_outline_fl{fl}.png"),outl)
        # text mask
        txt=outd/f"{Path(args.image).stem}_maskstage_fl{fl}.txt"
        np.savetxt(txt,mask,fmt="%d",delimiter=" ")
        print(f"→ stage grid({gh}×{gw}) saved: {txt}")
        # merged all
        keep_map=np.ones((gh,gw),np.uint8)
        for r,c in src: keep_map[r,c]=0
        keep_up=cv2.resize(keep_map,(config.DATA.IMG_SIZE,config.DATA.IMG_SIZE),interpolation=cv2.INTER_NEAREST)
        ma=img_np.copy(); ma[keep_up==0]=[1,0,0]; ma[keep_up==1]=[0,1,0]
        cv2.imwrite(str(outd/f"{Path(args.image).stem}_merged_all_fl{fl}.png"),(ma*255).astype(np.uint8)[...,::-1])
        # arrows
        arrow=draw_merge_arrows_with_outlines(img_b,src,dst,init_grid)
        cv2.imwrite(str(outd/f"{Path(args.image).stem}_merge_arrows_fl{fl}.png"),arrow)
        print(f"→ merge arrows saved: {outd}/{Path(args.image).stem}_merge_arrows_fl{fl}.png")
    print("\n✅ Done all layers!")