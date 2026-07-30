#!/usr/bin/env python3
"""Is a HIGHER on-object fraction actually better, or just redundant?

The scene has 2-5 disks among K=16 slots, so the ideal on-object fraction is
~0.13-0.31. A much higher fraction means several slots pile onto the SAME
object. Measures, among slots with on-object > 2.0:
  - how many DISTINCT objects (connected components) they cover
  - pairwise IoU of their top-k patch sets (redundancy)
  - objects covered per on-object slot  (1.0 = no redundancy)
"""
from __future__ import annotations
import argparse, glob
import numpy as np, torch, torch.nn.functional as F
import scipy.ndimage as ndi
from PIL import Image
from torchvision import transforms
from hippoact.utils.config import load_config
from validate_semantics import build_encoder

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", required=True); ap.add_argument("--n-clips", type=int, default=120)
    ap.add_argument("--mode", default="carryover"); ap.add_argument("--label", default="")
    args = ap.parse_args()
    cfg = load_config(args.config); size = cfg.encoder.image_size
    enc = build_encoder(cfg).cuda()
    ck = torch.load(args.ckpt, map_location="cuda", weights_only=False)
    enc.load_state_dict(ck["encoder"], strict=False); enc.eval()
    SA = enc.slot_attn
    norm = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    base = transforms.Compose([transforms.Resize(size), transforms.CenterCrop(size),
                               transforms.ToTensor()])
    n_obj_l, n_slot_l, n_cov_l, iou_l = [], [], [], []
    for c in sorted(glob.glob(args.data_dir + "/clip_*"))[:args.n_clips]:
        fs = sorted(glob.glob(c + "/*.png"))
        if len(fs) < 3: continue
        f0,f1,f2 = [base(Image.open(f).convert("RGB")) for f in fs[:3]]
        d01 = (f1-f0).abs().max(0).values > 0.02
        d12 = (f2-f1).abs().max(0).values > 0.02
        obj1 = (d01 & d12).cpu().numpy()
        if obj1.sum() < 30: continue
        lbl, n_obj = ndi.label(obj1)
        keep = [i for i in range(1, n_obj+1) if (lbl==i).sum() >= 30]
        if not keep: continue
        x1 = norm(f1)[None].cuda()
        with torch.no_grad():
            ft = enc.dino(x1)
            init = SA.sample_init(1, device=ft.device, dtype=ft.dtype)
            if args.mode == "carryover":
                sp = SA(enc.dino(norm(f0)[None].cuda()), slots_init=init)
                s1 = SA(ft, slots_init=sp.detach())
            else:
                s1 = SA(ft, slots_init=init)
            _, a1 = enc.slot_decoder(s1)
        K, N = a1.shape[1], a1.shape[2]; g = int(round(N**0.5))
        an = (a1[0]/(a1[0].sum(-1,keepdim=True)+1e-8))
        # per-object pooled masks
        omasks = []
        for i in keep:
            om = F.adaptive_avg_pool2d(torch.from_numpy((lbl==i).astype(np.float32))[None,None],
                                       (g,g)).reshape(-1).cuda()
            omasks.append(om/(om.sum()+1e-8))
        allobj = torch.stack(omasks).sum(0); allobj = allobj/(allobj.sum()+1e-8)
        on = (an*allobj).sum(-1)*N
        sel = (on > 2.0).nonzero().flatten()
        if len(sel) == 0: continue
        # which object does each selected slot cover most?
        assign = []
        for k in sel.tolist():
            scores = [float((an[k]*om).sum()*N) for om in omasks]
            assign.append(int(np.argmax(scores)))
        n_obj_l.append(len(keep)); n_slot_l.append(len(sel)); n_cov_l.append(len(set(assign)))
        # redundancy: pairwise IoU of top-k patches among selected slots
        if len(sel) > 1:
            kk = max(1, N//16)
            top = an[sel].topk(kk, dim=-1).indices
            m = torch.zeros(len(sel), N, device=an.device); m.scatter_(1, top, 1.0)
            inter = m @ m.t(); iou = inter/(2*kk-inter+1e-8)
            off = iou[~torch.eye(len(sel), dtype=bool, device=an.device)]
            iou_l.append(off.mean().item())
    print(f"{args.label or args.ckpt}  (mode={args.mode}, step={ck['step']})")
    print(f"  场景真实物体数        : {np.mean(n_obj_l):.2f}")
    print(f"  强锁物体的 slot 数    : {np.mean(n_slot_l):.2f}  (/16 = {np.mean(n_slot_l)/16:.3f})")
    print(f"  这些 slot 覆盖到的物体数: {np.mean(n_cov_l):.2f}")
    print(f"  物体/slot 比 (1.0=无冗余): {np.mean(n_cov_l)/max(np.mean(n_slot_l),1e-9):.3f}")
    print(f"  这些 slot 两两 IoU (0=不重叠): {np.mean(iou_l) if iou_l else float('nan'):.3f}")

if __name__ == "__main__":
    main()
