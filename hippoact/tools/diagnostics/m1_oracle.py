#!/usr/bin/env python3
"""M1 天花板: 任何匹配方案能达到的上界是多少?

Hungarian 假设两帧的 slot 集合是同一批实体的两次呈现。若 fresh init 下相邻帧
的分解本来就不是同一套划分, 则不存在正确的 1:1 对应, 再怎么调代价函数也无用。

三个上界:
  greedy_oracle  : 每个 slot 各自选 t=2 上 obj2 质量最大的 slot (非双射, 硬上界)
  hungarian_oracle: 用真值 gain 作为代价做双射分配 (双射可达上界)
  best_learned   : sweep 里最好的那组 (w=0.5/0.5, topk=64)
并给出经验地板 (identity/random), 以及归一化捕获率
  (learned - floor) / (oracle - floor)
"""
from __future__ import annotations
import argparse, glob
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from scipy.optimize import linear_sum_assignment
from torchvision import transforms
from hippoact.utils.config import load_config
from validate_semantics import build_encoder

def pool_mask(m,g,dev):
    x=F.adaptive_avg_pool2d(torch.from_numpy(m.astype(np.float32))[None,None],(g,g)).reshape(-1).to(dev)
    return x/(x.sum()+1e-8)
def topk_binary(a,k):
    m=torch.zeros_like(a); m.scatter_(1,a.topk(k,dim=-1).indices,1.0); return m

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",required=True); ap.add_argument("--ckpt",required=True)
    ap.add_argument("--data-dir",required=True); ap.add_argument("--n-clips",type=int,default=400)
    ap.add_argument("--shared-init",action="store_true",
                    help="pair 内两帧共享同一 init (Step M1b); 默认每帧独立 fresh init (M1)")
    args=ap.parse_args()
    cfg=load_config(args.config); size=cfg.encoder.image_size
    enc=build_encoder(cfg).cuda()
    ck=torch.load(args.ckpt,map_location="cuda",weights_only=False)
    enc.load_state_dict(ck["encoder"],strict=False); enc.eval()
    SA=enc.slot_attn
    nm=transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    base=transforms.Compose([transforms.Resize(size),transforms.CenterCrop(size),transforms.ToTensor()])
    res={k:[] for k in ("greedy_oracle","hungarian_oracle","best_learned","identity","random")}
    partition_cos, loc, used = [], [], 0
    for c in sorted(glob.glob(args.data_dir+"/clip_*"))[:args.n_clips]:
        fs=sorted(glob.glob(c+"/*.png"))
        if len(fs)<4: continue
        f=[base(Image.open(p).convert("RGB")) for p in fs[:4]]
        d=[((f[i+1]-f[i]).abs().max(0).values>0.02).numpy() for i in range(3)]
        obj1,obj2=d[0]&d[1], d[1]&d[2]
        if obj1.sum()<30 or obj2.sum()<30: continue
        used+=1
        with torch.no_grad():
            g1,g2=enc.dino(nm(f[1])[None].cuda()),enc.dino(nm(f[2])[None].cuda())
            N=g1.shape[1]; gg=int(round(N**0.5))
            if args.shared_init:
                init = SA.sample_init(1,device=g1.device,dtype=g1.dtype)
                s1, s2 = SA(g1,slots_init=init), SA(g2,slots_init=init)
            else:
                s1=SA(g1,slots_init=SA.sample_init(1,device=g1.device,dtype=g1.dtype))
                s2=SA(g2,slots_init=SA.sample_init(1,device=g2.device,dtype=g2.dtype))
            _,a1=enc.slot_decoder(s1); _,a2=enc.slot_decoder(s2)
        A1=a1[0]/(a1[0].sum(-1,keepdim=True)+1e-8)
        A2=a2[0]/(a2[0].sum(-1,keepdim=True)+1e-8)
        K=A1.shape[0]
        o1,o2=pool_mask(obj1,gg,A1.device),pool_mask(obj2,gg,A1.device)
        on1=(A1*o1).sum(-1)*N
        loc.append((on1>2.0).float().mean().item())
        sel=(on1>2.0).nonzero().flatten()
        if len(sel)==0: continue
        base_mass=(A1[sel]*o2).sum(-1)*N
        mass2=(A2*o2).sum(-1)*N                     # (K,) 每个 t=2 slot 在 obj2 上的质量
        # greedy oracle: 每个 sel slot 独立选最大
        res["greedy_oracle"] += ((mass2.max().expand(len(sel))-base_mass)>0).float().tolist()
        # hungarian oracle: 双射可达上界。目标必须是「gain>0 的个数」，
        # 且只在被选中的行上分配 —— 用 mass2[j]-base_i 作代价是可加分离的,
        # 任何双射总和都是常数, Hungarian 会退化成任意分配 (先前的 bug)。
        win = ((mass2[None,:] - base_mass[:,None]) > 0).float()   # (|sel|, K)
        _, cc = linear_sum_assignment(-win.cpu().numpy())
        res["hungarian_oracle"] += win[np.arange(len(sel)), cc].cpu().tolist()
        # best learned: w=0.5/0.5, topk=64
        k=64; bp,bc=topk_binary(A1,k),topk_binary(A2,k)
        inter=bp@bc.t(); iou=inter/(2*k-inter+1e-8)
        cos=F.normalize(s1[0],dim=-1)@F.normalize(s2[0],dim=-1).t()
        _,cl=linear_sum_assignment(-(0.5*iou+0.5*cos).cpu().numpy())
        pl=torch.as_tensor(cl,device=A1.device)
        res["best_learned"] += (((A2[pl[sel]]*o2).sum(-1)*N - base_mass)>0).float().tolist()
        for name,p in (("identity",torch.arange(K,device=A1.device)),
                       ("random",torch.randperm(K,device=A1.device))):
            res[name] += (((A2[p[sel]]*o2).sum(-1)*N - base_mass)>0).float().tolist()
        # 分解稳定性: 两帧 alpha 图集合的最优对齐余弦
        u=F.normalize(A1,dim=-1); v=F.normalize(A2,dim=-1)
        sim=(u@v.t()).cpu().numpy()
        _,cm=linear_sum_assignment(-sim)
        partition_cos.append(float(sim[np.arange(K),cm].mean()))
    print(f"clips_used={used}   shared_init={args.shared_init}")
    print(f"localization (on-object 交集) = {np.mean(loc):.3f}"
          f"    [参照 shared-init ckpt = 0.510]")
    print()
    for k in ("random","identity","best_learned","hungarian_oracle","greedy_oracle"):
        print(f"  {k:<18} = {np.mean(res[k]):.3f}")
    fl=np.mean(res["identity"]); le=np.mean(res["best_learned"]); orc=np.mean(res["hungarian_oracle"])
    print()
    print(f"  经验地板 (identity)      = {fl:.3f}")
    print(f"  双射可达上界 (oracle)    = {orc:.3f}")
    print(f"  归一化捕获率 = ({le:.3f}-{fl:.3f})/({orc:.3f}-{fl:.3f}) = "
          f"{(le-fl)/max(orc-fl,1e-9):.3f}")
    print()
    print(f"  相邻帧 alpha 分解的最优对齐余弦 = {np.mean(partition_cos):.3f}"
          f"   [1.0 = 两帧是同一套划分]")

if __name__ == "__main__":
    main()
