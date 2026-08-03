#!/usr/bin/env python3
"""W2.1 E2E-0 pre-flight. Three checks, all of them things that would otherwise
surface only as bad returns after a two-day run.

  1. env contract  — DCS with obs=hippoact emits {"rgb": uint8 (3,224,224),
     "state": (d_q,)}, and the underlying physics is unchanged (same reward
     sequence as the pixel baseline under the same seed and actions).
  2. encoder wiring — the world model builds HippoActAdapter from the Stage-1
     checkpoint, the architecture comes from the checkpoint's own snapshot,
     and encode() returns TD-MPC2's latent_dim for both the (B,...) acting
     shape and the (T,B,...) update shape.
  3. MPPI call count — **the expensive failure mode**. TD-MPC2 plans by rolling
     out in latent space, so the encoder must run exactly ONCE per act().
     If planning re-encoded, each env step would cost
     num_samples(512) x horizon(3) DINOv2 forwards instead of 1, i.e. a ~1500x
     slowdown that looks like "the encoder is just slow". PHASE2_PLAN §6 flags
     this; this is the assertion for it.

Run from the project root with the hippoact env's python.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "tdmpc2" / "tdmpc2"))
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

CKPT = os.environ.get(
    "HIPPOACT_CKPT",
    str(PROJECT_ROOT / "hippoact" / "outputs" / "stage1_seed1" / "ckpt_final.pt"),
)


def build_cfg(task, obs, **over):
    from common.parser import parse_cfg
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(PROJECT_ROOT / "third_party" / "tdmpc2" / "tdmpc2" / "config.yaml")
    cfg.task, cfg.obs = task, obs
    cfg.model_size, cfg.seed = 5, 1
    cfg.multitask = False
    for k, v in over.items():
        cfg[k] = v
    # parse_cfg wants a hydra runtime for work_dir; set the fields it derives
    cfg.work_dir = "/tmp/e2e0_smoke"
    cfg.task_title = task
    cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins - 1)
    from common import MODEL_SIZE
    for k, v in MODEL_SIZE[5].items():
        cfg[k] = v
    cfg.task_dim = 0
    cfg.tasks = [task]
    return cfg


def main():
    from envs import make_env
    ok = True

    print("=" * 72)
    print("1. env contract (obs=hippoact)")
    cfg = build_cfg("dcs-easy-walker-walk", "hippoact")
    env = make_env(cfg)
    obs = env.reset()
    rgb, state = obs["rgb"], obs["state"]
    print(f"   rgb   {tuple(rgb.shape)} {rgb.dtype}  range [{int(rgb.min())}, {int(rgb.max())}]")
    print(f"   state {tuple(state.shape)} {state.dtype}")
    print(f"   cfg.obs_shape = {dict(cfg.obs_shape)}")
    c1 = tuple(rgb.shape) == (3, 224, 224) and rgb.dtype == torch.uint8
    print(f"   -> {'PASS' if c1 else 'FAIL'} (uint8 3x224x224 as specified)")
    ok &= c1

    # physics must be untouched: same seed + same actions as the pixel arm
    rng = np.random.default_rng(0)
    acts = [torch.from_numpy(rng.uniform(-1, 1, env.action_space.shape).astype(np.float32))
            for _ in range(30)]
    def rewards(c):
        e = make_env(c); e.reset()
        return np.array([float(e.step(a)[1]) for a in acts])
    r_hip = rewards(build_cfg("dcs-easy-walker-walk", "hippoact"))
    r_pix = rewards(build_cfg("dcs-easy-walker-walk", "rgb"))
    dr = float(np.abs(r_hip - r_pix).max())
    c2 = dr < 1e-6
    print(f"   reward seq vs pixel arm: max|Δ| = {dr:.3e} -> {'PASS' if c2 else 'FAIL'}")
    ok &= c2

    print("=" * 72)
    print("2. encoder wiring")
    if not Path(CKPT).exists():
        print(f"   SKIP — no checkpoint at {CKPT}")
        return 0 if ok else 1
    cfg = build_cfg("dcs-easy-walker-walk", "hippoact",
                    encoder_type="hippoact", hippoact_ckpt=CKPT)
    env = make_env(cfg)
    from common.parser import cfg_to_dataclass
    from common.world_model import WorldModel
    dcfg = cfg_to_dataclass(cfg)
    model = WorldModel(dcfg).to("cuda")
    adapter = model._encoder
    print(f"   adapter: num_slots={adapter.num_slots} slot_dim={adapter.slot_dim} "
          f"image_size={adapter.image_size}  (read from ckpt snapshot)")
    frozen = all(not p.requires_grad for p in adapter.encoder.parameters())
    print(f"   Stage-1 modules frozen: {frozen}")

    o = env.reset()
    td = {k: v.unsqueeze(0).cuda() for k, v in o.items()}
    z = model.encode(td, None)
    c3 = tuple(z.shape) == (1, dcfg.latent_dim)
    print(f"   encode (B,...) -> {tuple(z.shape)}  expect (1,{dcfg.latent_dim}) "
          f"-> {'PASS' if c3 else 'FAIL'}")
    td5 = {k: v.unsqueeze(0).repeat(4, *([1] * v.ndim)) for k, v in td.items()}
    z5 = model.encode(td5, None)
    c4 = tuple(z5.shape) == (4, 1, dcfg.latent_dim)
    print(f"   encode (T,B,...) -> {tuple(z5.shape)}  expect (4,1,{dcfg.latent_dim}) "
          f"-> {'PASS' if c4 else 'FAIL'}")
    ok &= c3 and c4

    print("=" * 72)
    print("3. MPPI must call the encoder exactly once per act()")
    from tdmpc2 import TDMPC2
    cfg2 = build_cfg("dcs-easy-walker-walk", "hippoact",
                     encoder_type="hippoact", hippoact_ckpt=CKPT,
                     compile=False, mpc=True)
    env = make_env(cfg2)          # populates obs_shape / action_dim / seed_steps
    dcfg2 = cfg_to_dataclass(cfg2)
    agent = TDMPC2(dcfg2)
    enc = agent.model._encoder
    o = env.reset()
    enc._call_count = 0
    for t in range(3):
        a = agent.act(o, t0=(t == 0), eval_mode=True)
        o = env.step(a)[0]
    calls = enc._call_count
    c5 = calls == 3
    print(f"   3 act() calls -> encoder invoked {calls} times "
          f"(expect 3; {dcfg2.num_samples}x{dcfg2.horizon} would mean re-encoding "
          f"inside planning)")
    print(f"   -> {'PASS' if c5 else 'FAIL'}")
    ok &= c5

    print("=" * 72)
    print("4. fast path: env-side encoding must equal in-model encoding")
    # This is the claim the whole speedup rests on. With Stage-1 frozen the two
    # are the same map; assert it on real frames instead of trusting it.
    cfg3 = build_cfg("dcs-easy-walker-walk", "state", hippoact_precompute=CKPT)
    env3 = make_env(cfg3)
    print(f"   obs_shape = {dict(cfg3.obs_shape)}  "
          f"(= {adapter.num_slots}x{adapter.slot_dim} + proprio)")
    o3 = env3.reset()
    obs_env = o3[:adapter.num_slots * adapter.slot_dim]

    # Same physics state, same render. Walk down to DMControlWrapper rather
    # than counting wrappers (TensorWrapper/Timeout/HippoActSlots/DMControl).
    dmc_env = env3
    while type(dmc_env).__name__ != "DMControlWrapper":
        dmc_env = dmc_env.env
    f1 = dmc_env.render(width=224, height=224)
    f2 = dmc_env.render(width=224, height=224)
    dpix = int(np.abs(f1.astype(np.int32) - f2.astype(np.int32)).max())
    print(f"   render determinism: max|Δpixel| = {dpix}")

    # Feed that exact frame through a freshly built extractor — i.e. the code
    # path the in-model adapter uses. This is the claim the speedup rests on.
    from hippoact.adapters.slot_features import SlotFeatureExtractor
    rgb = torch.from_numpy(np.ascontiguousarray(f1.transpose(2, 0, 1)))
    with torch.no_grad():
        obs_model = SlotFeatureExtractor(CKPT).cuda().eval()(
            rgb.unsqueeze(0).cuda()).squeeze(0).cpu()
    dmax = float((obs_env - obs_model).abs().max())
    c6 = dpix == 0 and dmax < 1e-5
    print(f"   max|Δ features| env-side vs in-model = {dmax:.3e} "
          f"-> {'PASS' if c6 else 'FAIL'}")
    nonzero = int((obs_env != 0).sum())
    print(f"   non-zero slot features: {nonzero}/{len(obs_env)} "
          f"(zeros are slow-routed slots, expected)")
    ok &= c6

    # The equivalence only holds because the encoder is a deterministic
    # function of the frame. In `sampled` query mode it is not, unless the slot
    # init is fixed — assert that here rather than assuming it.
    ex_a = SlotFeatureExtractor(CKPT, slot_init_seed=0).cuda().eval()
    ex_b = SlotFeatureExtractor(CKPT, slot_init_seed=0).cuda().eval()
    ex_c = SlotFeatureExtractor(CKPT, slot_init_seed=1).cuda().eval()
    r = rgb.unsqueeze(0).cuda()
    with torch.no_grad():
        fa, fb, fc = ex_a(r).clone(), ex_b(r).clone(), ex_c(r).clone()
    d_same = float((fa - fb).abs().max())
    d_diff = float((fa - fc).abs().max())
    c7 = d_same < 1e-6 and d_diff > 1e-3
    print(f"   two instances, slot_init_seed=0: max|Δ| = {d_same:.2e} (must be 0)")
    print(f"   slot_init_seed 0 vs 1:           max|Δ| = {d_diff:.2f} "
          f"(the init is part of the encoder's identity — CP5e)")
    print(f"   -> {'PASS' if c7 else 'FAIL'}")
    ok &= c7

    print("=" * 72)
    print("RESULT:", "ALL PASS" if ok else "CHECK FAILURES ABOVE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
