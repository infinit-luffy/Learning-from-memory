#!/usr/bin/env python3
"""Collect walker frames from DMC / Distracting Control Suite for Stage-1.

Writes the clip layout the Stage-1 loader auto-detects, plus per-frame qpos/qvel
so the same data can later feed the proprio side of P2:

    out/clip_000000/frame_0000.png ... frame_00NN.png
    out/clip_000000/states.npz          qpos (T, nq), qvel (T, nv), action (T, na)

Rendering is done through ``env.physics.render`` rather than the pixels wrapper
so that qpos/qvel stay reachable; the distraction wrappers act on the physics /
skybox, so backgrounds still appear in the render.

Requires MUJOCO_GL=egl on a headless machine.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image


def build_env(domain, task, difficulty, davis_path, seed):
    if difficulty in (None, "none", "clean"):
        from dm_control import suite
        return suite.load(domain, task, task_kwargs={"random": seed})
    from distracting_control import suite as dcs
    return dcs.load(
        domain, task,
        difficulty=difficulty,
        background_dataset_path=davis_path,
        distraction_types=("background",),   # W1.1 只关心背景干扰
        dynamic=True,                        # 视频背景随时间播放
        task_kwargs={"random": seed},
        from_pixels=False, pixels_only=False,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="walker")
    ap.add_argument("--task", default="walk")
    ap.add_argument("--difficulty", default="clean", help="clean | easy | medium | hard")
    ap.add_argument("--davis-path", default="/var/tmp/hippoact_dcs/davis/DAVIS")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-frames", type=int, default=25000)
    ap.add_argument("--clip-len", type=int, default=25)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--camera-id", type=int, default=0)
    ap.add_argument("--rebuild-every", type=int, default=8,
                    help="每 N 个 clip 重建环境（换 DAVIS 背景视频）")
    ap.add_argument("--reset-every", type=int, default=2,
                    help="每 N 个 clip 重置回合（换姿态）")
    ap.add_argument("--save-segmentation", action="store_true",
                    help="额外存精确 walker 掩膜 (MuJoCo 分割渲染). "
                         "geom 0=floor, 1-7=walker 部件, -1=天空; "
                         "天空盒被 DAVIS 替换后仍返回 -1, 故 easy 上同样精确. "
                         "优先于帧差+闭运算的 proxy: 后者会把地板倒影一并圈入.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    assert os.environ.get("MUJOCO_GL") == "egl", "需要 MUJOCO_GL=egl"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    n_clips = args.n_frames // args.clip_len

    rng = np.random.default_rng(args.seed)
    written = 0
    ep, env, ts = -1, None, None
    # 重建环境 vs 重置回合分开：
    #   重建（换 DAVIS 视频）代价高，每 rebuild_every 个 clip 一次
    #   重置（换姿态）代价低，每 reset_every 个 clip 一次 —— 随机策略下 walker
    #   约 25 步就倒地，重置太稀会让绝大多数帧都是「躺平」，姿态多样性不足
    for c in range(n_clips):
        if c % args.rebuild_every == 0:
            ep += 1
            env = build_env(args.domain, args.task, args.difficulty,
                            args.davis_path, args.seed * 1000 + ep)
            ts = env.reset()
        elif c % args.reset_every == 0:
            ts = env.reset()
        spec = env.action_spec()
        cdir = out / f"clip_{c:06d}"; cdir.mkdir(exist_ok=True)
        qpos, qvel, acts, segs = [], [], [], []
        for t in range(args.clip_len):
            img = env.physics.render(height=args.size, width=args.size,
                                     camera_id=args.camera_id)
            Image.fromarray(img).save(cdir / f"frame_{t:04d}.png")
            if args.save_segmentation:
                seg = env.physics.render(height=args.size, width=args.size,
                                         camera_id=args.camera_id,
                                         segmentation=True)
                segs.append((seg[..., 0] >= 1).astype(np.uint8))
            qpos.append(env.physics.data.qpos.copy())
            qvel.append(env.physics.data.qvel.copy())
            a = rng.uniform(spec.minimum, spec.maximum, spec.shape).astype(np.float32)
            acts.append(a)
            ts = env.step(a)
            if ts.last():
                ts = env.reset()
            written += 1
        payload = dict(qpos=np.array(qpos), qvel=np.array(qvel),
                       action=np.array(acts))
        if args.save_segmentation:
            payload["walker_mask"] = np.array(segs)     # (T, H, W) uint8
        np.savez_compressed(cdir / "states.npz", **payload)
        if (c + 1) % 50 == 0:
            print(f"  {c+1}/{n_clips} clips  ({written} frames)", flush=True)
    print(f"Done. {n_clips} clips x {args.clip_len} = {written} frames "
          f"[{args.difficulty}] in {out}")


if __name__ == "__main__":
    main()
