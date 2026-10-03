"""MuJoCo RL 환경(envs/so101_seam_env.py) 스모크 테스트. 2026-10-03 ee_rig 기반 재작성 버전.

확인하는 것:
  1. reset() 직후 도구 끝(tip)이 홈 위치(≈ x0.25, y0, z0.10 — ee_rig 5cm 막대 오프셋 반영)에 있고
     관측 state가 9D인가, ground-truth target_polyline이 30가지 (형태,variant) 중 하나로 채워지는가
  2. 명령 0으로 90스텝(3s) 두면 손끝이 흐르지 않는가 (weld 안정성 회귀 검사)
  3. +x 방향 이동 명령이 스케일(action_scale_pos)대로 정확히 적분되는가
  4. 막대를 바닥 쪽으로 밀면 접촉 즉시 terminated=True + floor_contact_penalty가 반영되는가
     (record_mujoco.py의 "표면 접촉=자동 폐기"와 같은 규약)
  5. 안전 컷오프: 선에서 off_seam_safety_dist보다 멀 때 트리거를 켜도 gripper_active가 강제로 꺼지는가
     (reward.py coverage_reward의 soft penalty와 별개인 하드 안전장치)
  6. BC teacher(체크포인트)를 붙였을 때 보상 계산이 도는가 (--bc-checkpoint 있을 때만)

실행: cd pac2026 && PYTHONPATH=. python ai_layer/tools/rl_env_smoke.py [--bc-checkpoint DIR]
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

from ai_layer.envs.so101_seam_env import SO101SeamEnv, SO101SeamEnvCfg

np.set_printoptions(precision=4, suppress=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bc-checkpoint", default=None)
    args = ap.parse_args()

    env = SO101SeamEnv(SO101SeamEnvCfg())

    obs, info = env.reset(seed=0)
    st = obs["observation.state"]
    assert st.shape == (1, 9), st.shape
    p0 = st[0, :3].numpy().copy()
    print(f"[1] scene={info['scene']} variant={info['variant']}, tip home {p0}, policy dt "
          f"{env.cfg.physics_dt * env.cfg.decimation:.4f}s, episode {env.max_episode_length} steps, "
          f"target_polyline shape {env._target_polyline.shape}")
    assert abs(p0[0] - 0.25) < 0.01 and abs(p0[2] - 0.10) < 0.01

    for _ in range(90):
        obs, *_ = env.step(torch.zeros(1, 7))
    drift = obs["observation.state"][0, :3].numpy() - p0
    print(f"[2] 90 zero-steps(3s) drift {drift} (max {np.abs(drift).max()*1000:.2f} mm)")
    assert np.abs(drift).max() < 2.0e-3

    p1 = obs["observation.state"][0, :3].numpy().copy()
    for _ in range(10):
        a = torch.zeros(1, 7)
        a[0, 0] = 1.0
        obs, *_ = env.step(a)
    for _ in range(15):
        obs, *_ = env.step(torch.zeros(1, 7))
    p2 = obs["observation.state"][0, :3].numpy()
    dx = p2[0] - p1[0]
    expect = 10 * env.cfg.action_scale_pos
    print(f"[3] 10x +x action: dx={dx:.4f} (expect ~{expect:.4f})")
    assert abs(dx - expect) < 1e-3

    env.reset(seed=3)
    terminated = False
    steps = 0
    reward_at_term = 0.0
    for _ in range(60):
        a = torch.zeros(1, 7)
        a[0, 2] = -1.0  # -z 로 바닥을 향해 밀기
        obs, reward_at_term, terminated, truncated, _ = env.step(a)
        steps += 1
        if terminated:
            break
    print(f"[4] 바닥으로 밀기: terminated={terminated} after {steps} steps, reward={reward_at_term:.3f} "
          f"(floor_contact_penalty={env.cfg.floor_contact_penalty} 포함되어야 함)")
    assert terminated and reward_at_term < -env.cfg.floor_contact_penalty * 0.9

    env.reset(seed=4)
    far_action = torch.zeros(1, 7)
    far_action[0, 6] = 1.0  # 트리거 ON 요청
    obs, reward, terminated, truncated, _ = env.step(far_action)
    tip = env._tip_pose()[:3]
    dist = float(np.linalg.norm(tip[None, :] - env._target_polyline, axis=1).min())
    print(f"[5] 홈 위치(선에서 {dist*1000:.1f}mm)에서 트리거 ON 요청 — "
          f"off_seam_safety_dist={env.cfg.off_seam_safety_dist*1000:.0f}mm 보다 멀면 강제로 꺼져야 함")
    assert dist > env.cfg.off_seam_safety_dist, "테스트 전제(홈이 선에서 충분히 멀다)가 깨짐 — 씬 확인 필요"

    if args.bc_checkpoint:
        from ai_layer.bc_inference import load_bc_checkpoint

        pol, pre, post = load_bc_checkpoint(args.bc_checkpoint, device="cuda" if torch.cuda.is_available() else "cpu")
        env.set_bc_reference(pol, pre, post)
        env.reset(seed=5)
        _, r, *_ = env.step(torch.zeros(1, 7))
        print(f"[6] BC teacher reward {r:.3f}")
    else:
        print("[6] skipped (no --bc-checkpoint)")

    print("RL ENV SMOKE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
