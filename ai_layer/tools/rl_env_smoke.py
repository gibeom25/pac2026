"""MuJoCo RL 환경(envs/so101_seam_env.py) 스모크 테스트. IsaacLab 불필요, env_lerobot 에서 바로 실행.

확인하는 것:
  1. 시작(home) 자세에서 손끝이 경로 영역 위(≈ x 0.25, z 0.10)에 있고 관측 state 가 9D 인가
  2. 명령 0 으로 60 스텝(2 s) 두면 손끝이 흐르지 않는가 (서보 처짐/IK 잔차 누적 검사)
  3. 경로 영역 안의 이동/회전 명령을 정지 후 정확히 따라가는가 (±1 cm, ±0.03 rad)
  4. BC teacher(체크포인트)를 붙였을 때 보상 계산이 도는가 (--bc-checkpoint 있을 때만)
  5. 도달 한계 근처(x 0.45)로 밀면 IK 가 무너지지 않고 스냅으로 버티는가 (실패가 아니라 경고 출력)

실행: cd pac2026-team && PYTHONPATH=. /home/dy/pac2026/env_lerobot/bin/python ai_layer/tools/rl_env_smoke.py [--bc-checkpoint DIR]
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rt

from ai_layer.envs.so101_seam_env import SO101SeamEnv, SO101SeamEnvCfg

np.set_printoptions(precision=4, suppress=True)


def run(env, vec, n, settle=15):
    env.reset(seed=0)
    T0 = env._eef_pose()
    p0, R0 = T0[:3, 3].copy(), T0[:3, :3].copy()
    for _ in range(n):
        a = torch.zeros(1, 7)
        a[0, :6] = torch.tensor(vec, dtype=torch.float32)
        env.step(a)
    for _ in range(settle):
        env.step(torch.zeros(1, 7))
    T = env._eef_pose()
    return T[:3, 3] - p0, Rt.from_matrix(T[:3, :3] @ R0.T).as_rotvec()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bc-checkpoint", default=None)
    args = ap.parse_args()

    env = SO101SeamEnv(SO101SeamEnvCfg())
    obs, _ = env.reset(seed=0)
    st = obs["observation.state"]
    assert st.shape == (1, 9), st.shape
    p_home = st[0, :3].numpy()
    print(f"[1] home EE {p_home}, policy dt {env.cfg.physics_dt * env.cfg.decimation:.4f} s, episode {env.max_episode_length} steps")
    assert abs(p_home[0] - 0.25) < 0.02 and abs(p_home[2] - 0.10) < 0.02

    for _ in range(60):
        obs, *_ = env.step(torch.zeros(1, 7))
    drift = obs["observation.state"][0, :3].numpy() - p_home
    print(f"[2] 60 zero-steps drift {drift} (max {np.abs(drift).max()*1000:.2f} mm)")
    assert np.abs(drift).max() < 0.002

    dp, dw = run(env, [0, -0.5, 0, 0, 0, 0], 10)
    print(f"[3a] 10× -y 1cm : dp {dp} dw {dw}")
    assert abs(dp[1] + 0.10) < 0.01 and abs(dp[2]) < 0.01
    dp, dw = run(env, [0, 0, -0.5, 0, 0, 0], 10)
    print(f"[3b] 10× -z 1cm : dp {dp} dw {dw}")
    assert abs(dp[2] + 0.10) < 0.01
    dp, dw = run(env, [0.5, 0, 0, 0, 0, 0], 10)
    print(f"[3c] 10× +x 1cm : dp {dp} (x 0.25→0.35 = 경로 영역 상한)")
    assert abs(dp[0] - 0.10) < 0.01 and abs(dp[2]) < 0.01
    dp, dw = run(env, [0.5, 0, 0, 0, 1.0, 0], 4)
    print(f"[3d] 4× +x1cm +ry0.05 : dp {dp} dw {dw}")
    assert abs(dw[1] - 0.2) < 0.03 and abs(dp[0] - 0.04) < 0.01

    if args.bc_checkpoint:
        from ai_layer.bc_inference import load_bc_checkpoint

        pol, pre, post = load_bc_checkpoint(args.bc_checkpoint, device="cuda" if torch.cuda.is_available() else "cpu")
        env.set_bc_reference(pol, pre, post)
        _, r, *_ = env.step(torch.zeros(1, 7))
        print(f"[4] BC teacher reward {r:.3f}")
    else:
        print("[4] skipped (no --bc-checkpoint)")

    dp, dw = run(env, [0.5, 0, 0, 0, 0, 0], 20)
    print(f"[5] 20× +x 1cm (도달 한계 근처, 경고용): dp {dp} — 실패 아님. 경로 x 상한 0.35 를 넘는 목표는 IK 가 못 따라간다")

    print("RL ENV SMOKE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
