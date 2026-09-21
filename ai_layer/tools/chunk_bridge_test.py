"""control_bridge 검증.

  1. 규약 복사본 왕복: build -> encode -> decode 가 필드/바이트 크기 그대로인가 (3177 / 3695 byte)
  2. 한계 클립: validator 기준(policy 1 + v_max 0.15)으로 스텝이 잘리는가, 방향은 유지되는가
  3. eef 매핑 스위치 4종
  4. 스냅샷 -> state9 -> pose7 왕복, COMMIT_END/OBS_POSE 선택
  5. (--upstream <PAC_Supermoon clone>) 송지수 원본 코덱과 바이트 일치 + 원본 ChunkValidator 통과

실행: cd pac2026-team && PYTHONPATH=. python ai_layer/tools/chunk_bridge_test.py [--upstream /path/to/PAC_Supermoon]
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from ai_layer.control_bridge import protocol as P
from ai_layer.control_bridge.chunk_builder import BuildStats, ChunkLimits, EefMode, build_action_chunk, map_eef
from ai_layer.control_bridge.snapshot_adapter import choose_anchor, pose7_to_state9, state9_to_pose7

DT = 33_333_333


def make_model_chunk(T: int = 32, big: bool = False) -> np.ndarray:
    rng = np.random.default_rng(0)
    m = np.zeros((T, 7))
    m[:, :3] = rng.normal(0, 0.0008, size=(T, 3))  # |dp| 가 한계 0.005 를 우연히 넘지 않게
    m[:, 3:5] = rng.normal(0, 0.01, size=(T, 2))
    m[:, 5] = 0.0  # yaw 고정
    m[:, 6] = np.linspace(10, 90, T)  # gripper 0~100
    if big:
        m[3, :3] = [0.02, 0.0, 0.0]  # 2 cm 한 스텝 -> 한계 초과
        m[7, 3:6] = [0.0, 0.2, 0.0]  # 0.2 rad -> 한계 초과
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", default=None, help="PAC_Supermoon clone 경로 (jisu/control-layer)")
    args = ap.parse_args()

    # 1. 왕복
    m = make_model_chunk()
    t_obs = time.monotonic_ns()
    chunk = build_action_chunk(m, seq_id=7, snap_id=3, t_obs_ns=t_obs, anchor_mode=P.AnchorMode.COMMIT_END,
                               policy_id=1, dt_ns=DT, eef_mode=EefMode.GRIPPER_THRESHOLD, clip=False)
    buf = P.encode_chunk(chunk)
    assert len(buf) == P.CHUNK_SIZE == 3177, len(buf)
    back = P.decode_chunk(buf)
    assert back.seq_id == 7 and back.snap_id == 3 and back.t_obs_ns == t_obs and back.dt_ns == DT
    assert back.anchor_mode == P.AnchorMode.COMMIT_END and back.policy_id == 1 and back.n_steps == 32
    assert np.array_equal(back.steps, m[:, :6]) and np.array_equal(back.eef, (m[:, 6] >= 50).astype(np.uint8))
    print("[1] chunk encode/decode round-trip OK (3177 byte)")

    # 2. 한계 클립
    lim = ChunkLimits()
    pos_lim, rot_lim = lim.effective(DT)
    assert abs(pos_lim - 0.005) < 1e-9 and abs(rot_lim - 0.05) < 1e-9, (pos_lim, rot_lim)
    st = BuildStats()
    c2 = build_action_chunk(make_model_chunk(big=True), seq_id=1, snap_id=1, t_obs_ns=t_obs, dt_ns=DT, stats=st)
    pn = np.linalg.norm(c2.steps[:, :3], axis=1)
    rn = np.linalg.norm(c2.steps[:, 3:6], axis=1)
    assert pn.max() <= pos_lim + 1e-12 and rn.max() <= rot_lim + 1e-12
    assert st.pos_clipped_steps == 1 and st.rot_clipped_steps == 1, (st.pos_clipped_steps, st.rot_clipped_steps)
    assert np.allclose(c2.steps[3, :3] / pn[3], [1, 0, 0])  # 방향 유지
    print(f"[2] clip OK: pos<= {pos_lim:.4f} m, rot<= {rot_lim:.3f} rad, clipped {st.pos_clipped_steps}/{st.rot_clipped_steps}")

    # 3. eef 스위치
    g = np.array([0, 30, 50, 90, 1.0, 0.2])
    assert map_eef(g, EefMode.OFF).sum() == 0 and map_eef(g, EefMode.ON).sum() == 6
    assert list(map_eef(g, EefMode.GRIPPER_THRESHOLD, 50)) == [0, 0, 1, 1, 0, 0]
    assert list(map_eef(g[4:], EefMode.FROM_CHANNEL)) == [1, 0]
    print("[3] eef modes OK")

    # 4. 스냅샷 어댑터
    q = np.array([0.1, 0.2, 0.3, 0.9]); q /= np.linalg.norm(q)
    pose7 = np.concatenate([[0.3, -0.1, 0.12], q])
    s9 = pose7_to_state9(pose7)
    p7 = state9_to_pose7(s9)
    assert np.allclose(p7[:3], pose7[:3]) and (np.allclose(p7[3:], q, atol=1e-6) or np.allclose(p7[3:], -q, atol=1e-6))
    commit = P.CommitTrajectory(t_start_ns=1, dt_ns=10_000_000, poses=np.stack([pose7, pose7 + [0.01, 0, 0, 0, 0, 0, 0]]))
    snap = P.StateSnapshot(snap_id=11, t_meas_ns=2, measured=pose7, commit=commit, mode=P.ControlMode.TRACKING)
    a = choose_anchor(snap, P.AnchorMode.COMMIT_END)
    assert a.anchor_mode == P.AnchorMode.COMMIT_END and abs(a.state9[0] - 0.31) < 1e-6 and a.snap_id == 11
    b = choose_anchor(P.StateSnapshot(snap_id=12, measured=pose7), P.AnchorMode.COMMIT_END)
    assert b.anchor_mode == P.AnchorMode.OBS_POSE and abs(b.state9[0] - 0.30) < 1e-6  # commit 없으면 강등
    sbuf = P.encode_snapshot(snap)
    assert len(sbuf) == P.SNAPSHOT_SIZE == 3695
    sback = P.decode_snapshot(sbuf)
    assert sback.snap_id == 11 and sback.commit.n == 2 and np.allclose(sback.measured, pose7)
    print("[4] snapshot adapter + round-trip OK (3695 byte)")

    # 5. 원본 코덱/검증기와 대조
    if args.upstream:
        sys.path.insert(0, f"{args.upstream}/control")
        from control_layer.ingest.validator import ChunkValidator
        from control_layer.policy import PolicyTable
        from control_layer.protocol import chunk_codec as UC, snapshot_codec as US

        assert UC.CHUNK_SIZE == P.CHUNK_SIZE and US.SNAPSHOT_SIZE == P.SNAPSHOT_SIZE
        up = UC.decode_chunk(buf)  # 우리 바이트를 원본이 읽는다
        assert up.seq_id == 7 and up.n_steps == 32 and np.array_equal(up.steps, chunk.steps) and np.array_equal(up.eef, chunk.eef)
        assert UC.encode_chunk(up) == buf  # 원본이 다시 쓰면 바이트 동일
        ups = US.decode_snapshot(sbuf)
        assert ups.snap_id == 11 and US.encode_snapshot(ups) == sbuf
        policies = PolicyTable.load(f"{args.upstream}/control/config/chunk_policy.json")
        v = ChunkValidator(v_max=0.15, w_max=1.5)
        verdict = v.check_chunk(UC.decode_chunk(P.encode_chunk(c2)), time.monotonic_ns(), policies.get(1))
        assert verdict.ok, f"원본 validator 거부: {verdict.reason}"
        # 클립 안 한 큰 스텝은 거부되어야 정상
        raw = build_action_chunk(make_model_chunk(big=True), seq_id=2, snap_id=1, t_obs_ns=time.monotonic_ns(), dt_ns=DT, clip=False)
        v2 = ChunkValidator(v_max=0.15, w_max=1.5).check_chunk(UC.decode_chunk(P.encode_chunk(raw)), time.monotonic_ns(), policies.get(1))
        assert not v2.ok and v2.reason in ("step_pos_limit", "step_speed_limit"), v2.reason
        print(f"[5] upstream codec byte-identical + upstream ChunkValidator OK (unclipped rejected as {v2.reason!r})")
    else:
        print("[5] skipped (no --upstream)")

    print("CHUNK BRIDGE TEST OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
