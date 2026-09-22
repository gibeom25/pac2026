"""AI <-> 제어 계층 규약 (송지수 선배 PAC_Supermoon `jisu/control-layer` 에서 그대로 옮김).

원본: control/control_layer/protocol/{messages.py, chunk_codec.py, snapshot_codec.py}
원본 커밋: da9f29f (2026-09-18). 규약이 바뀌면 이 파일을 다시 옮기고
`ai_layer/tools/chunk_bridge_test.py --upstream <clone>` 으로 바이트 일치를 확인할 것.

AI 환경(env_lerobot)은 제어 저장소를 import 하지 않으므로(환경·프로세스 분리 원칙) 여기 복사본을 쓴다.
로직은 한 줄도 바꾸지 않았다. 세 파일을 한 모듈로 합치면서 상대 import 와 중복 정의만 정리했다.
"""

from __future__ import annotations

import struct

import numpy as np


class DecodeError(ValueError):
    pass


# ============================================================================ messages.py
"""AI 추론 계층 <-> 제어 계층 메시지 정의.

PDF 규약
--------
- Action sequence 는 EEF delta 로 통일한다.
- EEF 신호 = (dx, dy, dz, droll, dpitch, dyaw + EEF 동작 신호 0/1)

여기서 확정한 세부 규약
----------------------
- delta 는 **직전 스텝 기준 증분**.  스텝 k 는 스텝 k-1 의 목표에서의 변위.
      p_k = p_{k-1} + dp_k
      R_k = Exp(omega_k) @ R_{k-1}        (월드 좌표계, 왼쪽 곱)
  스텝 0 의 기준(anchor)은 anchor_mode 가 정한다.
- (droll, dpitch, dyaw) 는 **회전 벡터** (axis-angle, rad) 로 해석한다. rpy 아님.
- 단위: 위치 m, 회전 rad, 시각 ns (CLOCK_MONOTONIC).
- 증분이라 스텝이 하나라도 빠지면 뒤가 전부 틀어진다.
  따라서 청크는 반드시 한 메시지로 통째로(원자적으로) 주고받는다.
"""


from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

PROTOCOL_VERSION = 1

MAX_STEPS = 64  # 청크 한 개의 최대 스텝 수
MAX_COMMIT = 64  # 스냅샷이 실어보내는 약속 궤적의 최대 점 개수


class AnchorMode(IntEnum):
    """청크의 스텝 0 이 어느 pose 를 기준으로 하는가."""

    OBS_POSE = 0  # t_obs 시점의 측정 pose (제어 계층이 pose 이력에서 조회)
    COMMIT_END = 1  # 스냅샷의 commit_end 시점 예정 pose (AI 가 미래를 기준으로 계획)


class ControlMode(IntEnum):
    """제어 계층 상태. 스냅샷에 실어 AI 계층에 알려준다."""

    IDLE = 0
    TRACKING = 1
    DRAINING = 2  # 버퍼 고갈 -> 시간축을 늘려 감속 중
    HOLDING = 3  # 정지, 마지막 명령 pose 유지
    FAULT = 4


@dataclass
class ActionChunk:
    """AI 추론 계층 -> 제어 계층."""

    seq_id: int = 0  # 단조 증가
    snap_id: int = 0  # 기준으로 삼은 StateSnapshot
    t_obs_ns: int = 0  # 사용한 이미지의 촬영 시각
    anchor_mode: AnchorMode = AnchorMode.OBS_POSE
    policy_id: int = 0  # config/chunk_policy.json 의 정책 선택
    dt_ns: int = 33_333_333  # 스텝 간격
    # (N, 6) 증분 delta: dx dy dz | rx ry rz
    steps: np.ndarray = field(default_factory=lambda: np.zeros((0, 6)))
    # (N,) EEF 동작 신호 0/1
    eef: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.uint8))
    version: int = PROTOCOL_VERSION
    # 수신 시각. 송신 측은 채우지 않고 제어 계층 수신 스레드가 기록한다.
    t_recv_ns: int = 0

    @property
    def n_steps(self) -> int:
        return int(self.steps.shape[0])

    @property
    def duration_ns(self) -> int:
        return self.n_steps * self.dt_ns


@dataclass
class CommitTrajectory:
    """'이 시각까지는 반드시 이대로 실행한다'는 약속 궤적.

    AI 계층은 이걸 보고 추론이 끝났을 때 로봇이 어디 있을지 알 수 있다.
    """

    t_start_ns: int = 0
    dt_ns: int = 10_000_000  # 10 ms 간격 샘플
    poses: np.ndarray = field(default_factory=lambda: np.zeros((0, 7)))  # (M, 7)

    @property
    def n(self) -> int:
        return int(self.poses.shape[0])

    @property
    def t_end_ns(self) -> int:
        return self.t_start_ns + max(0, self.n - 1) * self.dt_ns


@dataclass
class StateSnapshot:
    """제어 계층 -> AI 추론 계층.

    AI 는 이 스냅샷을 입력으로 받아 추론하고, 결과 청크에 snap_id 를 되돌려준다.
    그래야 제어 계층이 '이 청크가 무엇을 전제로 만들어졌는지' 알 수 있다.
    """

    snap_id: int = 0
    t_meas_ns: int = 0  # 측정 시각
    measured: np.ndarray = field(default_factory=lambda: np.zeros(7))  # EEF pose (7,)
    commit: CommitTrajectory = field(default_factory=CommitTrajectory)
    t_infer_hat_ns: int = 0  # 추정 추론 지연 (p95)
    jitter_ns: int = 0
    mode: ControlMode = ControlMode.IDLE
    version: int = PROTOCOL_VERSION

    @property
    def commit_end_ns(self) -> int:
        return self.commit.t_end_ns


# ============================================================================ chunk_codec.py
"""ActionChunk 바이너리 직렬화.

고정 크기 구조체로 직렬화한다.
- RT 쪽에서 파싱할 때 메모리 할당이 없도록
- 다른 PC 로 확장해도 그대로 쓰도록 (고정폭 타입 + 패딩 없음 + 리틀엔디안)

레이아웃 (C 의 #pragma pack(1) 과 동일)
    <I  version
    <I  struct_size
    <Q  seq_id
    <Q  snap_id
    <q  t_obs_ns
    <B  anchor_mode
    <H  policy_id
    <I  dt_ns
    <H  n_steps
    steps[MAX_STEPS] : 각 49 byte = 6*float64 + uint8
"""





_CHUNK_HEADER_FMT = "<IIQQqBHIH"
_CHUNK_HEADER_SIZE = struct.calcsize(_CHUNK_HEADER_FMT)  # 41

STEP_DTYPE = np.dtype([("d", "<f8", (6,)), ("eef", "u1")])  # packed, 49 byte
STEP_SIZE = STEP_DTYPE.itemsize

CHUNK_SIZE = _CHUNK_HEADER_SIZE + MAX_STEPS * STEP_SIZE




def encode_chunk(chunk: ActionChunk) -> bytes:
    n = chunk.n_steps
    if n > MAX_STEPS:
        raise ValueError(f"n_steps {n} > MAX_STEPS {MAX_STEPS}")
    body = np.zeros(MAX_STEPS, dtype=STEP_DTYPE)
    if n:
        body["d"][:n] = chunk.steps
        body["eef"][:n] = chunk.eef
    header = struct.pack(
        _CHUNK_HEADER_FMT,
        chunk.version,
        CHUNK_SIZE,
        chunk.seq_id,
        chunk.snap_id,
        chunk.t_obs_ns,
        int(chunk.anchor_mode),
        chunk.policy_id,
        chunk.dt_ns,
        n,
    )
    return header + body.tobytes()


def decode_chunk(buf: bytes) -> ActionChunk:
    if len(buf) != CHUNK_SIZE:
        raise DecodeError(f"size {len(buf)} != {CHUNK_SIZE}")
    (version, struct_size, seq_id, snap_id, t_obs_ns, anchor_mode, policy_id, dt_ns, n) = struct.unpack_from(
        _CHUNK_HEADER_FMT, buf, 0
    )
    if version != PROTOCOL_VERSION:
        raise DecodeError(f"protocol version {version} != {PROTOCOL_VERSION}")
    if struct_size != CHUNK_SIZE:
        raise DecodeError(f"struct_size {struct_size} != {CHUNK_SIZE}")
    if n > MAX_STEPS:
        raise DecodeError(f"n_steps {n} > {MAX_STEPS}")
    body = np.frombuffer(buf, dtype=STEP_DTYPE, count=MAX_STEPS, offset=_CHUNK_HEADER_SIZE)
    return ActionChunk(
        seq_id=seq_id,
        snap_id=snap_id,
        t_obs_ns=t_obs_ns,
        anchor_mode=AnchorMode(anchor_mode),
        policy_id=policy_id,
        dt_ns=dt_ns,
        steps=np.array(body["d"][:n], dtype=float),
        eef=np.array(body["eef"][:n], dtype=np.uint8),
        version=version,
    )


# ============================================================================ snapshot_codec.py
"""StateSnapshot 바이너리 직렬화 (제어 계층 -> AI 추론 계층).

레이아웃
    <I  version
    <I  struct_size
    <Q  snap_id
    <q  t_meas_ns
    <B  mode
    <q  commit_t_start_ns
    <I  commit_dt_ns
    <H  commit_n
    <q  t_infer_hat_ns
    <q  jitter_ns
    measured      : 7 * float64
    commit_poses  : MAX_COMMIT * 7 * float64
"""





_SNAP_HEADER_FMT = "<IIQqBqIHqq"
_SNAP_HEADER_SIZE = struct.calcsize(_SNAP_HEADER_FMT)

_POSE_BYTES = 7 * 8
SNAPSHOT_SIZE = _SNAP_HEADER_SIZE + _POSE_BYTES + MAX_COMMIT * _POSE_BYTES




def encode_snapshot(snap: StateSnapshot) -> bytes:
    m = snap.commit.n
    if m > MAX_COMMIT:
        raise ValueError(f"commit n {m} > MAX_COMMIT {MAX_COMMIT}")
    poses = np.zeros((MAX_COMMIT, 7), dtype="<f8")
    if m:
        poses[:m] = snap.commit.poses
    header = struct.pack(
        _SNAP_HEADER_FMT,
        snap.version,
        SNAPSHOT_SIZE,
        snap.snap_id,
        snap.t_meas_ns,
        int(snap.mode),
        snap.commit.t_start_ns,
        snap.commit.dt_ns,
        m,
        snap.t_infer_hat_ns,
        snap.jitter_ns,
    )
    return header + np.asarray(snap.measured, dtype="<f8").tobytes() + poses.tobytes()


def decode_snapshot(buf: bytes) -> StateSnapshot:
    if len(buf) != SNAPSHOT_SIZE:
        raise DecodeError(f"size {len(buf)} != {SNAPSHOT_SIZE}")
    (version, struct_size, snap_id, t_meas_ns, mode, c_start, c_dt, c_n, t_infer, jitter) = struct.unpack_from(
        _SNAP_HEADER_FMT, buf, 0
    )
    if version != PROTOCOL_VERSION:
        raise DecodeError(f"protocol version {version} != {PROTOCOL_VERSION}")
    if struct_size != SNAPSHOT_SIZE:
        raise DecodeError(f"struct_size {struct_size} != {SNAPSHOT_SIZE}")
    off = _SNAP_HEADER_SIZE
    measured = np.array(np.frombuffer(buf, dtype="<f8", count=7, offset=off), dtype=float)
    off += _POSE_BYTES
    poses = np.frombuffer(buf, dtype="<f8", count=MAX_COMMIT * 7, offset=off).reshape(MAX_COMMIT, 7)
    return StateSnapshot(
        snap_id=snap_id,
        t_meas_ns=t_meas_ns,
        measured=measured,
        commit=CommitTrajectory(t_start_ns=c_start, dt_ns=c_dt, poses=np.array(poses[:c_n], dtype=float)),
        t_infer_hat_ns=t_infer,
        jitter_ns=jitter,
        mode=ControlMode(mode),
        version=version,
    )
