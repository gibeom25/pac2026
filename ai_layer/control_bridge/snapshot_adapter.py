"""StateSnapshot(제어 -> AI) -> 모델 관측 state (9D) 와 anchor 선택.

제어 규약: measured / commit.poses 는 (7,) = [px py pz, qx qy qz qw] **월드 좌표계**, 쿼터니언 (x,y,z,w).
모델 규약: observation.state (9,) = [xyz, rot6d] (kinematics.pose_to_state 와 동일 정의).

anchor_mode 별 state 선택 (control/README "모델 능력별"):
    OBS_POSE   (L0) : state = 스냅샷의 측정 pose. 청크 스텝 0 은 t_obs 시점 측정 pose 대비.
    COMMIT_END (L1) : state = 약속 궤적의 끝 pose (추론이 끝났을 때 로봇이 있을 곳).
                      ACT 처럼 state 를 입력받는 모델은 이쪽이 건너뛰기를 거의 없앤다.
                      commit 이 비어 있으면 OBS_POSE 로 자동 강등.

좌표계 가정 (확인 필요): 학습 state 는 URDF root(base) 기준 FK 다. 제어의 world == base 라고 가정한다.
example_so101.json 의 base_frame 이 "world" 이므로 현재는 일치. 다르면 HAL 의 world<->base 변환을
여기서 역적용해야 한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as _R

from ai_layer.control_bridge.protocol import AnchorMode, StateSnapshot


def pose7_to_state9(pose7: np.ndarray) -> np.ndarray:
    """[px py pz qx qy qz qw] -> [xyz, R[:,0], R[:,1]]"""
    p = np.asarray(pose7, dtype=float).reshape(7)
    R = _R.from_quat(p[3:7]).as_matrix()  # scipy 는 (x,y,z,w) 순서 = 제어 규약과 동일
    return np.concatenate([p[:3], R[:, 0], R[:, 1]]).astype(np.float32)


def state9_to_pose7(state9: np.ndarray) -> np.ndarray:
    s = np.asarray(state9, dtype=float).reshape(9)
    a1, a2 = s[3:6], s[6:9]
    b1 = a1 / max(np.linalg.norm(a1), 1e-8)
    a2 = a2 - np.dot(b1, a2) * b1
    b2 = a2 / max(np.linalg.norm(a2), 1e-8)
    b3 = np.cross(b1, b2)
    q = _R.from_matrix(np.stack([b1, b2, b3], axis=1)).as_quat()
    return np.concatenate([s[:3], q])


@dataclass
class AnchorChoice:
    anchor_mode: AnchorMode
    state9: np.ndarray  # 모델 observation.state
    anchor_pose7: np.ndarray  # 청크 스텝 0 의 기준 pose (기록/디버그용)
    snap_id: int


def choose_anchor(snap: StateSnapshot, prefer: AnchorMode = AnchorMode.COMMIT_END) -> AnchorChoice:
    if prefer == AnchorMode.COMMIT_END and snap.commit.n > 0:
        pose7 = np.asarray(snap.commit.poses[snap.commit.n - 1], dtype=float)
        return AnchorChoice(AnchorMode.COMMIT_END, pose7_to_state9(pose7), pose7, int(snap.snap_id))
    pose7 = np.asarray(snap.measured, dtype=float)
    return AnchorChoice(AnchorMode.OBS_POSE, pose7_to_state9(pose7), pose7, int(snap.snap_id))
