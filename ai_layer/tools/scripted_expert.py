"""정답 경로(ground truth)를 알고 있는 스크립트 전문가 — 사람 대신 EpisodeTicker를 조작한다.

2026-10-08: 시뮬레이션 시연은 "정답이 이미 정해진 일"이다 — 경로는 텍스처를 그릴 때 쓴 좌표
(`assets/so101/textures/*.json`, envs/seam_ground_truth.py), 높이는 HOVER_Z/MIN_TIP_Z 근처,
트리거는 경로 위에 있을 때만(dashed는 끊긴 구간도 연속 도포, branch는 메인 선만 — 둘 다
gen_seam_textures.py docstring의 규약). 그래서 사람이 조이스틱으로 따라 그리는 대신 이 모듈이
같은 일을 대량으로 한다. (2026-10-08: dashed는 기본적으로 선이 끊긴 구간에서 분사를 멈춘다 —
plan_episode(dashed="cut"). 예전 규약(끊긴 구간도 이어서 도포)은 dashed="bridge".) 정책 입력은 여전히 손목 카메라+state라서 정책은 선을 이미지로 찾아야
한다 — 정답을 아는 전문가가 시범을 보이고 카메라만 보는 학생이 따라 배우는 구조.

`ScriptedEEController`는 JoystickEEController/KeyboardEEController와 같은 공개 인터페이스
(poll/ee_velocity/rotation_rate/gripper_bit/episode_end_requested/discard_requested/close)를
구현한다 — EpisodeTicker 쪽은 손대지 않았으므로 기록되는 state/action/이미지/비드/높이제한이
사람 녹화와 완전히 같은 규약이다(action = 실측 도구 끝 pose 증분).

궤적은 에피소드 시작 시 한 번에 계획한다(`plan_episode`):
  1. 홈 -> 시작점 위 접근 높이로 이동 (이동 중 tilt로 서서히 기울임)
  2. 하강 -> 작업 높이(hover)
  3. 경로 추종 — 호 길이 기준 속도 프로파일(곡률 클수록 감속, 가감속 제한)
  4. 상승 후 잠깐 정지
EpisodeTicker는 ee_velocity()를 **현재 도구 자세 기준**(R_cmd) 속도로 해석하므로 월드 속도를
R_cmd.T로 돌려서 넘긴다. 매 프레임 "명령 목표(ticker.target_pos) -> 계획 목표" 차이로 속도를
내므로 외란(generate_demos.py의 disturbance)으로 밀려나도 그 자리에서 경로로 되돌아온다 — 이
복귀 구간이 데이터에 들어가야 BC가 경로를 벗어났을 때 돌아오는 법을 배운다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from lerobot.utils.rotation import Rotation

from ai_layer.envs.seam_ground_truth import SeamGroundTruth, dash_on_mask
from ai_layer.tools.record_mujoco import MIN_TIP_Z, ROD_HALF_LENGTH, _rod_tip_world

TIP_OFFSET_LOCAL = np.array([0.0, 0.0, -2 * ROD_HALF_LENGTH])  # ee_body 원점 -> 막대 끝 (로컬)
ROBOT_BASE_XY = np.array([0.0, 0.0])  # so101_new_calib.xml의 base body 위치 — A4 용지는 그 전방(+x)에 있다
PATH_DS = 0.0005  # 경로 재샘플 간격 [m]
GT_DENSE_POINTS = 600  # SeamGroundTruth 재샘플 점 수 — 기본 20점은 곡선을 따라가기엔 너무 거칠다

# 트리거 판정 — 도구 끝이 경로에서 이 거리 안이고 작업 높이 근처일 때만 분사. RL의
# off_seam_safety_dist(3cm)보다 훨씬 엄격하게 잡는다(사람도 선 위에서만 쏘니까).
TRIGGER_ON_XY_TOL = 0.0015
TRIGGER_OFF_XY_TOL = 0.003  # 히스테리시스 — 경계에서 깜빡이지 않게
TRIGGER_Z_TOL = 0.004


@dataclass
class ExpertParams:
    """에피소드마다 무작위로 뽑는 시연 스타일. generate_demos.py가 meta에 그대로 기록한다."""

    path_speed: float = 0.025  # 직선 구간 최대 속도 [m/s]
    hover_z: float = 0.01  # 경로 추종 중 도구 끝 높이 [m] — 실물 3D 펜은 약 1cm 띄워 그림
    approach_z: float = 0.05  # 시작점 위 접근 높이 [m]
    transit_speed: float = 0.05  # 접근/상승 이동 속도 [m/s]
    # 2026-10-08: 실물 3D 펜은 거의 수직이지만 위쪽이 로봇 쪽으로 미세하게 기울 수 있다 — 펜 축의
    # 위쪽이 로봇 베이스 방향으로 이 각도만큼 기운다. 5축 팔이라 도구가 늘 베이스를 향하므로
    # 기우는 방향은 도구 위치에 따라 프레임마다 바뀐다(plan_episode의 _tilt_toward_base).
    tilt_deg: float = 0.0
    start_offset: tuple[float, float] = (0.0, 0.0)  # 하강 지점의 경로 시작점 대비 xy 오프셋 [m]
    pause_frames: int = 10  # 시작 전/끝난 뒤 정지 프레임
    a_max: float = 0.08  # 접선 가감속 한계 [m/s^2]
    a_lat: float = 0.02  # 곡선에서 허용 횡가속 [m/s^2] — v <= sqrt(a_lat / 곡률)
    v_min: float = 0.006  # 급커브/코너 최저 속도 [m/s]
    n_disturb: int = 0  # 경로 추종 중 외란 횟수 — 실제 시점/크기는 generate_demos.py가 plan 이후 배치
    disturbances: list[dict] = field(default_factory=list)  # [{"frame": i, "offset": [x,y,z]}]


def sample_params(
    rng: np.random.Generator, max_tilt_deg: float, n_disturb_max: int, hover_range: tuple[float, float] = (0.009, 0.011)
) -> ExpertParams:
    start_r = rng.uniform(0.0, 0.004)
    start_a = rng.uniform(0.0, 2 * np.pi)
    return ExpertParams(
        path_speed=float(rng.uniform(0.018, 0.035)),
        hover_z=float(rng.uniform(*hover_range)),
        approach_z=float(rng.uniform(0.04, 0.07)),
        transit_speed=float(rng.uniform(0.04, 0.06)),
        tilt_deg=float(rng.uniform(0.0, max_tilt_deg)),
        start_offset=(float(start_r * np.cos(start_a)), float(start_r * np.sin(start_a))),
        pause_frames=int(rng.integers(5, 20)),
        a_lat=float(rng.uniform(0.015, 0.03)),
        n_disturb=int(rng.integers(0, n_disturb_max + 1)),
    )


def _resample(points: np.ndarray, ds: float) -> np.ndarray:
    seg = np.diff(points, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    keep = np.concatenate([[True], seg_len > 1e-9])
    points = points[keep]
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
    if cum[-1] < 1e-9:
        return points[:1]
    s = np.linspace(0.0, cum[-1], max(2, int(np.ceil(cum[-1] / ds)) + 1))
    return np.stack([np.interp(s, cum, points[:, k]) for k in range(points.shape[1])], axis=1)


def _xy_curvature(pts: np.ndarray, window: int = 10) -> np.ndarray:
    """등간격 점열의 점별 곡률 근사 [1/m] — ±window 점 사이 heading 변화 / 그 사이 거리."""
    n = len(pts)
    kappa = np.zeros(n)
    if n < 3:
        return kappa
    heading = np.arctan2(np.diff(pts[:, 1]), np.diff(pts[:, 0]))
    for k in range(n):
        a, b = max(0, k - window), min(n - 2, k + window - 1)
        if b <= a:
            continue
        dh = (heading[b] - heading[a] + np.pi) % (2 * np.pi) - np.pi
        kappa[k] = abs(dh) / max((b - a + 1) * PATH_DS, 1e-6)
    return kappa


def _time_parameterize(pts: np.ndarray, v_lim: np.ndarray, a_max: float, dt: float) -> np.ndarray:
    """등간격 점열 + 점별 속도 한계 -> 프레임 간격 dt로 샘플한 위치들. 양 끝 속도 0, 가감속 제한."""
    if len(pts) < 2:
        return pts[:1].copy()
    ds = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    v = v_lim.astype(float).copy()
    v[0] = 0.0
    for k in range(1, len(v)):
        v[k] = min(v[k], np.sqrt(v[k - 1] ** 2 + 2 * a_max * ds[k - 1]))
    v[-1] = 0.0
    for k in range(len(v) - 2, -1, -1):
        v[k] = min(v[k], np.sqrt(v[k + 1] ** 2 + 2 * a_max * ds[k]))
    v_mid = np.maximum((v[:-1] + v[1:]) / 2, 1e-4)
    t = np.concatenate([[0.0], np.cumsum(ds / v_mid)])
    s = np.concatenate([[0.0], np.cumsum(ds)])
    t_frames = np.arange(0.0, t[-1], dt)
    t_frames = np.append(t_frames, t[-1])
    s_frames = np.interp(t_frames, t, s)
    return np.stack([np.interp(s_frames, s, pts[:, k]) for k in range(pts.shape[1])], axis=1)


def _tilt_toward_base(tip_xy: np.ndarray, tilt_rad: float) -> np.ndarray:
    """펜 위쪽(로컬 +z)이 로봇 베이스 쪽으로 tilt_rad만큼 기운 자세 (T,3,3).

    r = 베이스 -> 도구 수평 단위벡터일 때 위쪽 축을 z에서 -r 쪽으로 돌리는 회전:
    회전축 = z x (-r) = (r_y, -r_x, 0). 예) r=(1,0)이면 y축 기준 -tilt -> 위쪽이 -x(로봇)로 기움.
    """
    r = tip_xy - ROBOT_BASE_XY[None]
    r /= np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-9)
    rotvec = np.stack([r[:, 1], -r[:, 0], np.zeros(len(r))], axis=1) * tilt_rad
    return np.stack([Rotation.from_rotvec(v).as_matrix() for v in rotvec])


@dataclass
class EpisodePlan:
    tip_ref: np.ndarray  # (T,3) 프레임별 목표 도구 끝 위치
    R_ref: np.ndarray  # (T,3,3) 프레임별 목표 자세
    follow_range: tuple[int, int]  # 경로 추종 구간 프레임 [start, end)
    path_xy: np.ndarray  # (M,2) 촘촘한 GT 경로 (트리거 판정/품질 평가용)
    path_tangent: np.ndarray  # (M,2) 단위 접선
    path_on: np.ndarray  # (M,) bool — 이 경로 점에서 분사해야 하는지 (점선의 끊긴 구간이면 False)


def plan_episode(
    gt: SeamGroundTruth, scene: str, variant: int, home_tip: np.ndarray, p: ExpertParams, dt: float,
    dashed: str = "cut", xy_transform=None,
) -> EpisodePlan:
    """xy_transform: GT 경로 xy(기존 시뮬 용지 기준)를 다른 장면 좌표로 옮기는 함수 (예: envs/piper_sim.py의
    seam_to_booth_xy). 강체 변환이어야 한다 — 점선 패턴이 호 길이 기준이라 그대로 유지된다."""
    polyline, _, _ = gt.load(scene, variant)
    xy = polyline[:, :2].astype(float)
    if xy_transform is not None:
        xy = xy_transform(xy)
    path_xy = _resample(xy, PATH_DS)
    tangent = np.gradient(path_xy, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
    path_on = dash_on_mask(path_xy, gt.dash_pattern(scene, variant) if dashed == "cut" else None)

    start_xy = path_xy[0] + np.asarray(p.start_offset)
    above = np.array([*start_xy, p.approach_z])
    down = np.array([*start_xy, p.hover_z])

    pause = np.repeat(home_tip[None], p.pause_frames, axis=0)
    approach = _resample(np.stack([home_tip, above]), PATH_DS)
    approach = _time_parameterize(approach, np.full(len(approach), p.transit_speed), p.a_max, dt)
    descend = _resample(np.stack([above, down]), PATH_DS)
    descend = _time_parameterize(descend, np.full(len(descend), p.transit_speed * 0.5), p.a_max, dt)

    # 시작 오프셋이 있으면 그 자리에서 경로 시작점으로 붙는 짧은 진입부가 추종 구간 앞에 붙는다
    follow_xy = _resample(np.concatenate([start_xy[None], path_xy]), PATH_DS)
    v_lim = np.clip(np.sqrt(p.a_lat / np.maximum(_xy_curvature(follow_xy), 1e-6)), p.v_min, p.path_speed)
    follow_xy = _time_parameterize(follow_xy, v_lim, p.a_max, dt)
    follow = np.concatenate([follow_xy, np.full((len(follow_xy), 1), p.hover_z)], axis=1)

    end = follow[-1]
    lift = _resample(np.stack([end, end + [0, 0, p.approach_z - p.hover_z]]), PATH_DS)
    lift = _time_parameterize(lift, np.full(len(lift), p.transit_speed), p.a_max, dt)
    tail = np.repeat(lift[-1:], p.pause_frames, axis=0)

    parts = [pause, approach, descend, follow, lift, tail]
    tip_ref = np.concatenate(parts)
    f0 = len(pause) + len(approach) + len(descend)
    follow_range = (f0, f0 + len(follow))

    # 자세: 홈(수직)에서 시작해 첫 정지 구간 동안 펜 기울기로 맞춘 뒤, 매 프레임 베이스 쪽으로 기울임 유지
    frac = np.clip(np.arange(len(tip_ref)) / max(1, len(pause)), 0.0, 1.0)
    R_ref = _tilt_toward_base(tip_ref[:, :2], np.deg2rad(p.tilt_deg))
    R_ref = np.stack([Rotation.from_rotvec(Rotation.from_matrix(R).as_rotvec() * f).as_matrix() for R, f in zip(R_ref, frac)])

    return EpisodePlan(
        tip_ref=tip_ref, R_ref=R_ref, follow_range=follow_range, path_xy=path_xy, path_tangent=tangent, path_on=path_on
    )


def nearest_on_path(path_xy: np.ndarray, xy: np.ndarray) -> tuple[int, float]:
    d = np.linalg.norm(path_xy - xy[None], axis=1)
    i = int(np.argmin(d))
    return i, float(d[i])


class ScriptedEEController:
    """EpisodeTicker에 사람 입력 대신 끼우는 컨트롤러. 프레임 인덱스는 poll()마다 1씩 증가한다
    (EpisodeTicker.tick()이 매 프레임 맨 처음에 poll()을 한 번 부른다)."""

    def __init__(self, ticker, plan: EpisodePlan):
        self.ticker = ticker
        self.plan = plan
        self.frame = -1
        self._trigger = False

    @property
    def done(self) -> bool:
        return self.frame >= len(self.plan.tip_ref) - 1

    def _ref(self) -> tuple[np.ndarray, np.ndarray]:
        i = min(max(self.frame, 0), len(self.plan.tip_ref) - 1)
        return self.plan.tip_ref[i], self.plan.R_ref[i]

    def poll(self) -> None:
        self.frame += 1
        self._update_trigger()

    def _update_trigger(self) -> None:
        f0, f1 = self.plan.follow_range
        if not (f0 <= self.frame < f1):
            self._trigger = False
            return
        tip = _rod_tip_world(self.ticker.data, self.ticker.rod_gid)
        i, d = nearest_on_path(self.plan.path_xy, tip[:2])
        if not self.plan.path_on[i]:  # 점선의 끊긴 구간 — 위치는 그대로 따라가되 분사만 멈춘다
            self._trigger = False
            return
        tip_ref, _ = self._ref()
        z_ok = abs(tip[2] - tip_ref[2]) < TRIGGER_Z_TOL and tip[2] >= MIN_TIP_Z - 1e-4
        tol = TRIGGER_OFF_XY_TOL if self._trigger else TRIGGER_ON_XY_TOL
        self._trigger = bool(z_ok and d < tol)

    def ee_velocity(self, max_linear: float = 0.05, invert_x: bool = False, invert_y: bool = False,
                    invert_z: bool = False) -> tuple[float, float, float]:
        tip_ref, R_ref = self._ref()
        target_des = tip_ref - R_ref @ TIP_OFFSET_LOCAL
        v_world = (target_des - self.ticker.target_pos) / self.ticker.dt
        speed = float(np.linalg.norm(v_world))
        if speed > max_linear:  # 외란 뒤 복귀할 때만 걸린다 — 계획 궤적 자체는 이보다 느리다
            v_world *= max_linear / speed
        v_local = self.ticker.R_cmd.T @ v_world  # EpisodeTicker가 R_cmd @ v로 다시 월드로 돌린다
        return float(v_local[0]), float(v_local[1]), float(v_local[2])

    def rotation_rate(self, max_angular: float = 1.0, invert_x: bool = False, invert_y: bool = False,
                      invert_z: bool = False) -> tuple[float, float, float]:
        _, R_ref = self._ref()
        w = Rotation.from_matrix(R_ref @ self.ticker.R_cmd.T).as_rotvec() / self.ticker.dt
        n = float(np.linalg.norm(w))
        if n > max_angular:
            w *= max_angular / n
        return float(w[0]), float(w[1]), float(w[2])

    def gripper_bit(self) -> float:
        return 1.0 if self._trigger else 0.0

    def episode_end_requested(self) -> bool:
        return False  # 종료는 generate_demos.py가 force_end로 직접 건다

    def discard_requested(self) -> bool:
        return False

    def close(self) -> None:
        pass
