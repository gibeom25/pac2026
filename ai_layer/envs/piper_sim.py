"""실물 Piper 부스를 본뜬 MuJoCo 시뮬레이션 (2026-10-09).

기존 시뮬(assets/so101/ee_rig.xml)은 로봇 팔 없이 도구만 떠다니는 리그라서 (1) 고정(overview)
카메라 화면을 만들 수 없고 (2) 좌표계/자세 분포가 실물 Piper 데이터와 달랐다(시뮬만 학습한 정책이
실물 시연에서 이동 방향을 108~124° 틀리게 예측, tools/eval_bc_offline.py). 이 모듈은 실물 데이터셋
ddyyuu/piper-user-20261008-flange-2cam-v1 과 같은 규약으로 동작한다:

  observation.state (9,)  = Piper 베이스 좌표계의 플랜지 pose [x,y,z, rot6d(회전행렬 앞 두 열)]
  action (7,)             = 직전 프레임 대비 플랜지 실측 증분 [dxyz(베이스), 월드 회전벡터, 트리거]
  observation.images.wrist / overview (240x320 RGB)

보정 근거 (실물 데이터셋에서 측정, 자세한 과정은 docs 없이 여기 숫자만 남긴다):
  - 로봇 모델: MuJoCo Menagerie agilex_piper (assets/piper, MIT). 데이터셋의 관절각으로 FK를 계산하면
    회전은 link6 프레임과 정확히 일치하고, 위치는 link6 + FLANGE_IN_LINK6 오프셋으로 평균 2.9mm 이내.
  - 플랜지 자세: 실물 시연 내내 거의 고정 — 플랜지 z축이 수직에서 14.5° 기울어 있다(TOOL_TILT_DEG).
  - 고정 카메라(D435i): 펜의 초록 LED를 고정 카메라 영상 619장에서 찾아 세션별 카메라 자세를 최소제곱으로
    맞춤(재투영 오차 중앙값 3.3px). 세션마다 위치 ±4cm, 방향 ±7° 정도 달라져서 에피소드마다 그 범위로 흔든다.
    LED가 플랜지 z축으로 0.145m(그리퍼 손가락 끝)에 있다고 가정했다.
  - 펜: 그리퍼에 플랜지 z축과 약 8° 어긋나게 물려 있다 — 같은 카메라로 영상 속 청록색 펜 몸통의 아래 끝을
    맞춰서 펜 축 방향(PEN_DIR_FLANGE)과 길이를 구함(오차 중앙값 4.2px). 펜 끝 = LED + 0.10m * 축.
    결과적으로 그릴 때 펜 끝 높이가 0~1cm — 로봇 베이스가 바닥(z=0)에 있다고 가정한 값이다.
    **펜 끝 위치(PEN_TIP_FLANGE)와 베이스 높이는 실측해서 고칠 것.**
  - 손목 카메라(D405): 실물 손목 영상과 렌더를 나란히 놓고 눈으로 맞춘 근사값.
  - 부스: 한 면(로봇 쪽, -x)이 열린 흰 직육면체, 반사되는 흰 바닥. 선(A4 텍스처)은 로봇 앞 x≈0.35에
    좌우(y) 방향으로 놓인다(실물 궤적 분포).

제어는 운동학적이다: 플랜지 목표 pose -> 감쇠 최소제곱 IK -> 관절각을 바로 대입(실물도 로컬 IK + 관절
명령). 동역학/충돌은 풀지 않고, 펜 끝이 바닥 아래로 내려가지 않게만 막는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from scipy.spatial.transform import Rotation

from ai_layer.envs.seam_ground_truth import TEXTURES_DIR, SeamGroundTruth
from ai_layer.tools.record_mujoco import BEAD_RGBA, BeadDrop, _draw_bead_trail

PIPER_XML = Path(__file__).resolve().parents[2] / "assets" / "piper" / "piper.xml"
CAMERA_HW = (240, 320)

FLANGE_IN_LINK6 = np.array([-0.0084, -0.0002, 0.0045])  # link6 원점 -> 실물 SDK 플랜지 (link6 축 기준)
PEN_LED_FLANGE = np.array([0.021, 0.016, 0.145])  # 플랜지 좌표계, 고정 카메라 보정에서 나온 값
PEN_DIR_FLANGE = np.array([0.121, -0.069, 0.990])  # 펜 축(위 -> 끝), 플랜지 좌표계
PEN_TIP_FLANGE = PEN_LED_FLANGE + 0.10 * PEN_DIR_FLANGE  # ≈ (0.033, 0.009, 0.244)
TOOL_TILT_DEG = 14.5  # 실물 시연의 펜 기울기 (위쪽이 로봇 쪽)
# 펜이 수직(끝이 아래)일 때 플랜지 자세 — 열: x=(-1,0,0) 로봇 쪽, y=(0,1,0), z=(0,0,-1) 펜 방향
FLANGE_R_VERTICAL = np.diag([-1.0, 1.0, -1.0])
HOME_FLANGE_POS = np.array([0.212, 0.003, 0.293])  # 실물 에피소드 시작 위치 평균 (std 2cm/1cm/0.5cm)
MIN_TIP_Z = 0.004
# IK 시작 관절각 — 실물 시연에서 흔한 자세(펜이 아래를 향하고 선 근처). menagerie "home" 키는 팔이
# 접혀 있고 플랜지가 앞을 향해 있어서 거기서 시작하면 수렴이 느리다.
Q_SEED = np.array([0.0, 1.5, -1.1, 0.0, 1.1, 0.0])

# 고정 카메라 공칭 자세 (세션 7개 중 프레임이 많은 세션들의 대표값)
OVERVIEW_POS = np.array([0.14, 0.365, 0.108])
OVERVIEW_FWD = np.array([0.56, -0.80, -0.20])
OVERVIEW_FOVY = 43.0
# 손목 카메라(D405): 실물 손목 영상에서 바닥에 비친 카메라 자신이 화면 가운데쯤 보이므로 거의 수직으로
# 내려다본다. 펜/손가락은 화면 아래, 로봇 반대쪽(+x)이 화면 위. 공칭 플랜지 자세(tilted_flange_R())에서의
# 월드 기준 값으로 적고 build_model()이 플랜지 좌표계로 바꾼다 (실물 영상과 렌더를 나란히 놓고 맞춘 근사값).
# 실물 손목 영상의 A4 용지(가로 297mm)가 화면 폭의 3/4쯤 -> 바닥에서 약 0.27m 높이, 카메라 바로 아래 점은
# 펜 끝보다 +x로 약 7.5cm.
WRIST_OFFSET_WORLD = np.array([0.10, 0.0, 0.02])  # 플랜지 -> 카메라
WRIST_FWD_WORLD = np.array([0.04, 0.0, -1.0])
WRIST_UP_WORLD = np.array([1.0, 0.0, 0.0])  # 화면 위쪽 방향
WRIST_FOVY = 58.0

PAPER_CENTER = np.array([0.35, 0.0])
PAPER_YAW_DEG = -90.0  # A4 긴 변(텍스처 가로)이 로봇 기준 좌->우(-y)로 가게
SO101_PAPER_CENTER = np.array([0.25, 0.0])  # seam_ground_truth.py가 쓰는 원래 용지 중심

BOOTH_X = (-0.02, 0.62)
BOOTH_Y = (-0.40, 0.40)
BOOTH_H = 0.55


def tilted_flange_R(tilt_deg: float = TOOL_TILT_DEG, yaw_deg: float = 0.0) -> np.ndarray:
    """펜 위쪽이 로봇(-x) 쪽으로 tilt_deg 기운 플랜지 자세 (실물: 월드 y축 기준 -tilt)."""
    R = Rotation.from_euler("zy", [yaw_deg, -tilt_deg], degrees=True).as_matrix()
    return R @ FLANGE_R_VERTICAL


def _look_quat(fwd: np.ndarray, up: np.ndarray) -> np.ndarray:
    """MuJoCo 카메라(-z가 시선, +y가 위) quat [w,x,y,z]."""
    f = fwd / np.linalg.norm(fwd)
    right = np.cross(f, up)
    right /= np.linalg.norm(right)
    up_c = np.cross(right, f)
    x, y, z, w = Rotation.from_matrix(np.stack([right, up_c, -f], axis=1)).as_quat()
    return np.array([w, x, y, z])


@dataclass
class BoothParams:
    """에피소드마다 흔드는 장면 요소. sample_booth()가 실물에서 본 범위로 뽑는다."""

    overview_pos: np.ndarray = field(default_factory=lambda: OVERVIEW_POS.copy())
    overview_fwd: np.ndarray = field(default_factory=lambda: OVERVIEW_FWD.copy())
    overview_fovy: float = OVERVIEW_FOVY
    paper_center: np.ndarray = field(default_factory=lambda: PAPER_CENTER.copy())
    paper_yaw_deg: float = PAPER_YAW_DEG
    light: float = 0.45  # 헤드라이트/조명 밝기
    floor_reflectance: float = 0.25
    floor_rgb: tuple[float, float, float] = (0.76, 0.77, 0.78)
    wall_rgb: tuple[float, float, float] = (0.80, 0.80, 0.79)
    paper_gray: float = 0.85  # 텍스처(거의 흰 종이)에 곱하는 밝기

    def to_json(self) -> dict:
        return {k: (v.round(4).tolist() if isinstance(v, np.ndarray) else v) for k, v in vars(self).items()}


def sample_booth(rng: np.random.Generator) -> BoothParams:
    yaw = np.deg2rad(rng.uniform(-8, 8))
    pitch = np.deg2rad(rng.uniform(-3, 3))
    f = OVERVIEW_FWD / np.linalg.norm(OVERVIEW_FWD)
    f = Rotation.from_euler("z", yaw).as_matrix() @ f
    f[2] += pitch
    gray = rng.uniform(0.70, 0.84)
    return BoothParams(
        overview_pos=OVERVIEW_POS + rng.uniform([-0.04, -0.03, -0.01], [0.04, 0.03, 0.01]),
        overview_fwd=f,
        overview_fovy=OVERVIEW_FOVY + rng.uniform(-2, 2),
        paper_center=PAPER_CENTER + rng.uniform([-0.03, -0.04], [0.03, 0.04]),
        paper_yaw_deg=PAPER_YAW_DEG + rng.uniform(-10, 10) + (180.0 if rng.random() < 0.5 else 0.0),
        light=float(rng.uniform(0.35, 0.6)),
        floor_reflectance=float(rng.uniform(0.1, 0.35)),
        floor_rgb=tuple(float(gray + rng.uniform(-0.02, 0.02)) for _ in range(3)),
        wall_rgb=tuple(float(min(1.0, gray + 0.04 + rng.uniform(-0.02, 0.02))) for _ in range(3)),
        paper_gray=float(min(1.0, gray + rng.uniform(0.0, 0.1))),
    )


def seam_texture_png(scene: str, variant: int) -> Path:
    name = scene if variant == 0 else f"{scene}_{variant}"
    return TEXTURES_DIR / f"a4_weld_seam_{name}.png"


def build_model(scene: str, variant: int, booth: BoothParams) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(PIPER_XML))
    spec.meshdir = str(PIPER_XML.parent / "assets")
    for light in list(spec.lights):
        spec.delete(light)
    spec.visual.headlight.diffuse = [booth.light] * 3
    spec.visual.headlight.ambient = [0.35] * 3
    spec.visual.headlight.specular = [0.0] * 3
    spec.visual.quality.shadowsize = 2048

    tex = spec.add_texture(name="seam", type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(seam_texture_png(scene, variant)))
    paper = spec.add_material(name="paper", rgba=[booth.paper_gray] * 3 + [1], specular=0.1, shininess=0.3,
                              reflectance=booth.floor_reflectance)
    paper.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex.name
    floor_mat = spec.add_material(name="booth_floor", rgba=[*booth.floor_rgb, 1], specular=0.3, shininess=0.6,
                                  reflectance=booth.floor_reflectance)
    wall_mat = spec.add_material(name="booth_wall", rgba=[*booth.wall_rgb, 1], specular=0.05, shininess=0.1)

    wb = spec.worldbody
    cx, cy = (BOOTH_X[0] + BOOTH_X[1]) / 2, (BOOTH_Y[0] + BOOTH_Y[1]) / 2
    hx, hy = (BOOTH_X[1] - BOOTH_X[0]) / 2, (BOOTH_Y[1] - BOOTH_Y[0]) / 2
    nogeom = dict(contype=0, conaffinity=0)
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[hx + 0.3, hy + 0.3, 0.01],
                pos=[cx, cy, 0], material="booth_floor", **nogeom)
    for name, pos, size in [
        ("wall_back", [BOOTH_X[1], cy, BOOTH_H / 2], [0.005, hy, BOOTH_H / 2]),
        ("wall_left", [cx, BOOTH_Y[1], BOOTH_H / 2], [hx, 0.005, BOOTH_H / 2]),
        ("wall_right", [cx, BOOTH_Y[0], BOOTH_H / 2], [hx, 0.005, BOOTH_H / 2]),
        ("ceiling", [cx, cy, BOOTH_H], [hx, hy, 0.005]),
    ]:
        wb.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos, size=size, material="booth_wall", **nogeom)
    wb.add_light(name="booth_light", pos=[cx, cy, BOOTH_H - 0.02], dir=[0, 0, -1],
                 diffuse=[booth.light * 0.6] * 3, specular=[0.1] * 3, castshadow=0, cutoff=80, exponent=1)
    yaw = np.deg2rad(booth.paper_yaw_deg)
    wb.add_geom(name="a4_paper", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0.1485, 0.105, 0.001],
                pos=[*booth.paper_center, 0.0005], quat=[np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)], material="paper", **nogeom)
    wb.add_camera(name="overview", pos=booth.overview_pos, quat=_look_quat(booth.overview_fwd, np.array([0, 0, 1.0])),
                  fovy=booth.overview_fovy)

    link6 = spec.body("link6")
    o = FLANGE_IN_LINK6  # 플랜지와 link6은 축이 같고 원점만 다르다
    link6.add_site(name="flange", pos=o, size=[0.003, 0, 0], rgba=[1, 0, 0, 0], group=4)
    pen = dict(contype=0, conaffinity=0, group=1)
    # 펜: 그리퍼 사이에 물린 청록 몸통 + 아래쪽 검은 노즐 (실물 3D 펜 모양 근사)
    d = PEN_DIR_FLANGE / np.linalg.norm(PEN_DIR_FLANGE)
    tip = o + PEN_TIP_FLANGE
    top, body_end = tip - 0.17 * d, tip - 0.014 * d
    link6.add_geom(name="pen_body", type=mujoco.mjtGeom.mjGEOM_CYLINDER, fromto=[*top, *body_end], size=[0.0095, 0, 0],
                   rgba=[0.25, 0.78, 0.8, 1], **pen)
    link6.add_geom(name="pen_nozzle", type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[*body_end, *(tip - 0.002 * d)],
                   size=[0.0035, 0, 0], rgba=[0.12, 0.12, 0.13, 1], **pen)
    link6.add_geom(name="pen_led", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.003, 0, 0],
                   pos=o + PEN_LED_FLANGE, rgba=[0.3, 1.0, 0.4, 1], **pen)
    link6.add_site(name="pen_tip", pos=tip, size=[0.002, 0, 0], rgba=[1, 0, 0, 0], group=4)
    Rn = tilted_flange_R()
    link6.add_camera(name="wrist", pos=o + Rn.T @ WRIST_OFFSET_WORLD,
                     quat=_look_quat(Rn.T @ WRIST_FWD_WORLD, Rn.T @ WRIST_UP_WORLD), fovy=WRIST_FOVY)
    model = spec.compile()
    # 실물 그리퍼(link6~8)는 검정 — menagerie 모델은 회색이라 색만 바꾼다
    dark = {model.body(n).id for n in ("link6", "link7", "link8")}
    pen_geoms = {model.geom(n).id for n in ("pen_body", "pen_nozzle", "pen_led")}
    for g in range(model.ngeom):
        if model.geom_bodyid[g] in dark and g not in pen_geoms and model.geom_group[g] == 2:
            model.geom_matid[g] = -1
            model.geom_rgba[g] = [0.07, 0.07, 0.08, 1]
    return model


def seam_to_booth_xy(xy_so101: np.ndarray, booth: BoothParams) -> np.ndarray:
    """seam_ground_truth.py가 주는 (기존 시뮬 용지 기준) 경로 xy -> 이 부스의 월드 xy."""
    yaw = np.deg2rad(booth.paper_yaw_deg)
    R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    return (np.asarray(xy_so101) - SO101_PAPER_CENTER) @ R.T + booth.paper_center


class PiperSim:
    """한 에피소드(장면 1개)짜리 시뮬레이터. 플랜지 목표 pose를 받아 IK로 관절을 맞추고 카메라를 렌더한다."""

    def __init__(self, scene: str, variant: int, booth: BoothParams | None = None, render: bool = True):
        self.scene, self.variant = scene, variant
        self.booth = booth or BoothParams()
        self.model = build_model(scene, variant, self.booth)
        self.data = mujoco.MjData(self.model)
        self.site = self.model.site("flange").id
        self.tip_site = self.model.site("pen_tip").id
        self.qadr = np.array([self.model.joint(f"joint{i}").qposadr[0] for i in range(1, 7)])
        self.dadr = np.array([self.model.joint(f"joint{i}").dofadr[0] for i in range(1, 7)])
        self.qrange = self.model.jnt_range[[self.model.joint(f"joint{i}").id for i in range(1, 7)]]
        self.renderer = mujoco.Renderer(self.model, height=CAMERA_HW[0], width=CAMERA_HW[1]) if render else None
        self.bead_points: list[BeadDrop] = []
        self.bead_rgba = BEAD_RGBA
        key = self.model.key("home").id
        mujoco.mj_resetDataKeyframe(self.model, self.data, key)
        self.data.qpos[self.qadr] = Q_SEED
        self._set_fingers(0.0095)
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)

    def _set_fingers(self, half_open: float) -> None:
        self.data.qpos[self.model.joint("joint7").qposadr[0]] = half_open
        self.data.qpos[self.model.joint("joint8").qposadr[0]] = -half_open

    # ---- 자세 ----
    def flange_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self.data.site_xpos[self.site].copy(), self.data.site_xmat[self.site].reshape(3, 3).copy()

    def tip_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.tip_site].copy()

    def flange_pose_xyzrotvec(self) -> np.ndarray:
        p, R = self.flange_pose()
        return np.concatenate([p, Rotation.from_matrix(R).as_rotvec()])

    def joint_positions(self) -> np.ndarray:
        return self.data.qpos[self.qadr].copy()

    def solve_ik(self, pos: np.ndarray, R: np.ndarray, iters: int = 100, tol: float = 1e-5) -> float:
        """플랜지를 (pos, R)로 — 현재 관절에서 시작하는 감쇠 최소제곱. 남은 위치 오차 [m] 반환."""
        m, d = self.model, self.data
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        lam2 = 1e-4
        err_p = np.inf
        for _ in range(iters):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            cur_p, cur_R = self.flange_pose()
            err_p_vec = pos - cur_p
            err_r = Rotation.from_matrix(R @ cur_R.T).as_rotvec()
            err_p = float(np.linalg.norm(err_p_vec))
            if err_p < tol and np.linalg.norm(err_r) < 1e-4:
                break
            mujoco.mj_jacSite(m, d, jacp, jacr, self.site)
            J = np.vstack([jacp[:, self.dadr], jacr[:, self.dadr]])
            e = np.concatenate([err_p_vec, err_r])
            dq = J.T @ np.linalg.solve(J @ J.T + lam2 * np.eye(6), e)
            q = d.qpos[self.qadr] + np.clip(dq, -0.3, 0.3)
            d.qpos[self.qadr] = np.clip(q, self.qrange[:, 0], self.qrange[:, 1])
        mujoco.mj_kinematics(m, d)
        mujoco.mj_camlight(m, d)  # 카메라/조명 pose는 mj_kinematics가 아니라 여기서 갱신된다
        return err_p

    def move_flange(self, pos: np.ndarray, R: np.ndarray) -> np.ndarray:
        """목표 플랜지 pose로 이동(펜 끝이 MIN_TIP_Z 아래로 내려가지 않게 위로 보정). 실제 도달 pose (6,) 반환."""
        pos = np.asarray(pos, dtype=float).copy()
        tip_z = pos[2] + (R @ PEN_TIP_FLANGE)[2]
        if tip_z < MIN_TIP_Z:
            pos[2] += MIN_TIP_Z - tip_z
        self.solve_ik(pos, R)
        return self.flange_pose_xyzrotvec()

    # ---- 비드/렌더 ----
    def step_beads(self, trigger: bool, dt: float) -> None:
        if trigger:
            self.bead_points.append(BeadDrop(self.tip_pos()))
        for b in self.bead_points:
            b.step(dt)

    def render(self, camera: str) -> np.ndarray:
        self.renderer.update_scene(self.data, camera=camera)
        _draw_bead_trail(self.renderer.scene, self.bead_points, self.bead_rgba)
        return self.renderer.render()

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()


def seam_path_booth(gt: SeamGroundTruth, scene: str, variant: int, booth: BoothParams) -> np.ndarray:
    polyline, _, _ = gt.load(scene, variant)
    return seam_to_booth_xy(polyline[:, :2], booth)
