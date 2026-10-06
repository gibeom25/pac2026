"""실로봇 연결 헬퍼 — SOFollower(lerobot 내장 드라이버) + IK를 한 번에 구성한다."""

from __future__ import annotations

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SOFollower

from ai_layer.kinematics import build_arm_kinematics
from ai_layer.tools.record_mujoco import CAMERA_HW


def build_real_robot(
    port: str,
    wrist_camera_index: int,
    overview_camera_index: int | None = None,
    max_relative_target: float | None = 5.0,
    calibrate: bool = False,
) -> tuple[SOFollower, RobotKinematics]:
    """SOFollower에 연결하고 IK 솔버를 같이 돌려준다.

    calibrate=False가 기본 — 매번 GUI에서 보정 프롬프트를 띄우지 않는다. **최초 1회는 터미널에서
    lerobot 공식 보정 도구를 먼저 돌려둘 것**:
        python -m lerobot.scripts.lerobot_calibrate --robot.type=so101_follower --robot.port=<port>
    보정 파일이 없는 채로 connect(calibrate=False)를 부르면 lerobot이 바로 에러를 낸다(여기서
    조용히 넘어가지 않음) — 그 경우 이 함수가 그 예외를 그대로 올린다(호출 측이 메시지로 안내).

    max_relative_target: 한 번의 send_action()당 관절이 움직일 수 있는 최대 각도(도) — lerobot
    자체 안전장치(SOFollower.send_action이 여기서 넘는 이동을 깎는다). 기본 5도로 보수적으로
    잡았다 — **이 코드는 실제 하드웨어로 검증하지 못했다**(이 환경엔 로봇이 없음). 처음 실제
    팔로 테스트할 땐 이보다 더 낮춰서(1~2도) 시작하고, 충분히 떨어져서 비상정지 가능한 상태로
    지켜볼 것.
    """
    cameras = {
        "wrist": OpenCVCameraConfig(
            index_or_path=wrist_camera_index, fps=30, width=CAMERA_HW[1], height=CAMERA_HW[0]
        ),
    }
    if overview_camera_index is not None:
        cameras["overview"] = OpenCVCameraConfig(
            index_or_path=overview_camera_index, fps=30, width=CAMERA_HW[1], height=CAMERA_HW[0]
        )

    config = SOFollowerRobotConfig(port=port, cameras=cameras, max_relative_target=max_relative_target)
    robot = SOFollower(config)
    robot.connect(calibrate=calibrate)

    kin = build_arm_kinematics()
    return robot, kin
