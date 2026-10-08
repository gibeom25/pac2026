"""데이터 수집 탭 — 카메라 뷰(손목/오버뷰) + 로봇 상태 + 수집 현황 차트 + 세션/에피소드 제어.

record_mujoco.py/record_gui.py(Dear PyGui)와 같은 물리/데이터셋/balanced 샘플링 로직을 그대로
쓴다(ai_layer/tools/episode_ticker.py의 EpisodeTicker/MujocoDualCamera, record_mujoco.py의
BalancedSceneSampler/_build_combos/_mjcf_path). GUI 프레임마다 QTimer가 tick() 한 번씩 부른다
— 이 구조는 record_gui.py의 while 루프와 동일하고, 여기서 새로 하는 건 Qt 위젯 배치/갱신뿐이다.

2026-10-06: "조작을 테스트한 후에 녹화를 시작"할 수 있게 조작 테스트 모드를 추가했다 —
check_ee.py와 같은 "저장 없이 자유롭게 움직여보기"를 GUI 안에 넣은 것. 세션과 달리 설정(속도/
반전)을 켜둔 채로 실시간으로 바꿔가며 체감해볼 수 있고, 만족스러우면 컨트롤러를 새로 만들지
않고 그대로 들고 녹화 세션으로 넘어간다(이미 확인한 입력 장치를 재연결할 필요 없음).
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import mujoco
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ai_layer.gui.camera_view import CameraView
from ai_layer.gui.dataset_prep import DatasetCancelled, build_dataset_gui
from ai_layer.gui.qt_keyboard_input import QtKeyboardEEController
from ai_layer.real.connection import build_real_robot
from ai_layer.real.real_robot_ticker import RealRobotEpisodeTicker
from ai_layer.tools.episode_ticker import EpisodeTicker, MujocoDualCamera
from ai_layer.tools.joystick_input import JoystickEEController
from ai_layer.tools.record_mujoco import (
    MAX_ANGULAR_SPEED_DEFAULT,
    N_VARIANTS,
    SCENE_VARIANTS,
    BalancedSceneSampler,
    _build_combos,
    _mjcf_path,
)
from ai_layer.tools.teleop_input import resolve_input_mode

SOURCE_SIM = "시뮬레이션"
SOURCE_REAL = "실로봇"


class CollectTab(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        self._ctl = None
        self._keyboard_filter_installed = False
        self._robot = None  # 실로봇(SOFollower) — 연결되면 테스트/세션 전환 동안 재사용
        self._kin = None
        self._dataset = None
        self._sampler: BalancedSceneSampler | None = None
        self._combos: list[tuple[str, int]] = []
        self._args: argparse.Namespace | None = None
        self._ticker: EpisodeTicker | None = None
        self._saved_count = 0
        self._force_end = False
        self._force_discard = False
        self._test_mode = False

        self._build_ui()
        self._on_source_changed(self.source_combo.currentText())  # 초기 표시 상태 맞춤(기본=시뮬레이션)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

    # ---- UI 구성 ----------------------------------------------------------
    def _build_ui(self) -> None:
        root_layout = QHBoxLayout(self)

        # 2026-10-06: 왼쪽 열(설정+세션/에피소드+로봇상태+차트)을 그냥 쌓으면 각 행의 최소
        # 높이가 다 더해져서 "창 자체의 최소 높이"가 돼버려 작은 화면에서 창을 못 줄이는
        # 문제가 있었다(실측: 최소 높이 1048px로 고정됨) — QScrollArea로 감싸서 왼쪽 열의
        # 내용 높이가 창 전체의 최소 크기를 강제하지 않게 하고, 공간이 모자라면 왼쪽만
        # 스크롤되게 한다.
        left_container = QWidget()
        left_col = QVBoxLayout(left_container)
        left_col.addWidget(self._build_settings_group())
        left_col.addWidget(self._build_real_robot_group())
        left_col.addWidget(self._build_session_group())
        left_col.addWidget(self._build_state_group())
        # 카메라 밑에 있던 수집 현황 그래프를 왼쪽 열 맨 아래로 옮김 — 오른쪽은 카메라 전용
        # 공간으로 비워둔다.
        self._chart = pg.PlotWidget(title="형태별 수집 현황 (저장된 에피소드 수)")
        self._chart.setBackground("w")
        self._chart.setFixedHeight(220)
        left_col.addWidget(self._chart)
        left_col.addStretch(1)

        left_scroll = QScrollArea()
        left_scroll.setWidget(left_container)
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        left_scroll.setMinimumWidth(360)
        # 2026-10-06: 남는 가로 공간을 전부 카메라 쪽(stretch=1)에 몰아주던 걸 바꿨다 — 카메라는
        # 이제 원본 비율을 유지한 채로만 커지므로(camera_view.py의 _AspectRatioLabel) 창을 넓혀도
        # 화면이 늘어나 보이지 않고 그냥 레터박스만 커졌었는데, "차라리 좌측을 늘려"라는 요청대로
        # 남는 공간은 왼쪽 설정 패널이 가져가게 둘 다 stretch=1로 맞췄다(같이 커짐).
        root_layout.addWidget(left_scroll, stretch=1)
        self._left_scroll = left_scroll  # repo-id 등 검증 실패 시 해당 필드로 스크롤하는 데 씀

        right_col = QVBoxLayout()
        root_layout.addLayout(right_col, stretch=1)
        # 카메라 두 대를 가로로 나란히(좁게) 두지 않고 세로로 쌓아서 오른쪽 열 공간을 꽉
        # 채운다 — CameraView 자체가 채워주는 영역만큼 늘어난다(camera_view.py의 Expanding
        # 정책 참고).
        self._wrist_view = CameraView("손목 카메라 (데이터셋 저장 화면)")
        self._overview_view = CameraView("전체(오버뷰) 카메라")
        right_col.addWidget(self._wrist_view, stretch=1)
        right_col.addWidget(self._overview_view, stretch=1)

        self._status_label = QLabel("대기 중 — 설정을 확인하고 조작 테스트나 세션 시작을 누르세요.")
        right_col.addWidget(self._status_label)

    def _build_settings_group(self) -> QGroupBox:
        box = QGroupBox("설정 (조작 테스트/세션 중엔 속도·반전만 실시간 반영)")
        form = QFormLayout(box)

        # 2026-10-06: 시뮬/실로봇 전환 — EE 컨트롤러(조이스틱/키보드)는 그대로 재사용하고
        # (ee_velocity()만 주면 되는 하드웨어 중립 인터페이스), 백엔드만 MuJoCo 틱(EpisodeTicker)
        # 또는 실로봇 틱(ai_layer/real/real_robot_ticker.py의 RealRobotEpisodeTicker)으로 바뀐다
        # — 데이터셋 스키마는 완전히 동일해서 BC/RL 학습 코드는 안 바뀐다. "scene/variant"는
        # 시뮬 전용(아래 실로봇 연결 섹션이 실로봇 전용)이고, 선택 안 한 쪽 설정은 그냥 무시된다.
        self.source_combo = QComboBox()
        self.source_combo.addItems([SOURCE_SIM, SOURCE_REAL])
        self.source_combo.currentTextChanged.connect(self._on_source_changed)
        form.addRow("소스", self.source_combo)

        self.repo_id_edit = QLineEdit()
        self.repo_id_edit.setPlaceholderText("예: my-user/so101-weld-demo (dry-run이거나 root만 채워도 됨)")
        form.addRow("repo-id", self.repo_id_edit)

        self.root_edit = QLineEdit()
        self.root_edit.setPlaceholderText("저장 경로. 비우면 datasets/<repo-id> — root만 채워도 repo-id는 폴더명으로 자동 채움")
        form.addRow("root", self.root_edit)

        self.dry_run_check = QCheckBox("dry-run (저장 안 함, 연습용)")
        form.addRow("", self.dry_run_check)

        self.scene_combo = QComboBox()
        self.scene_combo.addItems(["balanced", *SCENE_VARIANTS])
        form.addRow("scene", self.scene_combo)

        self.variant_spin = QSpinBox()
        self.variant_spin.setRange(-1, N_VARIANTS - 1)
        self.variant_spin.setValue(-1)
        self.variant_spin.setToolTip("-1=무작위/균형 샘플링")
        form.addRow("variant", self.variant_spin)

        self.num_episodes_spin = QSpinBox()
        self.num_episodes_spin.setRange(1, 1000)
        self.num_episodes_spin.setValue(30)
        form.addRow("num-episodes", self.num_episodes_spin)

        self.episode_seconds_spin = QDoubleSpinBox()
        self.episode_seconds_spin.setRange(0.0, 600.0)
        self.episode_seconds_spin.setValue(0.0)
        self.episode_seconds_spin.setSuffix(" s (0=무제한)")
        form.addRow("episode-seconds", self.episode_seconds_spin)

        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(5, 60)
        self.fps_spin.setValue(30)
        form.addRow("fps", self.fps_spin)

        self.max_linear_spin = QDoubleSpinBox()
        self.max_linear_spin.setRange(0.001, 1.0)
        self.max_linear_spin.setSingleStep(0.01)
        self.max_linear_spin.setValue(0.05)
        self.max_linear_spin.setSuffix(" m/s")
        form.addRow("max-linear-speed", self.max_linear_spin)

        self.max_angular_spin = QDoubleSpinBox()
        self.max_angular_spin.setRange(0.1, 5.0)
        self.max_angular_spin.setSingleStep(0.1)
        self.max_angular_spin.setValue(MAX_ANGULAR_SPEED_DEFAULT)
        self.max_angular_spin.setSuffix(" rad/s")
        form.addRow("max-angular-speed", self.max_angular_spin)

        invert_row = QHBoxLayout()
        self.invert_x_check = QCheckBox("x")
        self.invert_y_check = QCheckBox("y")
        self.invert_z_check = QCheckBox("z")
        self.invert_roll_check = QCheckBox("roll")
        self.invert_pitch_check = QCheckBox("pitch")
        for cb in (self.invert_x_check, self.invert_y_check, self.invert_z_check, self.invert_roll_check, self.invert_pitch_check):
            invert_row.addWidget(cb)
        form.addRow("반전", invert_row)

        self.input_combo = QComboBox()
        self.input_combo.addItems(["auto", "joystick", "keyboard"])
        form.addRow("input", self.input_combo)

        self.recalibrate_check = QCheckBox("조이스틱 스로틀 재보정 (처음 켤 때만 체크 — 터미널에 뜨는 보정 안내도 봐야 함)")
        form.addRow("", self.recalibrate_check)

        self._settings_form = form  # _on_source_changed()가 scene/variant 행을 숨기는 데 씀
        return box

    def _build_real_robot_group(self) -> QGroupBox:
        """소스=실로봇일 때만 쓰는 연결 설정 — lerobot의 SOFollower(SO-101 팔로워 드라이버)로
        직접 연결한다. scene/variant는 실로봇에선 의미가 없어(물리적 리셋이 없음) 무시된다."""
        box = QGroupBox("실로봇 연결 (소스=실로봇일 때만 사용)")
        form = QFormLayout(box)

        self.real_port_edit = QLineEdit()
        self.real_port_edit.setPlaceholderText("예: /dev/ttyACM0")
        form.addRow("port", self.real_port_edit)

        self.real_wrist_cam_spin = QSpinBox()
        self.real_wrist_cam_spin.setRange(0, 20)
        form.addRow("손목 카메라 index", self.real_wrist_cam_spin)

        self.real_overview_cam_spin = QSpinBox()
        self.real_overview_cam_spin.setRange(-1, 20)
        self.real_overview_cam_spin.setValue(-1)
        self.real_overview_cam_spin.setToolTip("-1=오버뷰 카메라 없음(손목 화면을 그대로 복사해서 보여줌)")
        form.addRow("오버뷰 카메라 index (-1=없음)", self.real_overview_cam_spin)

        self.real_max_rel_target_spin = QDoubleSpinBox()
        self.real_max_rel_target_spin.setRange(0.1, 90.0)
        self.real_max_rel_target_spin.setSingleStep(0.5)
        self.real_max_rel_target_spin.setValue(5.0)
        self.real_max_rel_target_spin.setSuffix(" deg/tick")
        self.real_max_rel_target_spin.setToolTip(
            "한 번에 보낼 수 있는 관절 이동량 상한(안전장치, lerobot 내장) — 처음 실제 하드웨어로 "
            "테스트할 땐 더 낮춰서(1~2도) 시작할 것."
        )
        form.addRow("max-relative-target", self.real_max_rel_target_spin)

        self.real_calibrate_check = QCheckBox(
            "연결 시 재보정 (체크하면 터미널에 보정 안내가 뜰 수 있음 — 보통은 꺼두고 "
            "lerobot_calibrate를 미리 한 번 따로 돌려둘 것)"
        )
        form.addRow("", self.real_calibrate_check)

        self._real_robot_group = box  # _on_source_changed()가 소스=시뮬레이션일 때 통째로 숨김
        return box

    def _build_session_group(self) -> QGroupBox:
        box = QGroupBox("조작 테스트 / 세션 / 에피소드")
        layout = QVBoxLayout(box)

        test_row = QHBoxLayout()
        self.start_test_btn = QPushButton("조작 테스트 (저장 안 함)")
        self.stop_test_btn = QPushButton("테스트 중지")
        self.test_to_record_btn = QPushButton("테스트 통과 → 바로 녹화 시작")
        self.stop_test_btn.setEnabled(False)
        self.test_to_record_btn.setEnabled(False)
        self.start_test_btn.clicked.connect(self._on_start_test)
        self.stop_test_btn.clicked.connect(self._on_stop_test)
        self.test_to_record_btn.clicked.connect(self._on_test_to_record)
        test_row.addWidget(self.start_test_btn)
        test_row.addWidget(self.stop_test_btn)
        test_row.addWidget(self.test_to_record_btn)
        layout.addLayout(test_row)

        session_row = QHBoxLayout()
        self.start_session_btn = QPushButton("세션 시작")
        self.end_session_btn = QPushButton("세션 종료")
        self.end_session_btn.setEnabled(False)
        self.start_session_btn.clicked.connect(self._on_start_session)
        self.end_session_btn.clicked.connect(self._on_end_session)
        session_row.addWidget(self.start_session_btn)
        session_row.addWidget(self.end_session_btn)
        layout.addLayout(session_row)

        episode_row = QHBoxLayout()
        self.start_episode_btn = QPushButton("에피소드 시작")
        self.save_btn = QPushButton("저장+종료 (BTN_THUMB / ENTER)")
        self.discard_btn = QPushButton("폐기+재시도 (BTN_THUMB2 / BACKSPACE)")
        for btn in (self.start_episode_btn, self.save_btn, self.discard_btn):
            btn.setEnabled(False)
        self.start_episode_btn.clicked.connect(self._on_start_episode)
        self.save_btn.clicked.connect(self._on_request_save)
        self.discard_btn.clicked.connect(self._on_request_discard)
        episode_row.addWidget(self.start_episode_btn)
        episode_row.addWidget(self.save_btn)
        episode_row.addWidget(self.discard_btn)
        layout.addLayout(episode_row)

        return box

    def _build_state_group(self) -> QGroupBox:
        box = QGroupBox("로봇 상태")
        form = QFormLayout(box)
        self.state_scene_label = QLabel("-")
        self.state_time_label = QLabel("-")
        self.state_pos_label = QLabel("-")
        self.state_tip_label = QLabel("-")
        self.state_trigger_label = QLabel("-")
        self.state_contact_label = QLabel("-")
        self.state_bead_label = QLabel("-")
        self.state_progress_label = QLabel("-")
        form.addRow("씬/variant", self.state_scene_label)
        form.addRow("경과 시간", self.state_time_label)
        form.addRow("EE 목표 위치", self.state_pos_label)
        form.addRow("막대 끝(tip) z", self.state_tip_label)
        form.addRow("트리거", self.state_trigger_label)
        form.addRow("접촉", self.state_contact_label)
        form.addRow("비드 점수", self.state_bead_label)
        form.addRow("진행", self.state_progress_label)
        return box

    # ---- 공통 ---------------------------------------------------------
    def _on_source_changed(self, source: str) -> None:
        """소스=시뮬레이션/실로봇에 따라 안 쓰는 설정을 숨긴다(2026-10-08 정리 — 이전엔 둘 다
        항상 보여서 "scene/variant가 실로봇에선 뭘 하는 거지" 같은 혼란이 있었다). 값 자체는
        안 지운다 — 다시 시뮬레이션으로 돌아가면 scene/variant가 그대로 남아있다."""
        is_real = source == SOURCE_REAL
        self._real_robot_group.setVisible(is_real)
        self._settings_form.setRowVisible(self.scene_combo, not is_real)
        self._settings_form.setRowVisible(self.variant_spin, not is_real)

    def _collect_args(self) -> argparse.Namespace:
        repo_id = self.repo_id_edit.text().strip() or None
        root = self.root_edit.text().strip() or None
        if repo_id is None and root is not None:
            # 2026-10-06: repo-id를 안 채우고 root만 채운 채로 "repo-id 필요" 에러를 보고
            # 헷갈렸다는 피드백 — root가 있으면 그 폴더 이름으로 repo-id를 그냥 만들어준다
            # (repo_id는 로컬 저장 시 사실상 이름표일 뿐이라 root 폴더명을 그대로 써도 무방).
            repo_id = Path(root).name or root
        return argparse.Namespace(
            source=self.source_combo.currentText(),
            repo_id=repo_id,
            root=root,
            dry_run=self.dry_run_check.isChecked(),
            fps=self.fps_spin.value(),
            num_episodes=self.num_episodes_spin.value(),
            episode_seconds=self.episode_seconds_spin.value() or None,
            task="weld seam following demo (ee-native, joystick, qt-gui)",
            scene=self.scene_combo.currentText(),
            variant=self.variant_spin.value(),
            max_linear_speed=self.max_linear_spin.value(),
            max_angular_speed=self.max_angular_spin.value(),
            invert_x=self.invert_x_check.isChecked(),
            invert_y=self.invert_y_check.isChecked(),
            invert_z=self.invert_z_check.isChecked(),
            invert_roll=self.invert_roll_check.isChecked(),
            invert_pitch=self.invert_pitch_check.isChecked(),
            input=self.input_combo.currentText(),
            recalibrate_joystick=self.recalibrate_check.isChecked(),
            real_port=self.real_port_edit.text().strip(),
            real_wrist_cam=self.real_wrist_cam_spin.value(),
            real_overview_cam=(self.real_overview_cam_spin.value() if self.real_overview_cam_spin.value() >= 0 else None),
            real_max_rel_target=self.real_max_rel_target_spin.value(),
            real_calibrate=self.real_calibrate_check.isChecked(),
        )

    def _apply_live_feel_settings(self) -> None:
        """조작 테스트 중엔 씬을 다시 로드할 필요 없는 설정(속도/반전)만 매 프레임 반영한다 —
        scene/num-episodes 등은 모델 재로드가 필요해서 테스트 중 바꿔도 다음 테스트 에피소드가
        새로 시작될 때만 적용됨(_start_test_episode 참고)."""
        if self._ticker is None:
            return
        a = self._ticker.args
        a.max_linear_speed = self.max_linear_spin.value()
        a.max_angular_speed = self.max_angular_spin.value()
        a.invert_x = self.invert_x_check.isChecked()
        a.invert_y = self.invert_y_check.isChecked()
        a.invert_z = self.invert_z_check.isChecked()
        a.invert_roll = self.invert_roll_check.isChecked()
        a.invert_pitch = self.invert_pitch_check.isChecked()

    def _build_controller(self, args: argparse.Namespace):
        """조이스틱/키보드 컨트롤러를 만든다. 실패하면 None을 돌려주고(QMessageBox로 이미 알림)
        호출 측이 그 자리에서 중단하면 된다."""
        resolved_input = resolve_input_mode(args.input)
        try:
            if resolved_input == "keyboard":
                ctl = QtKeyboardEEController()
                QApplication.instance().installEventFilter(ctl)
                self._keyboard_filter_installed = True
            else:
                ctl = JoystickEEController(recalibrate=args.recalibrate_joystick)
        except RuntimeError as e:
            QMessageBox.critical(self, "입력 장치 오류", str(e))
            return None
        return ctl

    def _close_controller(self) -> None:
        if self._ctl is not None:
            if self._keyboard_filter_installed:
                QApplication.instance().removeEventFilter(self._ctl)
                self._keyboard_filter_installed = False
            self._ctl.close()
            self._ctl = None

    def _focus_field(self, widget: QWidget) -> None:
        """설정 패널이 스크롤 영역이라 필요한 입력칸이 화면 밖에 있을 수 있다(2026-10-06:
        repo-id가 스크롤에 가려 안 보여서 엉뚱한 칸에 입력하고 헷갈렸다는 피드백) — 에러를
        보여줄 때 그 칸이 보이게 스크롤하고 포커스까지 준다."""
        self._left_scroll.ensureWidgetVisible(widget)
        widget.setFocus()

    def _connect_real_robot(self, args: argparse.Namespace) -> bool:
        """실로봇에 연결한다. 이미 연결돼 있으면(테스트에서 이어오는 경우) 그대로 재사용한다.
        실패하면 QMessageBox로 알리고 False — 이 코드는 실제 하드웨어로 검증하지 못했다(이
        환경엔 로봇이 없음)."""
        if self._robot is not None:
            return True
        if not args.real_port:
            self._focus_field(self.real_port_edit)
            QMessageBox.warning(self, "포트 필요", "실로봇 연결에 필요한 port를 입력하세요(예: /dev/ttyACM0).")
            return False
        try:
            robot, kin = build_real_robot(
                port=args.real_port,
                wrist_camera_index=args.real_wrist_cam,
                overview_camera_index=args.real_overview_cam,
                max_relative_target=args.real_max_rel_target,
                calibrate=args.real_calibrate,
            )
        except Exception as e:  # noqa: BLE001 — lerobot/시리얼/카메라 쪽에서 다양한 예외가 올 수 있어 경계에서 폭넓게 받음
            QMessageBox.critical(self, "실로봇 연결 오류", str(e))
            return False
        self._robot = robot
        self._kin = kin
        return True

    def _close_real_robot(self) -> None:
        if self._robot is not None:
            try:
                self._robot.disconnect()
            except Exception:  # noqa: BLE001 — 종료 경로라 실패해도 상태만 정리하고 넘어감
                pass
            self._robot = None
            self._kin = None

    def _make_ticker(self, scene: str | None = None, variant: int | None = None):
        """소스(시뮬/실로봇)에 따라 새 에피소드용 ticker를 만든다. 실로봇은 scene/variant를
        무시한다 — 물리적 "리셋"이 없어서 로봇의 현재 자세가 곧 다음 에피소드의 시작점이다."""
        if self._args.source == SOURCE_REAL:
            return RealRobotEpisodeTicker(self._args, self._robot, self._kin)
        if scene is None:
            scene = self.scene_combo.currentText()
            if scene == "balanced":
                scene = random.choice(SCENE_VARIANTS)
        if variant is None:
            variant = self.variant_spin.value()
            if variant < 0:
                variant = random.randint(0, N_VARIANTS - 1)
        mjcf_path = _mjcf_path(scene, variant)
        model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        dual_cam = MujocoDualCamera(model)
        return EpisodeTicker(self._args, model, data, dual_cam, scene, variant)

    # ---- 조작 테스트 -----------------------------------------------------
    def _on_start_test(self) -> None:
        args = self._collect_args()
        if args.source == SOURCE_REAL and not self._connect_real_robot(args):
            return
        ctl = self._build_controller(args)
        if ctl is None:
            if args.source == SOURCE_REAL:
                self._close_real_robot()
            return
        self._ctl = ctl
        self._args = args
        self._dataset = None
        self._test_mode = True
        self._start_test_episode()

        self.start_test_btn.setEnabled(False)
        self.stop_test_btn.setEnabled(True)
        self.test_to_record_btn.setEnabled(True)
        self.start_session_btn.setEnabled(False)
        self._timer.start(max(1, 1000 // args.fps))
        self._status_label.setText(
            "조작 테스트 중 — 설정(속도/반전)을 바꿔가며 움직여보세요(저장 안 함). "
            "ENTER/BACKSPACE로 다음 씬. 만족스러우면 '테스트 통과'를 누르세요."
        )

    def _start_test_episode(self) -> None:
        self._ticker = self._make_ticker()
        self._force_end = False
        self._force_discard = False

    def _on_stop_test(self) -> None:
        self._timer.stop()
        if self._ticker is not None:
            self._ticker.dual_cam.close()
            self._ticker = None
        self._close_controller()
        self._close_real_robot()
        self._test_mode = False
        self.start_test_btn.setEnabled(True)
        self.stop_test_btn.setEnabled(False)
        self.test_to_record_btn.setEnabled(False)
        self.start_session_btn.setEnabled(True)
        self._wrist_view.clear()
        self._overview_view.clear()
        self._status_label.setText("테스트 종료.")

    def _on_test_to_record(self) -> None:
        args = self._collect_args()
        if not args.dry_run and not args.repo_id:
            self._focus_field(self.repo_id_edit)
            QMessageBox.warning(self, "repo-id 필요", "dry-run이 아니면 repo-id나 root 중 하나는 입력해야 합니다.")
            return
        self._timer.stop()
        if self._ticker is not None:
            self._ticker.dual_cam.close()
            self._ticker = None
        self._test_mode = False
        self._begin_recording_session(args, reuse_ctl=True)

    # ---- 세션 ---------------------------------------------------------
    def _on_start_session(self) -> None:
        args = self._collect_args()
        if not args.dry_run and not args.repo_id:
            self._focus_field(self.repo_id_edit)
            QMessageBox.warning(self, "repo-id 필요", "dry-run이 아니면 repo-id나 root 중 하나는 입력해야 합니다.")
            return
        self._begin_recording_session(args, reuse_ctl=False)

    def _begin_recording_session(self, args: argparse.Namespace, reuse_ctl: bool) -> None:
        """실제 녹화 세션을 연다. reuse_ctl=True면 조작 테스트에서 이미 만든 컨트롤러(+실로봇
        연결)를 다시 만들지 않고 그대로 쓴다(테스트에서 바로 넘어오는 경로)."""
        is_real = args.source == SOURCE_REAL
        if is_real and not reuse_ctl and not self._connect_real_robot(args):
            return
        if not reuse_ctl:
            ctl = self._build_controller(args)
            if ctl is None:
                if is_real:
                    self._close_real_robot()
                return
            self._ctl = ctl

        combos = [] if is_real else _build_combos(args.scene, args.variant)
        try:
            dataset = None if args.dry_run else build_dataset_gui(args, self)
            if is_real:
                sampler = None
            elif args.dry_run:
                sampler = BalancedSceneSampler(combos, counts_path=None)
            else:
                counts_path = Path(dataset.root) / "meta" / "scene_balance.json"
                sampler = BalancedSceneSampler(combos, counts_path)
        except DatasetCancelled:
            if reuse_ctl:
                # 테스트에서 넘어오다 취소된 경우 — 컨트롤러(+실로봇 연결)는 아직 살아있으니
                # 테스트로 되돌아간다.
                self._test_mode = True
                self.test_to_record_btn.setEnabled(True)
                self.stop_test_btn.setEnabled(True)
                self._start_test_episode()
                self._timer.start(max(1, 1000 // args.fps))
                self._status_label.setText("취소됨 — 조작 테스트를 계속합니다.")
            else:
                self._close_controller()
                if is_real:
                    self._close_real_robot()
            return

        self._dataset = dataset
        self._sampler = sampler
        self._combos = combos
        self._args = args
        self._saved_count = 0

        self._set_settings_enabled(False)
        self.start_session_btn.setEnabled(False)
        self.end_session_btn.setEnabled(True)
        self.start_episode_btn.setEnabled(True)
        self.start_test_btn.setEnabled(False)
        self.stop_test_btn.setEnabled(False)
        self.test_to_record_btn.setEnabled(False)
        self._refresh_chart()
        self._timer.start(max(1, 1000 // args.fps))
        self._status_label.setText("세션 시작됨 — 에피소드 시작을 누르세요.")

    def _set_settings_enabled(self, enabled: bool) -> None:
        for w in (
            self.source_combo,
            self.repo_id_edit, self.root_edit, self.dry_run_check, self.scene_combo, self.variant_spin,
            self.num_episodes_spin, self.episode_seconds_spin, self.fps_spin, self.max_linear_spin,
            self.max_angular_spin, self.invert_x_check, self.invert_y_check, self.invert_z_check,
            self.invert_roll_check, self.invert_pitch_check, self.input_combo, self.recalibrate_check,
            self.real_port_edit, self.real_wrist_cam_spin, self.real_overview_cam_spin,
            self.real_max_rel_target_spin, self.real_calibrate_check,
        ):
            w.setEnabled(enabled)

    def _on_end_session(self) -> None:
        self._timer.stop()
        if self._ticker is not None:
            self._ticker.dual_cam.close()
            self._ticker = None
        self._close_controller()
        self._close_real_robot()
        if self._dataset is not None:
            self._dataset.finalize()
            self._status_label.setText(f"세션 종료 — 데이터셋: {self._dataset.root}")
            self._dataset = None
        else:
            self._status_label.setText("세션 종료 (dry-run, 저장 안 함)")

        self._set_settings_enabled(True)
        self.start_session_btn.setEnabled(True)
        self.end_session_btn.setEnabled(False)
        self.start_test_btn.setEnabled(True)
        for btn in (self.start_episode_btn, self.save_btn, self.discard_btn):
            btn.setEnabled(False)
        self._wrist_view.clear()
        self._overview_view.clear()

    # ---- 에피소드 ------------------------------------------------------
    def _on_start_episode(self) -> None:
        assert self._args is not None
        if self._saved_count >= self._args.num_episodes:
            self._status_label.setText("모든 에피소드 완료 — 세션 종료를 누르세요.")
            return
        if self._sampler is not None:
            scene, variant = self._sampler.pick()
            self._ticker = self._make_ticker(scene, variant)
        else:
            self._ticker = self._make_ticker()
            scene, variant = self._ticker.scene, self._ticker.variant
        self._force_end = False
        self._force_discard = False
        self.start_episode_btn.setEnabled(False)
        self.save_btn.setEnabled(True)
        self.discard_btn.setEnabled(True)
        self._status_label.setText(
            f"녹화 중 — 에피소드 {self._saved_count + 1}/{self._args.num_episodes} scene={scene} variant={variant}"
        )

    def _on_request_save(self) -> None:
        self._force_end = True

    def _on_request_discard(self) -> None:
        self._force_discard = True

    def _tick(self) -> None:
        if self._ticker is None or self._ctl is None:
            return
        if self._test_mode:
            self._apply_live_feel_settings()

        ticker = self._ticker
        result = ticker.tick(self._ctl, self._dataset, self._force_end, self._force_discard)

        self._wrist_view.set_frame(ticker.last_wrist_frame)
        self._overview_view.set_frame(ticker.last_overview_frame)
        self.state_scene_label.setText(f"{ticker.scene} / {ticker.variant}")
        self.state_time_label.setText(f"{ticker.step / self._args.fps:.1f} s")
        if ticker.target_pos is not None:
            p = ticker.target_pos
            self.state_pos_label.setText(f"x={p[0]:+.3f} y={p[1]:+.3f} z={p[2]:+.3f}")
        if ticker.last_tip is not None:
            self.state_tip_label.setText(f"{ticker.last_tip[2]:+.3f}")
        self.state_trigger_label.setText(f"{ticker.last_trigger:.0f}")
        self.state_contact_label.setText("접촉" if ticker.last_contact else "정상")
        self.state_bead_label.setText(str(len(ticker.bead_points)))
        if self._test_mode:
            self.state_progress_label.setText("조작 테스트 중 (저장 안 함)")
        else:
            self.state_progress_label.setText(f"{self._saved_count}/{self._args.num_episodes}")

        if result is not None:
            ticker.dual_cam.close()
            if self._test_mode:
                self._start_test_episode()
                return
            if result == "saved":
                if self._sampler is not None:
                    self._sampler.commit(ticker.scene, ticker.variant)
                self._saved_count += 1
            self._ticker = None
            self.start_episode_btn.setEnabled(True)
            self.save_btn.setEnabled(False)
            self.discard_btn.setEnabled(False)
            label = "저장됨" if result == "saved" else "폐기됨"
            self._status_label.setText(
                f"에피소드 {label} ({self._saved_count}/{self._args.num_episodes}) — "
                "에피소드 시작을 누르면 다음 씬으로 이어서."
            )
            self._refresh_chart()

    # ---- 차트 ----------------------------------------------------------
    def _refresh_chart(self) -> None:
        if self._sampler is None:
            return
        scene_names = sorted({s for s, _ in self._combos})
        agg: dict[str, int] = {s: 0 for s in scene_names}
        for key, n in self._sampler.counts.items():
            s = key.split(":")[0]
            agg[s] = agg.get(s, 0) + n
        self._chart.clear()
        xs = list(range(len(scene_names)))
        heights = [agg[s] for s in scene_names]
        bar = pg.BarGraphItem(x=xs, height=heights, width=0.6, brush="steelblue")
        self._chart.addItem(bar)
        self._chart.getAxis("bottom").setTicks([[(i, name) for i, name in enumerate(scene_names)]])

    # ---- 종료 ----------------------------------------------------------
    def shutdown(self) -> None:
        """메인 윈도우가 닫힐 때 호출 — 세션/테스트가 켜져 있으면 안전하게 정리."""
        if self.end_session_btn.isEnabled():
            self._on_end_session()
        elif self.stop_test_btn.isEnabled():
            self._on_stop_test()
