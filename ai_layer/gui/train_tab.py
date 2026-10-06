"""학습 탭 — BC(train_bc.py)/RL(train_rl.py)을 subprocess로 실행하고 로그 + metrics.jsonl을
실시간으로 보여준다. 두 스크립트 다 이미 `--out-dir/<...>` 밑에 `metrics.jsonl`을 한 줄씩
쌓고 있어서(train_bc.py는 2026-10-06에 이 GUI를 위해 추가, train_rl.py는 원래부터) 그 파일을
주기적으로 tail해서 그래프를 그린다 — 콘솔 출력을 regex로 긁지 않는다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pyqtgraph as pg
from PyQt6.QtCore import QProcess, QTimer
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from PyQt6.QtCore import Qt

_REPO_ROOT = Path(__file__).resolve().parents[2]


class _TrainingSection(QWidget):
    """BC/RL 공통 틀: 설정 폼 + Start/Stop + 로그 뷰 + metrics.jsonl 실시간 그래프.

    서브클래스는 `_build_settings_form(form)`로 자기만의 입력 위젯을 추가하고,
    `_build_command()`로 (argv 리스트, out_dir)을 돌려주고, `_metric_series()`로 그릴
    (레이블, metrics.jsonl 레코드에서 뽑을 키) 목록을 준다.
    """

    series_spec: list[tuple[str, str]] = []  # [(표시 이름, metrics.jsonl 키)]

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.process: QProcess | None = None
        self._metrics_path: Path | None = None
        self._metrics_pos = 0
        self._series_x: dict[str, list[float]] = {name: [] for name, _ in self.series_spec}
        self._series_y: dict[str, list[float]] = {name: [] for name, _ in self.series_spec}

        layout = QVBoxLayout(self)
        form_box = QGroupBox("설정")
        self._form = QFormLayout(form_box)
        self._build_settings_form(self._form)
        layout.addWidget(form_box)

        btn_row = QHBoxLayout()
        self.start_btn = QPushButton("학습 시작")
        self.stop_btn = QPushButton("중지")
        self.stop_btn.setEnabled(False)
        self.start_btn.clicked.connect(self._on_start)
        self.stop_btn.clicked.connect(self._on_stop)
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)
        layout.addLayout(btn_row)

        splitter = QSplitter(Qt.Orientation.Vertical)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)
        self.plot = pg.PlotWidget()
        self.plot.setBackground("w")
        self.plot.addLegend()
        self._curves = {}
        colors = ["b", "r", "g", "m", "c"]
        for i, (name, _key) in enumerate(self.series_spec):
            self._curves[name] = self.plot.plot([], [], pen=colors[i % len(colors)], name=name)
        splitter.addWidget(self.log_view)
        splitter.addWidget(self.plot)
        layout.addWidget(splitter)

        self._metrics_timer = QTimer(self)
        self._metrics_timer.timeout.connect(self._poll_metrics)

    # 서브클래스가 구현 ----------------------------------------------------
    def _build_settings_form(self, form: QFormLayout) -> None:
        raise NotImplementedError

    def _build_command(self) -> tuple[list[str], Path] | None:
        """(argv, out_dir) 또는 입력 오류면 None(에러는 자체적으로 표시)."""
        raise NotImplementedError

    # 공통 ----------------------------------------------------------------
    def _on_start(self) -> None:
        built = self._build_command()
        if built is None:
            return
        argv, out_dir = built
        out_dir.mkdir(parents=True, exist_ok=True)
        self._metrics_path = out_dir / "metrics.jsonl"
        self._metrics_pos = 0
        for name in self._series_x:
            self._series_x[name].clear()
            self._series_y[name].clear()
            self._curves[name].setData([], [])
        self.log_view.clear()

        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(_REPO_ROOT))
        env = self.process.processEnvironment()
        env.insert("PYTHONPATH", str(_REPO_ROOT))
        self.process.setProcessEnvironment(env)
        self.process.setProgram(sys.executable)
        self.process.setArguments(argv)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._on_output)
        self.process.finished.connect(self._on_finished)
        self.process.start()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._metrics_timer.start(1000)
        self.log_view.appendPlainText(f"$ {sys.executable} {' '.join(argv)}\n")

    def _on_stop(self) -> None:
        if self.process is not None and self.process.state() != QProcess.ProcessState.NotRunning:
            self.process.terminate()
            QTimer.singleShot(3000, self._kill_if_alive)

    def _kill_if_alive(self) -> None:
        if self.process is not None and self.process.state() != QProcess.ProcessState.NotRunning:
            self.process.kill()

    def _on_output(self) -> None:
        if self.process is None:
            return
        data = self.process.readAllStandardOutput().data().decode(errors="replace")
        if data:
            self.log_view.appendPlainText(data.rstrip("\n"))

    def _on_finished(self) -> None:
        self._metrics_timer.stop()
        self._poll_metrics()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.log_view.appendPlainText("\n[프로세스 종료]")

    def _poll_metrics(self) -> None:
        if self._metrics_path is None or not self._metrics_path.exists():
            return
        with self._metrics_path.open("r") as f:
            f.seek(self._metrics_pos)
            new_lines = f.readlines()
            self._metrics_pos = f.tell()
        if not new_lines:
            return
        for line in new_lines:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            step = rec.get("step")
            if step is None:
                continue
            for name, key in self.series_spec:
                val = rec.get(key)
                if val is not None:
                    self._series_x[name].append(step)
                    self._series_y[name].append(val)
        for name, _key in self.series_spec:
            self._curves[name].setData(self._series_x[name], self._series_y[name])

    def shutdown(self) -> None:
        if self.process is not None and self.process.state() != QProcess.ProcessState.NotRunning:
            self.process.terminate()


class BCTrainingSection(_TrainingSection):
    series_spec = [("loss", "loss")]

    def _build_settings_form(self, form: QFormLayout) -> None:
        self.repo_id_edit = QLineEdit()
        self.repo_id_edit.setPlaceholderText("lerobot-record/수집 GUI로 만든 데이터셋 repo-id 또는 로컬 경로")
        form.addRow("repo-id", self.repo_id_edit)
        self.root_edit = QLineEdit()
        self.root_edit.setPlaceholderText("로컬에 있으면 그 경로 (비우면 repo-id로 자동 탐색)")
        form.addRow("root", self.root_edit)
        self.out_dir_edit = QLineEdit("outputs/bc_act")
        form.addRow("out-dir", self.out_dir_edit)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 100000)
        self.epochs_spin.setValue(100)
        form.addRow("epochs", self.epochs_spin)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 1024)
        self.batch_spin.setValue(8)
        form.addRow("batch-size", self.batch_spin)
        self.device_combo = QComboBox()
        self.device_combo.addItems(["cuda", "cpu"])
        form.addRow("device", self.device_combo)
        self.xyz_only_check = QCheckBox("xyz-only (회전 마스킹, 4DOF MVP)")
        form.addRow("", self.xyz_only_check)

    def _build_command(self) -> tuple[list[str], Path] | None:
        repo_id = self.repo_id_edit.text().strip()
        if not repo_id:
            self.log_view.appendPlainText("[오류] repo-id를 입력하세요.")
            return None
        out_dir = Path(self.out_dir_edit.text().strip() or "outputs/bc_act")
        argv = [
            str(_REPO_ROOT / "ai_layer" / "train_bc.py"),
            "--repo-id", repo_id,
            "--epochs", str(self.epochs_spin.value()),
            "--batch-size", str(self.batch_spin.value()),
            "--device", self.device_combo.currentText(),
            "--out-dir", str(out_dir),
        ]
        root = self.root_edit.text().strip()
        if root:
            argv += ["--root", root]
        if self.xyz_only_check.isChecked():
            argv += ["--xyz-only"]
        out_dir_abs = out_dir if out_dir.is_absolute() else _REPO_ROOT / out_dir
        return argv, out_dir_abs


class RLTrainingSection(_TrainingSection):
    series_spec = [
        ("reward_mean", "reward_mean"),
        ("critic_loss", "critic_loss"),
        ("bc_action_distance_mean", "bc_action_distance_mean"),
    ]

    def _build_settings_form(self, form: QFormLayout) -> None:
        self.out_dir_edit = QLineEdit("outputs/rl_sac")
        form.addRow("out-dir", self.out_dir_edit)
        self.bc_checkpoint_edit = QLineEdit()
        self.bc_checkpoint_edit.setPlaceholderText("train_bc.py 체크포인트 폴더 (선택 — R_imitation 용)")
        form.addRow("bc-checkpoint", self.bc_checkpoint_edit)
        self.num_steps_spin = QSpinBox()
        self.num_steps_spin.setRange(1, 10_000_000)
        self.num_steps_spin.setValue(200_000)
        form.addRow("num-steps", self.num_steps_spin)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 4096)
        self.batch_spin.setValue(256)
        form.addRow("batch-size", self.batch_spin)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2**31 - 1)
        self.seed_spin.setValue(0)
        form.addRow("seed", self.seed_spin)

    def _build_command(self) -> tuple[list[str], Path] | None:
        out_dir = Path(self.out_dir_edit.text().strip() or "outputs/rl_sac")
        argv = [
            str(_REPO_ROOT / "ai_layer" / "train_rl.py"),
            "--num-steps", str(self.num_steps_spin.value()),
            "--batch-size", str(self.batch_spin.value()),
            "--seed", str(self.seed_spin.value()),
            "--out-dir", str(out_dir),
        ]
        bc_ckpt = self.bc_checkpoint_edit.text().strip()
        if bc_ckpt:
            argv += ["--bc-checkpoint", bc_ckpt]
        out_dir_abs = out_dir if out_dir.is_absolute() else _REPO_ROOT / out_dir
        return argv, out_dir_abs


class TrainTab(QWidget):
    """BC/RL 하위 탭을 묶는다."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        sub_tabs = QTabWidget()
        self.bc_section = BCTrainingSection()
        self.rl_section = RLTrainingSection()
        sub_tabs.addTab(self.bc_section, "BC (지도학습)")
        sub_tabs.addTab(self.rl_section, "RL (SAC)")
        layout.addWidget(sub_tabs)

    def shutdown(self) -> None:
        self.bc_section.shutdown()
        self.rl_section.shutdown()
