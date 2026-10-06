"""numpy (H,W,3) uint8 RGB 프레임 -> QPixmap 변환, 카메라 화면용 QLabel 위젯."""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap, QResizeEvent
from PyQt6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget


def rgb_to_pixmap(frame: np.ndarray) -> QPixmap:
    """mujoco.Renderer.render()는 매 호출마다 내부 버퍼를 재사용하므로, QImage가 그 버퍼를 계속
    참조하게 두면(copy 없이) 다음 render() 호출 때 화면이 깨진다 — 반드시 복사해서 떼어낸다."""
    frame = np.ascontiguousarray(frame)
    h, w, _ = frame.shape
    qimg = QImage(frame.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


class _AspectRatioLabel(QLabel):
    """원본 비율을 유지한 채 라벨 영역 안에 꽉 차게(letterbox) 그린다 — setScaledContents(True)는
    라벨 모양대로 늘려버려서(2026-10-06 지적: "이미지를 좌우로 늘린 것 같다") 안 쓴다. 원본
    QPixmap을 들고 있다가 라벨 크기가 바뀔 때마다 그 크기에 맞춰 다시 스케일한다."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._original: QPixmap | None = None

    def set_original_pixmap(self, pixmap: QPixmap) -> None:
        self._original = pixmap
        self._apply_scaled()

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802 — Qt override 시그니처
        super().resizeEvent(event)
        self._apply_scaled()

    def _apply_scaled(self) -> None:
        if self._original is None or self._original.isNull():
            return
        scaled = self._original.scaled(
            self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        super().setPixmap(scaled)


class CameraView(QWidget):
    """제목 + 카메라 화면 한 장. 고정 크기를 안 주고 부모 레이아웃이 내준 영역을 꽉 채우되
    (2026-10-06: 손목/오버뷰를 세로로 쌓아서 오른쪽 열을 꽉 채워달라는 요청), 원본 종횡비는
    유지한다(같은 날 지적: 늘어나 보이지 않게 — 남는 공간은 왼쪽 설정 패널이 가져가도록
    collect_tab.py의 스트레치 비율도 같이 조정함)."""

    def __init__(self, title: str, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label = _AspectRatioLabel()
        self._image_label.setMinimumSize(160, 120)  # 너무 쪼그라들지 않게 최소값만 둠
        self._image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._image_label.setStyleSheet("background-color: #222; color: #888;")
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setText("대기 중")
        layout.addWidget(title_label)
        layout.addWidget(self._image_label, stretch=1)

    def set_frame(self, frame: np.ndarray) -> None:
        self._image_label.set_original_pixmap(rgb_to_pixmap(frame))

    def clear(self) -> None:
        self._image_label.set_original_pixmap(QPixmap())
        self._image_label.clear()
        self._image_label.setText("대기 중")
