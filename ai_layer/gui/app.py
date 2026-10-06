#!/usr/bin/env python
"""PyQt6 데이터 수집 + 학습 통합 GUI 진입점.

실행 (pac2026 conda 환경, PyQt6/pyqtgraph 필요: pip install PyQt6 pyqtgraph):
  PYTHONPATH=. python ai_layer/gui/app.py

데이터 수집 탭: 손목/오버뷰 카메라 실시간 화면, 로봇 상태, 형태별 수집 현황 차트, 세션/에피소드
버튼. 설정(repo-id, scene, 속도, 입력 장치 등)은 전부 화면에서 조정 — CLI 플래그 없음.
입력은 조이스틱 자동 탐색, 없으면 Qt 네이티브 키보드 입력(이 창에 포커스가 있을 때 WASD 등)으로
자동 전환 — ai_layer/tools/keyboard_input.py(터미널 raw 모드)와 달리 이 GUI 전용
QtKeyboardEEController(ai_layer/gui/qt_keyboard_input.py)를 쓴다(진짜 press/release 이벤트를
주는 Qt 쪽이 터미널/GLFW보다 더 정확함).

학습 탭: train_bc.py/train_rl.py를 subprocess로 실행하고 로그 + metrics.jsonl을 실시간으로
그래프로 보여준다.
"""

from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication

from ai_layer.gui.main_window import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.showMaximized()  # 2026-10-06: 기본으로 전체 화면(최대화)으로 열어달라는 요청
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
