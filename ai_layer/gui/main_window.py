from __future__ import annotations

from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import QMainWindow, QTabWidget

from ai_layer.gui.collect_tab import CollectTab
from ai_layer.gui.train_tab import TrainTab


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PAC2026 — 데이터 수집 & 학습")

        tabs = QTabWidget()
        self.collect_tab = CollectTab()
        self.train_tab = TrainTab()
        tabs.addTab(self.collect_tab, "데이터 수집")
        tabs.addTab(self.train_tab, "학습")
        self.setCentralWidget(tabs)
        self.resize(1500, 900)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 — Qt override 시그니처
        self.collect_tab.shutdown()
        self.train_tab.shutdown()
        super().closeEvent(event)
