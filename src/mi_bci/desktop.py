"""Qt acquisition monitor. Run with: python -m mi_bci.desktop"""

import sys
import time
from collections import deque
from pathlib import Path

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (QApplication, QComboBox, QFileDialog, QFormLayout, QFrame,
                               QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox,
                               QPlainTextEdit, QPushButton, QSpinBox, QVBoxLayout, QWidget)

from .acquisition import BluetoothSerialSource, FileSource, LSLSource, SerialSource, TCPSource
from .project import Project
from .sources import ReplaySource, SyntheticSource
from .stream import StreamState


class AcquisitionWorker(QThread):
    block = Signal(object)
    failed = Signal(str)
    finished_source = Signal()

    def __init__(self, source, paced=False):
        super().__init__()
        self.source = source
        self.paced = paced
        self._stop = False

    def run(self):
        project = None
        source = self.source
        try:
            if isinstance(source, ReplaySource):
                project = Project.open(source.project.root)
                source = ReplaySource(project, source.recording_id)
            source.start()
            while not self._stop:
                block = source.read()
                if block is None:
                    break
                self.block.emit(block)
                if self.paced:
                    time.sleep(len(block.samples) / block.sampling_rate)
        except Exception as exc:
            if not self._stop:
                self.failed.emit(f"{type(exc).__name__}: {exc}")
        finally:
            source.stop()
            if project:
                project.close()
            self.finished_source.emit()

    def halt(self):
        self._stop = True
        self.wait(3000)


class Plot(QFrame):
    def __init__(self, spectrum=False):
        super().__init__()
        self.spectrum = spectrum
        self.setMinimumHeight(170 if spectrum else 260)
        self.setStyleSheet("background: white; border: 1px solid #e8e7e3; border-radius: 8px")
        self.data = []
        self.rate = 250

    def set_data(self, data, rate):
        self.data = data
        self.rate = rate
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        width, height = self.width(), self.height()
        painter.setPen(QPen(QColor("#e8e7e3"), 1))
        for fraction in (0.25, 0.5, 0.75):
            painter.drawLine(36, int(height * fraction), width - 16, int(height * fraction))
        if not self.data:
            painter.setPen(QColor("#65706e"))
            painter.drawText(self.rect(), Qt.AlignCenter, "等待信号")
            return
        colors = ("#27837b", "#b27a20", "#486e9b", "#87569a")
        if self.spectrum:
            try:
                import numpy as np
                for index, samples in enumerate(self.data[:4]):
                    if len(samples) < 4:
                        continue
                    values = np.asarray(samples[-min(len(samples), 1024):], dtype=float)
                    frequencies = np.fft.rfftfreq(len(values), 1 / self.rate)
                    power = np.abs(np.fft.rfft((values - values.mean()) * np.hanning(len(values)))) ** 2
                    band = frequencies <= 50
                    scale = max(float(power[band].max()), 1e-12)
                    painter.setPen(QPen(QColor(colors[index]), 2))
                    points = [(36 + int(float(f) / 50 * (width - 52)),
                               height - 25 - int(float(p) / scale * (height - 45)))
                              for f, p in zip(frequencies[band], power[band])]
                    for a, b in zip(points, points[1:]):
                        painter.drawLine(*a, *b)
            except ImportError:
                painter.drawText(self.rect(), Qt.AlignCenter, "频谱需要 NumPy")
        else:
            lanes = min(len(self.data), 4)
            for index, samples in enumerate(self.data[:lanes]):
                if len(samples) < 2:
                    continue
                center = (index + 0.5) * height / lanes
                peak = max(max(abs(v) for v in samples), 1.0)
                painter.setPen(QPen(QColor(colors[index]), 1.5))
                points = [(36 + int(pos / (len(samples) - 1) * (width - 52)),
                           int(center - value / peak * height / lanes * 0.36))
                          for pos, value in enumerate(samples)]
                for a, b in zip(points, points[1:]):
                    painter.drawLine(*a, *b)


class Monitor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MI-BCI | 采集与回放")
        self.resize(1120, 800)
        self.project = None
        self.worker = None
        self.research_window = None
        self.online_window = None
        self.ai_window = None
        self.samples = []
        self.setStyleSheet("""
            QWidget { background: #faf8f5; color: #273b39; font-size: 14px; }
            QLineEdit, QComboBox, QSpinBox, QPlainTextEdit { background: white; border: 1px solid #cccac4; padding: 5px; }
            QPushButton { background: #4a9d9a; color: white; border: 0; border-radius: 5px; padding: 8px 13px; }
            QPushButton:disabled { background: #b9c5c2; }
            QPushButton:focus, QLineEdit:focus, QComboBox:focus { border: 2px solid #236d69; }
        """)
        main = QWidget()
        self.setCentralWidget(main)
        layout = QVBoxLayout(main)
        heading = QLabel("脑电采集 / 回放")
        heading.setStyleSheet("font-size: 23px; font-weight: 600")
        layout.addWidget(heading)
        project_row = QHBoxLayout()
        self.project_label = QLabel("未打开项目")
        project_row.addWidget(self.project_label, 1)
        for label, action in (("打开项目", self.open_project), ("导入文件", self.import_recording),
                              ("实验与训练", self.open_research), ("在线控制", self.open_online),
                              ("AI 辅助研究", self.open_ai)):
            button = QPushButton(label)
            button.clicked.connect(action)
            project_row.addWidget(button)
        layout.addLayout(project_row)
        form = QFormLayout()
        self.kind = QComboBox()
        self.kind.addItems(["模拟", "项目回放", "CSV / EDF / BDF", "LSL", "串口", "TCP", "蓝牙串口"])
        self.target = QLineEdit()
        self.target.setPlaceholderText("文件路径 / LSL 流名 / COM 端口 / host:port")
        self.rate = QSpinBox()
        self.rate.setRange(1, 10000)
        self.rate.setValue(250)
        self.recording = QComboBox()
        form.addRow("信号来源", self.kind)
        form.addRow("地址或文件", self.target)
        form.addRow("CSV 采样率", self.rate)
        form.addRow("项目记录", self.recording)
        layout.addLayout(form)
        controls = QHBoxLayout()
        self.start_button = QPushButton("开始")
        self.start_button.clicked.connect(self.start_source)
        self.stop_button = QPushButton("停止")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_source)
        self.state_label = QLabel("空闲")
        self.state_label.setAccessibleName("连接状态")
        controls.addWidget(self.start_button)
        controls.addWidget(self.stop_button)
        controls.addWidget(self.state_label, 1)
        layout.addLayout(controls)
        self.channel_label = QLabel("通道：无")
        self.metrics_label = QLabel("采样率：--    最新时间：--    健康：--")
        layout.addWidget(self.channel_label)
        layout.addWidget(self.metrics_label)
        layout.addWidget(QLabel("实时波形"))
        self.wave = Plot()
        layout.addWidget(self.wave)
        layout.addWidget(QLabel("频谱 0–50 Hz"))
        self.spectrum = Plot(spectrum=True)
        layout.addWidget(self.spectrum)
        layout.addWidget(QLabel("采集日志"))
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(130)
        layout.addWidget(self.log)
        self.kind.currentTextChanged.connect(self._update_fields)
        self._update_fields()

    def _update_fields(self):
        kind = self.kind.currentText()
        self.target.setEnabled(kind not in {"模拟", "项目回放"})
        self.rate.setEnabled(kind == "CSV / EDF / BDF")
        self.recording.setEnabled(kind == "项目回放")

    def _log(self, message):
        self.log.appendPlainText(time.strftime("%H:%M:%S") + "  " + message)

    def open_project(self):
        path = QFileDialog.getExistingDirectory(self, "打开项目")
        if path:
            self.load_project(path)

    def load_project(self, path):
        if path:
            try:
                if self.research_window:
                    self.research_window.close()
                    self.research_window = None
                if self.online_window:
                    self.online_window.close()
                    self.online_window = None
                if self.ai_window:
                    self.ai_window.close()
                    self.ai_window = None
                if self.project:
                    self.project.close()
                self.project = Project.open(path)
                self.project_label.setText(f"项目：{self.project.name}")
                self.recording.clear()
                for row in self.project.db.execute("SELECT id, source_kind FROM recordings WHERE status = 'complete' ORDER BY created_at"):
                    self.recording.addItem(f"{row['source_kind']} · {row['id']}", row["id"])
                self._log("项目已打开")
            except Exception as exc:
                self._fault(str(exc))

    def import_recording(self):
        if not self.project:
            QMessageBox.warning(self, "需要项目", "请先打开项目")
            return
        path, _ = QFileDialog.getOpenFileName(self, "导入脑电文件", "", "EEG (*.csv *.edf *.bdf)")
        if path:
            from .acquisition import import_file
            try:
                identifier = import_file(self.project, path, self.rate.value())
                self.recording.addItem(f"{path.rsplit('.', 1)[-1]} · {identifier}", identifier)
                self._log(f"导入完成：{identifier}")
            except Exception as exc:
                self._fault(str(exc))

    def open_research(self):
        if not self.project:
            QMessageBox.warning(self, "需要项目", "请先打开项目")
            return
        from .research_desktop import ResearchWindow
        if self.research_window is None or self.research_window.project is not self.project:
            self.research_window = ResearchWindow(self.project, self)
        self.research_window.show()
        self.research_window.raise_()

    def open_online(self):
        if not self.project:
            QMessageBox.warning(self, "需要项目", "请先打开项目")
            return
        from .online_desktop import OnlineWindow
        if self.online_window is None:
            self.online_window = OnlineWindow(self)
        self.online_window.show()
        self.online_window.raise_()

    def open_ai(self):
        if not self.project:
            QMessageBox.warning(self, "需要项目", "请先打开项目")
            return
        from .ai_desktop import AIWindow

        if self.ai_window is None or self.ai_window.project is not self.project:
            self.ai_window = AIWindow(self.project, self)
        self.ai_window.show()
        self.ai_window.raise_()

    def start_source(self):
        try:
            if self.online_window:
                self.online_window.stop()
            kind, target = self.kind.currentText(), self.target.text().strip()
            if kind == "模拟":
                source = SyntheticSource(total_samples=250000)
            elif kind == "项目回放":
                if not self.project or self.recording.currentData() is None:
                    raise ValueError("请先打开包含记录的项目")
                source = ReplaySource(self.project, self.recording.currentData())
            elif kind == "CSV / EDF / BDF":
                source = FileSource(target, self.rate.value() if target.lower().endswith(".csv") else None)
            elif kind == "LSL":
                source = LSLSource(target or None)
            elif kind == "串口":
                source = SerialSource(target)
            elif kind == "蓝牙串口":
                source = BluetoothSerialSource(target)
            else:
                host, port = target.rsplit(":", 1)
                source = TCPSource(host, int(port))
            self.samples = []
            self.wave.set_data([], 250)
            self.spectrum.set_data([], 250)
            self.worker = AcquisitionWorker(source, paced=kind in {"模拟", "项目回放", "CSV / EDF / BDF"})
            self.worker.block.connect(self._block)
            self.worker.failed.connect(self._fault)
            self.worker.finished_source.connect(self._finished)
            self.worker.start()
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.state_label.setText("连接中")
            self.state_label.setStyleSheet("")
            self.channel_label.setText("通道：等待数据")
            self.metrics_label.setText("采样率：--    最新时间：--    健康：--")
            self._log(f"连接：{kind}")
        except Exception as exc:
            self._fault(str(exc))

    def _block(self, block):
        if len(self.samples) != len(block.channels):
            self.samples = [deque(maxlen=max(500, int(block.sampling_rate * 10))) for _ in block.channels]
        for sample in block.samples:
            for index, value in enumerate(sample):
                self.samples[index].append(value)
        data = [list(channel) for channel in self.samples]
        self.wave.set_data(data, block.sampling_rate)
        self.spectrum.set_data(data, block.sampling_rate)
        self.state_label.setText("采集中" if block.state == StreamState.RUNNING else str(block.state))
        self.channel_label.setText("通道：" + "  ".join(f"{ch.name} 接收中 ({ch.kind}, {ch.unit})" for ch in block.channels))
        self.metrics_label.setText(f"采样率：{block.sampling_rate:g} Hz    最新时间：{block.timestamps[-1]:.3f} s    健康：{block.health.value}")
        for event in block.events:
            self._log(f"事件：{event.code} @ {event.timestamp:.3f} s")
        if self.research_window:
            self.research_window.capture_block(block)
        if self.online_window:
            self.online_window.feed(block)

    def _fault(self, message):
        if self.online_window:
            self.online_window.source_fault(message)
        if self.research_window and self.research_window.timer.isActive():
            self.research_window.stop_protocol()
        self.state_label.setText("故障 / 连接断开；波形已冻结")
        self.state_label.setStyleSheet("color: #ae3427; font-weight: 600")
        self.channel_label.setText(self.channel_label.text().replace("接收中", "失联"))
        self.metrics_label.setText(self.metrics_label.text().replace("健康：good", "健康：lost"))
        self._log("故障：" + message)

    def _finished(self):
        if self.online_window:
            self.online_window.source_fault("脑电输入结束")
        if self.research_window and self.research_window.timer.isActive():
            self.research_window.stop_protocol()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        if not self.state_label.text().startswith("故障"):
            self.state_label.setText("已停止")
            self._log("采集结束")

    def stop_source(self):
        if self.worker and self.worker.isRunning():
            self.worker.halt()

    def closeEvent(self, event):
        if self.ai_window:
            self.ai_window.close()
        if self.online_window:
            self.online_window.close()
        self.stop_source()
        if self.research_window:
            self.research_window.close()
        if self.project:
            self.project.close()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    window = Monitor()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
