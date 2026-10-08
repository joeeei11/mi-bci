"""Operator-facing online control using the acquisition monitor's EEG blocks."""

import queue
import sqlite3
import time
from pathlib import Path

from PySide6.QtCore import QThread, QTimer, Signal
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
                               QMessageBox, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget)

from .online import LineTransport, OnlineConfig, OnlineSession, VirtualSTM32
from .project import Project


class OnlineWorker(QThread):
    updated = Signal(object)

    def __init__(self, project_path, kind, target, config=None):
        super().__init__()
        self.project_path = project_path
        self.kind = kind
        self.target = target
        self.config = config
        self.blocks = queue.Queue(maxsize=8)
        self.stop_requested = False
        self.fault_reason = None

    def submit(self, block):
        try:
            self.blocks.put_nowait(block)
        except queue.Full:
            self.fail("在线处理跟不上脑电输入")

    def fail(self, reason):
        self.fault_reason = reason
        self.stop_requested = True

    def halt(self):
        self.stop_requested = True
        self.wait()

    def run(self):
        project = None
        session = None
        transport = VirtualSTM32() if self.kind == "虚拟 STM32" else LineTransport(
            {"USB 串口": "usb", "蓝牙串口": "bluetooth", "TCP / Wi-Fi": "tcp"}[self.kind], self.target)
        try:
            connection = sqlite3.connect(Path(self.project_path) / "metadata.sqlite")
            connection.execute("PRAGMA foreign_keys = ON")
            project = Project(Path(self.project_path), connection)
            session = OnlineSession(project, transport, config=self.config)
            previous_count = 0
            while not self.stop_requested:
                try:
                    block = self.blocks.get(timeout=0.05)
                except queue.Empty:
                    if session.state == "running":
                        session.tick()
                    block = None
                if block is not None:
                    if self.stop_requested:
                        break
                    if session.state == "idle":
                        session.start([channel.name for channel in block.channels], block.sampling_rate)
                    if session.state == "running":
                        session.feed(block)
                for event in session.events[previous_count:]:
                    self.updated.emit({"event": event, "state": session.state,
                                       "stop_confirmed": session.stop_confirmed})
                previous_count = len(session.events)
                if session.state == "fault":
                    break
            if self.fault_reason and session.state != "fault":
                session.fault(self.fault_reason)
            if session.state != "idle":
                session.stop()
            self.updated.emit({"state": session.state, "stop_confirmed": session.stop_confirmed,
                               "events": session.events[previous_count:]})
        except Exception as exc:
            if session is not None:
                session.fault(str(exc))
            self.updated.emit({"state": "fault", "reason": str(exc),
                               "stop_confirmed": bool(session and session.stop_confirmed)})
        finally:
            transport.close()
            if project:
                project.close()


class OnlineWindow(QWidget):
    def __init__(self, monitor):
        super().__init__()
        self.monitor = monitor
        self.worker = None
        self.fault_latched = False
        self.setWindowTitle("MI-BCI | 在线控制")
        self.resize(620, 450)
        self.setStyleSheet("QWidget{background:#faf8f5;color:#273b39;font-size:14px} "
                           "QPushButton{background:#4a9d9a;color:white;border:0;border-radius:5px;padding:8px} "
                           "QLineEdit,QComboBox,QPlainTextEdit{background:white;border:1px solid #cccac4;padding:5px}")
        layout = QVBoxLayout(self)
        self.summary = QLabel("空闲 · 无设备输出")
        self.summary.setStyleSheet("font-size:18px;font-weight:600")
        layout.addWidget(self.summary)
        form = QFormLayout()
        self.transport = QComboBox()
        self.transport.addItems(["虚拟 STM32", "USB 串口", "蓝牙串口", "TCP / Wi-Fi"])
        self.target = QLineEdit()
        self.target.setPlaceholderText("COM3 或 192.168.1.10:9000")
        form.addRow("STM32 连接", self.transport)
        form.addRow("地址", self.target)
        self.window_seconds = QDoubleSpinBox()
        self.window_seconds.setRange(0.2, 10)
        self.window_seconds.setValue(1.0)
        self.window_seconds.setSingleStep(0.25)
        self.hop_seconds = QDoubleSpinBox()
        self.hop_seconds.setRange(0.05, 2)
        self.hop_seconds.setValue(0.25)
        self.hop_seconds.setSingleStep(0.05)
        self.cooldown_seconds = QDoubleSpinBox()
        self.cooldown_seconds.setRange(0, 10)
        self.cooldown_seconds.setValue(1.0)
        form.addRow("推理窗口（秒）", self.window_seconds)
        form.addRow("推理步长（秒）", self.hop_seconds)
        form.addRow("指令冷却（秒）", self.cooldown_seconds)
        layout.addLayout(form)
        self.model = QLabel("模型：--")
        self.signal = QLabel("信号：--")
        self.confidence = QLabel("判断：--")
        self.command = QLabel("指令：STOP · 未连接")
        for widget in (self.model, self.signal, self.confidence, self.command):
            layout.addWidget(widget)
        row = QHBoxLayout()
        self.start_button = QPushButton("启动在线控制")
        self.start_button.clicked.connect(self.start)
        self.stop_button = QPushButton("STOP")
        self.stop_button.setStyleSheet("background:#ae3427;color:white")
        self.stop_button.clicked.connect(self.stop)
        self.reset_button = QPushButton("确认设备安全并复位")
        self.reset_button.setEnabled(False)
        self.reset_button.clicked.connect(self.reset_fault)
        row.addWidget(self.start_button)
        row.addWidget(self.stop_button)
        row.addWidget(self.reset_button)
        layout.addLayout(row)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log)
        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.check_source)

    def start(self):
        monitor = self.monitor
        if self.fault_latched:
            return
        if self.worker and self.worker.isRunning():
            return
        if not monitor.project or not monitor.project.active_model():
            QMessageBox.warning(self, "需要模型", "请先打开项目并激活模型")
            return
        if not monitor.worker or not monitor.worker.isRunning():
            QMessageBox.warning(self, "需要信号", "请先开始脑电采集或回放")
            return
        kind, target = self.transport.currentText(), self.target.text().strip()
        if kind != "虚拟 STM32" and not target:
            QMessageBox.warning(self, "需要地址", "请输入 STM32 端口或地址")
            return
        config = OnlineConfig(window_seconds=self.window_seconds.value(), hop_seconds=self.hop_seconds.value(),
                              cooldown_seconds=self.cooldown_seconds.value())
        try:
            config.validate()
        except ValueError as exc:
            QMessageBox.warning(self, "参数错误", str(exc))
            return
        model = monitor.project.active_model()
        if QMessageBox.question(self, "确认启动在线输出",
                                f"项目：{monitor.project.name}\n模型：{model['id']}\n设备：{kind} {target}\n确认允许向设备发送控制指令？") != QMessageBox.Yes:
            return
        monitor.project.set_config({"online_control": config.__dict__})
        self.worker = OnlineWorker(monitor.project.root, kind, target, config)
        self.worker.updated.connect(self.update_status)
        self.worker.start()
        self.timer.start()
        self.start_button.setEnabled(False)
        self.transport.setEnabled(False)
        self.target.setEnabled(False)
        for control in (self.window_seconds, self.hop_seconds, self.cooldown_seconds):
            control.setEnabled(False)
        self.model.setText(f"模型：{model['algorithm']} · {model['id']}")
        self.summary.setText("连接中 · 等待脑电")

    def feed(self, block):
        if self.worker and self.worker.isRunning():
            self.signal.setText(f"信号：{block.health.value} · {block.sampling_rate:g} Hz")
            self.worker.submit(block)

    def check_source(self):
        if self.worker and self.worker.isRunning() and (not self.monitor.worker or not self.monitor.worker.isRunning()):
            self.worker.fail("脑电输入结束")
            self.worker.wait()
        if not self.worker or not self.worker.isRunning():
            self.timer.stop()
            self.start_button.setEnabled(not self.fault_latched)
            self.transport.setEnabled(True)
            self.target.setEnabled(True)
            for control in (self.window_seconds, self.hop_seconds, self.cooldown_seconds):
                control.setEnabled(True)

    def update_status(self, update):
        for event in ([update["event"]] if "event" in update else update.get("events", [])):
            kind = event["kind"]
            if kind == "prediction":
                self.confidence.setText(f"判断：{event['label']} · {event['confidence']:.1%}")
            if kind == "command":
                self.command.setText(f"指令：{event['command']} · {'ACK' if event['acknowledged'] else '未确认'}")
            if kind == "fault":
                self.confidence.setText("判断：--")
            self.log.appendPlainText(f"{time.strftime('%H:%M:%S')}  {kind}  {event}")
        if update.get("reason"):
            self.log.appendPlainText("故障：" + update["reason"])
        state = update.get("state", "idle")
        confirmed = update.get("stop_confirmed", False)
        if state == "fault":
            self.fault_latched = True
            self.start_button.setEnabled(False)
            self.reset_button.setEnabled(True)
            self.summary.setText("故障 · STOP 已确认" if confirmed else "故障 · STOP 未确认；请检查设备")
            self.summary.setStyleSheet("color:#ae3427;font-size:18px;font-weight:600")
        elif state == "running":
            self.summary.setText("在线推理中")
        elif state == "idle":
            self.summary.setText("已停止 · STOP 已确认" if confirmed else "空闲 · 无设备输出")

    def stop(self):
        if self.worker and self.worker.isRunning():
            self.worker.halt()
        self.check_source()

    def reset_fault(self):
        if self.worker and self.worker.isRunning():
            return
        if QMessageBox.question(self, "确认复位", "请确认 STM32 输出已安全停止、连接与信号已检查。现在复位？") != QMessageBox.Yes:
            return
        self.fault_latched = False
        if self.monitor.project:
            with self.monitor.project.db:
                self.monitor.project._audit("online_reset", {"confirmed_by_operator": True})
        self.reset_button.setEnabled(False)
        self.start_button.setEnabled(True)
        self.summary.setText("已复位 · 需要重新确认启动")
        self.summary.setStyleSheet("font-size:18px;font-weight:600")

    def source_fault(self, message):
        if self.worker and self.worker.isRunning():
            self.worker.fail(message)
            self.worker.wait()
            self.log.appendPlainText("信号故障：" + message)

    def closeEvent(self, event):
        self.stop()
        super().closeEvent(event)
