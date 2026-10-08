"""Guided experiment and offline model workspace."""

import hashlib
import time
import uuid
from dataclasses import asdict, replace
from pathlib import Path

from PySide6.QtCore import QThread, QTimer, Signal, Qt
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
                               QHBoxLayout, QLabel, QMessageBox, QPushButton,
                               QSpinBox, QVBoxLayout, QWidget)

from .project import Project
from .research import (ProtocolConfig, generate_report, load_iv2a,
                       load_recording, project_training_config, train_and_evaluate)
from .stream import EEGEvent


class TrainingWorker(QThread):
    succeeded = Signal(str)
    failed = Signal(str)

    def __init__(self, project_path, source, algorithm):
        super().__init__()
        self.project_path, self.source, self.algorithm = project_path, source, algorithm

    def run(self):
        project = None
        try:
            project = Project.open(self.project_path)
            config = project_training_config(project)
            if self.source[0] == "iv2a":
                trials, labels, channels, rate = load_iv2a(self.source[1], config)
                benchmark_subject = "iv2a-" + Path(self.source[1]).stem[1:3]
                with Path(self.source[1]).open("rb") as file:
                    source_hash = hashlib.file_digest(file, "sha256").hexdigest()
                source = {"kind": "BCI Competition IV-2a", "file": Path(self.source[1]).name,
                          "sha256": source_hash}
                recording_id = None
            else:
                recording_id = self.source[1]
                trials, labels, channels, rate = load_recording(project, recording_id, config)
                source = None
            identifier = train_and_evaluate(project, trials, labels, channels, rate,
                                            self.algorithm, config, recording_id, source,
                                            benchmark_subject if self.source[0] == "iv2a" else None)
            self.succeeded.emit(identifier)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        finally:
            if project:
                project.close()


class ResearchWindow(QWidget):
    def __init__(self, project: Project, monitor=None):
        super().__init__()
        self.project = project
        self.monitor = monitor
        self.worker = None
        self.capture = None
        self.pending_cues = []
        self.session_id = None
        self.schedule = []
        self.position = 0
        self.setWindowTitle("MI-BCI | 实验与模型")
        self.resize(720, 560)
        self.setStyleSheet("QWidget{background:#faf8f5;color:#273b39;font-size:14px} QPushButton{background:#4a9d9a;color:white;border:0;border-radius:5px;padding:8px} QPushButton:disabled{background:#b9c5c2;color:#273b39} QPushButton:focus,QComboBox:focus,QSpinBox:focus,QDoubleSpinBox:focus{border:2px solid #236d69} QComboBox,QSpinBox,QDoubleSpinBox{background:white;border:1px solid #cccac4;padding:4px}")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"实验与模型  ·  {project.name}"))
        form = QFormLayout()
        self.trials = QSpinBox()
        self.trials.setRange(1, 100)
        self.trials.setValue(20)
        self.durations = {}
        form.addRow("每侧试次", self.trials)
        for key, label, default in (("prepare", "准备（秒）", 2), ("cue", "提示（秒）", 1),
                                    ("imagery", "想象（秒）", 4), ("rest", "休息（秒）", 2)):
            control = QDoubleSpinBox()
            control.setRange(0.1, 60)
            control.setValue(default)
            control.setSingleStep(0.5)
            form.addRow(label, control)
            self.durations[key] = control
        layout.addLayout(form)
        row = QHBoxLayout()
        self.start_button = QPushButton("开始实验")
        self.start_button.clicked.connect(self.start_protocol)
        self.stop_button = QPushButton("停止实验")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_protocol)
        row.addWidget(self.start_button)
        row.addWidget(self.stop_button)
        layout.addLayout(row)
        self.cue = QLabel("等待实验")
        self.cue.setAlignment(Qt.AlignCenter)
        self.cue.setStyleSheet("font-size:36px;font-weight:600;min-height:75px;background:white")
        layout.addWidget(self.cue)
        layout.addWidget(QLabel("离线训练"))
        row = QHBoxLayout()
        self.recording = QComboBox()
        self.recording.addItem("选择项目记录", None)
        for item in project.db.execute("SELECT id, source_kind FROM recordings WHERE status = 'complete' ORDER BY created_at"):
            self.recording.addItem(f"{item['source_kind']} · {item['id']}", item["id"])
        self.algorithm = QComboBox()
        self.algorithm.addItems(["FBCSP+LDA", "EEGNet"])
        row.addWidget(self.recording, 2)
        row.addWidget(self.algorithm, 1)
        layout.addLayout(row)
        row = QHBoxLayout()
        self.train_button = QPushButton("训练所选记录")
        self.train_button.clicked.connect(self.train_recording)
        self.benchmark_button = QPushButton("载入 IV-2a 文件并训练")
        self.benchmark_button.clicked.connect(self.train_iv2a)
        row.addWidget(self.train_button)
        row.addWidget(self.benchmark_button)
        layout.addLayout(row)
        self.result = QLabel("尚无本次训练结果")
        layout.addWidget(self.result)
        layout.addWidget(QLabel("模型版本"))
        row = QHBoxLayout()
        self.models = QComboBox()
        self.refresh_models()
        activate = QPushButton("激活所选模型")
        activate.clicked.connect(self.activate_model)
        report = QPushButton("生成报告")
        report.clicked.connect(self.save_report)
        row.addWidget(self.models, 2)
        row.addWidget(activate)
        row.addWidget(report)
        layout.addLayout(row)
        self.timer = QTimer(self)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self.advance_protocol)

    def start_protocol(self):
        config = ProtocolConfig(self.trials.value(), *(self.durations[key].value() for key in
                                ("prepare", "cue", "imagery", "rest")))
        self.schedule = list(config.schedule())
        self.config = asdict(config)
        self.project.set_config({"research_protocol": self.config})
        self.session_id = uuid.uuid4().hex[:12]
        self.position = 0
        self.started_at = time.monotonic()
        self.pending_cues = []
        if self.monitor and self.monitor.worker and self.monitor.worker.isRunning():
            source_kind = "calibration_" + type(self.monitor.worker.source).__name__.lower()
            self.capture = self.project.begin_live_recording(source_kind, self.session_id)
            self.result.setText("校准记录中")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.train_button.setEnabled(False)
        self.benchmark_button.setEnabled(False)
        self.advance_protocol()
        self.timer.start()

    def advance_protocol(self):
        elapsed = time.monotonic() - self.started_at
        while self.position < len(self.schedule) and elapsed >= self.schedule[self.position][0]:
            _, code, _ = self.schedule[self.position]
            self.project.record_protocol_event(self.session_id, elapsed, code, self.config)
            if code in {"left", "right"} and self.capture:
                self.pending_cues.append(code)
            self.position += 1
        if self.position == len(self.schedule):
            self.timer.stop()
            self.cue.setText("实验结束")
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self._finish_capture(completed=True)
            self.train_button.setEnabled(True)
            self.benchmark_button.setEnabled(True)
        else:
            onset, code, duration = self.schedule[self.position - 1]
            names = {"prepare": "准备", "left": "左手", "right": "右手", "imagery_left": "想象左手",
                     "imagery_right": "想象右手", "rest": "休息"}
            self.cue.setText(f"{names[code]}  {max(0, duration - (elapsed - onset)):.1f}s")

    def stop_protocol(self):
        if self.timer.isActive():
            self.timer.stop()
            self.project.record_protocol_event(self.session_id, time.monotonic() - self.started_at,
                                               "stop", self.config)
        self._finish_capture(completed=False)
        self.cue.setText("实验已停止")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        if not self.worker or not self.worker.isRunning():
            self.train_button.setEnabled(True)
            self.benchmark_button.setEnabled(True)

    def capture_block(self, block):
        if not self.capture:
            return
        cues = tuple(EEGEvent(min(index, len(block.samples) - 1), code,
                              block.timestamps[min(index, len(block.samples) - 1)])
                     for index, code in enumerate(self.pending_cues))
        self.pending_cues.clear()
        try:
            self.capture.append(replace(block, events=block.events + cues))
        except Exception as exc:
            self.result.setText("校准记录故障：" + str(exc))
            self.stop_protocol()

    def _finish_capture(self, completed):
        if not self.capture:
            return
        capture, self.capture = self.capture, None
        identifier = capture.finish(completed=completed)
        if identifier:
            if completed:
                self.recording.addItem(f"校准 · {identifier}", identifier)
                self.recording.setCurrentIndex(self.recording.count() - 1)
            self.result.setText(f"校准记录{'已保存' if completed else '已中断'}：{identifier}")

    def train_recording(self):
        if self.recording.currentData() is None:
            QMessageBox.warning(self, "需要数据", "请选择带有左右手事件的项目记录")
            return
        self._train(("recording", self.recording.currentData()))

    def train_iv2a(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择 IV-2a 训练文件", "", "IV-2a (*.mat *.gdf)")
        if path:
            self._train(("iv2a", path))

    def _train(self, source):
        if self.worker and self.worker.isRunning():
            return
        if not any("research_training" in revision for revision in self.project.config_history()):
            self.project.set_config({"research_training": asdict(project_training_config(self.project))})
        self.worker = TrainingWorker(self.project.root, source, self.algorithm.currentText())
        self.worker.succeeded.connect(self._trained)
        self.worker.failed.connect(self._failed)
        self.worker.start()
        self.train_button.setEnabled(False)
        self.benchmark_button.setEnabled(False)
        self.start_button.setEnabled(False)
        self.result.setText("训练中…")

    def _trained(self, identifier):
        self.train_button.setEnabled(True)
        self.benchmark_button.setEnabled(True)
        self.start_button.setEnabled(True)
        self.refresh_models()
        model = next(model for model in self.project.models() if model["id"] == identifier)
        self.result.setText(f"{model['algorithm']} · {identifier} · 准确率 {model['metrics']['accuracy']:.1%}")

    def _failed(self, message):
        self.train_button.setEnabled(True)
        self.benchmark_button.setEnabled(True)
        self.start_button.setEnabled(True)
        self.result.setText("训练失败：" + message)

    def refresh_models(self):
        self.models.clear()
        for model in self.project.models():
            self.models.addItem(f"{model['algorithm']} · {model['id']} · {model['metrics']['accuracy']:.1%}", model["id"])

    def activate_model(self):
        identifier = self.models.currentData()
        if not identifier:
            return
        model = next(item for item in self.project.models() if item["id"] == identifier)
        if QMessageBox.question(self, "确认激活", f"激活模型 {identifier}？") != QMessageBox.Yes:
            return
        try:
            self.project.activate_model(identifier, self.project.subject, model["provenance"]["channels"],
                                        model["provenance"]["sampling_rate"])
            self.result.setText(f"已激活 {identifier}")
        except Exception as exc:
            QMessageBox.warning(self, "激活失败", str(exc))

    def save_report(self):
        path, _ = QFileDialog.getSaveFileName(self, "保存实验报告", str(self.project.root / "reports" / "report.html"), "HTML (*.html)")
        if path:
            generate_report(self.project, path)
            self.result.setText("报告已生成：" + path)

    def closeEvent(self, event):
        self.stop_protocol()
        if self.worker and self.worker.isRunning():
            self.worker.wait()
        super().closeEvent(event)
