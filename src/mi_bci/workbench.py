"""Desktop shell rendered from the approved interactive prototype."""

import json
import re
import sys
from pathlib import Path

from PySide6.QtCore import QObject, QUrl, Slot
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineCore import QWebEngineProfile, QWebEngineSettings
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QApplication, QFileDialog, QMainWindow

from .desktop import Monitor
from .project import Project


def prototype_dir():
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / "prototype"
    return Path(__file__).resolve().parents[2] / ".scratch/mi-bci-research-platform/prototype"


def data_root():
    drive = Path("D:/MI-BCI")
    if drive.drive and Path(drive.drive + "/").exists():
        return drive
    return Path.home() / "Documents/MI-BCI"


def configure_font(app):
    families = set(QFontDatabase.families())
    for name in ("Microsoft YaHei UI", "Microsoft YaHei", "Noto Sans CJK SC", "SimHei"):
        if name in families:
            app.setFont(QFont(name, 10))
            return name
    return app.font().family()


class Bridge(QObject):
    def __init__(self, monitor):
        super().__init__()
        self.monitor = monitor
        self.source_confirmed = False

    def _reply(self, **items):
        return json.dumps(items, ensure_ascii=False)

    @Slot(str, str, str, result=str)
    def create_project(self, name, subject, folder):
        try:
            if not name.strip() or not re.fullmatch(r"[A-Za-z0-9_-]+", subject.strip()):
                raise ValueError("请输入项目名称和英数字受试者代号")
            parent = Path(folder).expanduser()
            parent.mkdir(parents=True, exist_ok=True)
            slug = re.sub(r"[^A-Za-z0-9_-]", "-", name.strip()).strip("-") or "project"
            path = parent / f"{slug}-{subject.strip()}"
            if path.exists():
                raise FileExistsError(f"项目目录已存在：{path}")
            project = Project.create(path, name.strip(), subject.strip())
            project.close()
            self.monitor.load_project(path)
            return self._reply(ok=True, path=str(path))
        except Exception as exc:
            return self._reply(ok=False, error=str(exc))

    @Slot(result=str)
    def open_project(self):
        path = QFileDialog.getExistingDirectory(None, "打开项目", str(data_root() / "projects"))
        if not path:
            return self._reply(ok=False, cancelled=True)
        self.monitor.load_project(path)
        if not self.monitor.project or self.monitor.project.root != Path(path):
            return self._reply(ok=False, error="项目打开失败；请检查采集日志")
        return self._reply(ok=True)

    @Slot(str, str, result=str)
    def configure_source(self, kind, target):
        choices = {"synthetic": "模拟", "replay": "项目回放", "file": "CSV / EDF / BDF",
                   "lsl": "LSL", "serial": "串口", "tcp": "TCP", "bluetooth": "蓝牙串口"}
        if kind not in choices:
            return self._reply(ok=False, error="请选择有效的脑电来源")
        self.monitor.kind.setCurrentText(choices[kind])
        self.monitor.target.setText(target)
        self.source_confirmed = True
        return self._reply(ok=True)

    @Slot(result=str)
    def start_acquisition(self):
        if not self.source_confirmed:
            return self._reply(ok=False, error="请先在设备连接页确认信号源")
        self.monitor.start_source()
        return self._reply(ok=bool(self.monitor.worker and self.monitor.worker.isRunning()),
                           error=self.monitor.state_label.text())

    @Slot(result=str)
    def stop_acquisition(self):
        self.monitor.stop_source()
        return self._reply(ok=True)

    @Slot(result=str)
    def open_training(self):
        if not self.monitor.project:
            return self._reply(ok=False, error="请先创建或打开项目")
        self.monitor.open_research()
        return self._reply(ok=True)

    @Slot(str, str, result=str)
    def start_online(self, kind, target):
        if not self.monitor.project:
            return self._reply(ok=False, error="请先创建或打开项目")
        from .online_desktop import OnlineWindow
        if self.monitor.online_window is None:
            self.monitor.online_window = OnlineWindow(self.monitor)
        choices = {"virtual": "虚拟 STM32", "usb": "USB 串口", "bluetooth": "蓝牙串口", "tcp": "TCP / Wi-Fi"}
        if kind not in choices:
            return self._reply(ok=False, error="请选择有效的控制端点")
        self.monitor.online_window.transport.setCurrentText(choices[kind])
        self.monitor.online_window.target.setText(target)
        self.monitor.online_window.start()
        worker = self.monitor.online_window.worker
        return self._reply(ok=bool(worker and worker.isRunning()))

    @Slot(result=str)
    def reset_fault(self):
        online = self.monitor.online_window
        if not online or not online.fault_latched:
            return self._reply(ok=False, error="没有待复位的在线故障")
        online.reset_fault()
        return self._reply(ok=not online.fault_latched)

    @Slot(result=str)
    def stop_all(self):
        online = self.monitor.online_window
        if online:
            online.stop()
        self.monitor.stop_source()
        return self.snapshot()

    @Slot(result=str)
    def open_ai(self):
        if not self.monitor.project:
            return self._reply(ok=False, error="请先创建或打开项目")
        self.monitor.open_ai()
        return self._reply(ok=True)

    @Slot(result=str)
    def snapshot(self):
        monitor = self.monitor
        project = monitor.project
        online = monitor.online_window
        model = project.active_model() if project else None
        models = project.models() if project else []
        worker = online.worker if online else None
        return self._reply(
            ok=True, project=project.name if project else "", subject=project.subject if project else "",
            project_path=str(project.root) if project else "", source=monitor.kind.currentText(),
            source_confirmed=self.source_confirmed, acquisition=monitor.state_label.text(),
            acquiring=bool(monitor.worker and monitor.worker.isRunning()),
            channels=monitor.channel_label.text(), metrics=monitor.metrics_label.text(),
            logs=monitor.log.toPlainText().splitlines()[-15:],
            recordings=monitor.recording.count(), models=[{"id": item["id"], "algorithm": item["algorithm"],
                "accuracy": item["metrics"].get("accuracy")} for item in models],
            active_model=model["id"] if model else "", online_state=online.summary.text() if online else "空闲 · 无设备输出",
            online_running=bool(worker and worker.isRunning()), fault_latched=bool(online and online.fault_latched),
            stop_confirmed=bool(worker and not worker.isRunning() and "STOP 已确认" in online.summary.text()) if online else False,
            confidence=online.confidence.text() if online else "判断：--",
            command=online.command.text() if online else "指令：STOP · 未连接",
            events=online.log.toPlainText().splitlines()[-15:] if online else [],
            samples=[list(channel)[-250::2] for channel in monitor.samples[:2]])


class Workbench(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MI-BCI 研究工作台")
        self.resize(1440, 900)
        self.monitor = Monitor()
        self.bridge = Bridge(self.monitor)
        self.channel = QWebChannel(self)
        self.channel.registerObject("backend", self.bridge)
        self.profile = QWebEngineProfile("mi-bci", self)
        storage = data_root() / "web-profile"
        storage.mkdir(parents=True, exist_ok=True)
        self.profile.setCachePath(str(storage / "cache"))
        self.profile.setPersistentStoragePath(str(storage / "storage"))
        from PySide6.QtWebEngineCore import QWebEnginePage
        self.view = QWebEngineView(self)
        self.view.setPage(QWebEnginePage(self.profile, self.view))
        self.view.page().setWebChannel(self.channel)
        self.view.settings().setFontFamily(QWebEngineSettings.StandardFont, QApplication.font().family())
        self.setCentralWidget(self.view)
        source = prototype_dir()
        html = (source / "index.html").read_text(encoding="utf-8")
        html = html.replace('<script src="app.js"></script>',
                            '<script src="qrc:///qtwebchannel/qwebchannel.js"></script>\n'
                            '<script src="workbench_runtime.js"></script>')
        self.view.setHtml(html, QUrl.fromLocalFile(str(source / "index.html")))

    def closeEvent(self, event):
        self.monitor.close()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    configure_font(app)
    window = Workbench()
    window.show()
    return app.exec()
