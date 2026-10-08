import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtWidgets import QApplication
    from mi_bci.desktop import AcquisitionWorker, Monitor
    from mi_bci.research_desktop import ResearchWindow
    from mi_bci.online_desktop import OnlineWindow
    from PySide6.QtWidgets import QMessageBox
except ImportError:
    QApplication = None

from mi_bci import Project, ReplaySource, SyntheticSource


@unittest.skipIf(QApplication is None, "optional PySide6 desktop dependency is not installed")
class DesktopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_worker_updates_monitor_and_renders_signal(self):
        window = Monitor()
        worker = AcquisitionWorker(SyntheticSource(total_samples=100, block_size=25), paced=False)
        worker.block.connect(window._block)
        worker.start()
        deadline = time.monotonic() + 5
        while (worker.isRunning() or not window.samples) and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.01)
        worker.wait(1000)
        self.app.processEvents()
        self.assertFalse(worker.isRunning())
        self.assertIn("C3 接收中", window.channel_label.text())
        self.assertIn("250 Hz", window.metrics_label.text())
        self.assertEqual(len(window.samples[0]), 100)
        window.show()
        self.app.processEvents()
        image = window.wave.grab().toImage()
        self.assertFalse(image.isNull())
        self.assertNotEqual(image.pixelColor(image.width() // 2, image.height() // 2).name(), "#faf8f5")
        window._fault("source disconnected")
        self.assertIn("波形已冻结", window.state_label.text())
        self.assertIn("C3 失联", window.channel_label.text())
        self.assertIn("健康：lost", window.metrics_label.text())
        window.close()

    def test_replay_worker_opens_its_own_project_connection(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Project.create(Path(directory) / "project", "Study")
            source = SyntheticSource(total_samples=20, block_size=20)
            source.start()
            identifier = project.record("synthetic", [source.read()])
            window = Monitor()
            worker = AcquisitionWorker(ReplaySource(project, identifier))
            worker.block.connect(window._block)
            worker.start()
            deadline = time.monotonic() + 5
            while (worker.isRunning() or not window.samples) and time.monotonic() < deadline:
                self.app.processEvents()
                time.sleep(0.01)
            worker.wait(1000)
            self.app.processEvents()
            self.assertEqual(len(window.samples[0]), 20)
            project.close()
            window.close()

    def test_guided_protocol_records_cues_and_rest(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Project.create(Path(directory) / "project", "Study")
            window = ResearchWindow(project)
            window.trials.setValue(1)
            for control in window.durations.values():
                control.setValue(0.1)
            window.start_protocol()
            window.started_at -= 2
            window.advance_protocol()
            codes = [event["code"] for event in project.protocol_events()]
            self.assertEqual(codes.count("left"), 1)
            self.assertEqual(codes.count("right"), 1)
            self.assertEqual(codes.count("rest"), 2)
            self.assertEqual(codes[-1], "stop")
            self.assertTrue(any("research_protocol" in revision for revision in project.config_history()))
            window.close()
            project.close()

    def test_active_acquisition_becomes_labeled_calibration_recording(self):
        class RunningWorker:
            source = SyntheticSource()

            def isRunning(self):
                return True

        class MonitorStub:
            worker = RunningWorker()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "project"
            project = Project.create(root, "Study")
            window = ResearchWindow(project, MonitorStub())
            window.trials.setValue(1)
            for control in window.durations.values():
                control.setValue(0.1)
            window.start_protocol()
            source = SyntheticSource(total_samples=200, block_size=25)
            source.start()
            for onset, _, _ in window.schedule[:-1]:
                window.started_at = time.monotonic() - onset - 0.001
                window.advance_protocol()
                window.capture_block(source.read())
            window.started_at = time.monotonic() - window.schedule[-1][0] - 0.001
            window.advance_protocol()
            identifier = window.recording.currentData()
            self.assertIsNotNone(identifier)
            window.close()
            project.close()
            project = Project.open(root)
            codes = [event.code for block in project.blocks(identifier) for event in block.events]
            self.assertEqual(codes.count("left"), 1)
            self.assertEqual(codes.count("right"), 1)
            self.assertEqual(project.db.execute("SELECT session_id FROM calibration_sessions WHERE recording_id = ?",
                                                (identifier,)).fetchone()[0], window.session_id)
            self.assertEqual(project.db.execute("SELECT status FROM recordings WHERE id = ?", (identifier,)).fetchone()[0], "complete")
            project.close()

    def test_stopped_calibration_is_not_offered_for_training(self):
        class RunningWorker:
            source = SyntheticSource()

            def isRunning(self):
                return True

        class MonitorStub:
            worker = RunningWorker()

        with tempfile.TemporaryDirectory() as directory:
            project = Project.create(Path(directory) / "project", "Study")
            window = ResearchWindow(project, MonitorStub())
            window.start_protocol()
            source = SyntheticSource(total_samples=25, block_size=25)
            source.start()
            window.capture_block(source.read())
            window.stop_protocol()
            row = project.db.execute("SELECT id, status FROM recordings").fetchone()
            self.assertEqual(row["status"], "interrupted")
            self.assertEqual(window.recording.count(), 1)
            window.close()
            project.close()

    def test_online_fault_requires_explicit_reset(self):
        monitor = Monitor()
        window = OnlineWindow(monitor)
        window.show()
        self.app.processEvents()
        self.assertFalse(window.grab().toImage().isNull())
        window.update_status({"state": "fault", "reason": "ACK timeout", "stop_confirmed": False})
        self.assertIn("STOP 未确认", window.summary.text())
        self.assertFalse(window.start_button.isEnabled())
        with patch.object(QMessageBox, "question", return_value=QMessageBox.Yes):
            window.reset_fault()
        self.assertTrue(window.start_button.isEnabled())
        window.close()
        monitor.close()


if __name__ == "__main__":
    unittest.main()
