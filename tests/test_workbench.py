import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QTWEBENGINE_DISABLE_SANDBOX", "1")

from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication

from mi_bci.workbench import Workbench, configure_font, prototype_dir


class WorkbenchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        configure_font(cls.app)
        cls.window = Workbench()
        cls.window.show()
        cls.loaded = False
        loop = QEventLoop()
        cls.window.view.loadFinished.connect(lambda ok: (setattr(cls, "loaded", ok), loop.quit()))
        QTimer.singleShot(15000, loop.quit)
        loop.exec()

    @classmethod
    def tearDownClass(cls):
        cls.window.close()

    def js(self, expression):
        loop = QEventLoop()
        result = []
        self.window.view.page().runJavaScript(expression, lambda value: (result.append(value), loop.quit()))
        QTimer.singleShot(5000, loop.quit)
        loop.exec()
        self.assertTrue(result, "JavaScript did not return")
        return result[0]

    def test_current_prototype_loaded_with_font_and_six_pages(self):
        self.assertTrue(self.loaded)
        self.assertTrue((prototype_dir() / "warm-theme.css").is_file())
        self.assertEqual(self.js("document.querySelectorAll('#nav button').length"), 6)
        self.assertEqual(self.js("document.querySelectorAll('.view').length"), 6)
        self.assertIn("项目", self.js("document.querySelector('.view.active h1').textContent"))
        self.assertIn("Microsoft YaHei", self.js("getComputedStyle(document.body).fontFamily"))
        self.assertEqual(self.js("getComputedStyle(document.body).backgroundColor"), "rgb(250, 248, 245)")
        self.assertTrue(self.js("document.querySelector('#project-folder').value.startsWith('D:')"))
        for view in ("device", "acquisition", "training", "online", "ai", "project"):
            self.js(f"document.querySelector('[data-view={view}]').click()")
            self.assertEqual(self.js("document.querySelector('.view.active').id"), f"view-{view}")

    def test_project_creation_and_global_stop_use_backend(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parents[1] / ".scratch") as folder:
            try:
                self.js("window.__errors=[]; window.addEventListener('error', e => window.__errors.push(e.message));"
                        "window.addEventListener('unhandledrejection', e => window.__errors.push(e.reason.stack || String(e.reason)));"
                        "document.querySelector('#project-name').value='研究验证';"
                        "document.querySelector('#subject-id').value='sub001';"
                        "document.querySelector('#project-folder').value=" + json.dumps(folder) + ";"
                        "document.querySelector('#create-project').click()")
                loop = QEventLoop()
                QTimer.singleShot(500, loop.quit)
                loop.exec()
                path = Path(folder) / "project-sub001"
                self.assertTrue((path / "metadata.sqlite").is_file())
                self.assertEqual(json.loads(self.window.bridge.snapshot())["project"], "研究验证")
                self.assertEqual(self.js("document.querySelector('.view.active').id"), "view-device",
                                 self.js("JSON.stringify(window.__errors)"))
                self.js("document.querySelector('#global-stop').click()")
                loop = QEventLoop()
                QTimer.singleShot(150, loop.quit)
                loop.exec()
                self.assertFalse(json.loads(self.window.bridge.snapshot())["online_running"])
            finally:
                if self.window.monitor.project:
                    self.window.monitor.project.close()
                    self.window.monitor.project = None


if __name__ == "__main__":
    unittest.main()
