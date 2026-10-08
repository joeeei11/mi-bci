"""Frozen Windows desktop entry point."""

import os
import sys


def main():
    if "--self-test" in sys.argv:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--disable-gpu --disable-gpu-compositing"
        from PySide6.QtCore import QEventLoop, QTimer
        from PySide6.QtWidgets import QApplication
        from mi_bci.workbench import Workbench, configure_font

        app = QApplication.instance() or QApplication([])
        configure_font(app)
        window = Workbench()
        loop = QEventLoop()
        loaded = []
        window.view.loadFinished.connect(lambda ok: (loaded.append(ok), loop.quit()))
        window.show()
        QTimer.singleShot(15000, loop.quit)
        loop.exec()
        if not window.isVisible() or loaded != [True]:
            window.close()
            return 1
        content = []
        window.view.page().runJavaScript(
            "document.querySelectorAll('#nav button').length === 6 && "
            "getComputedStyle(document.body).backgroundColor === 'rgb(250, 248, 245)'",
            lambda value: (content.append(value), loop.quit()))
        QTimer.singleShot(5000, loop.quit)
        loop.exec()
        window.close()
        return 0 if content == [True] else 1

    from mi_bci.workbench import main as desktop_main

    return desktop_main()


if __name__ == "__main__":
    raise SystemExit(main())
