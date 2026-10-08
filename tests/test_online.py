import tempfile
import unittest
import hashlib
import socket
import threading
from unittest.mock import patch
from pathlib import Path

from mi_bci import Channel, EEGBlock, Project
from mi_bci.online import LineTransport, ModelPredictor, OnlineConfig, OnlineSession, VirtualSTM32
from mi_bci.stream import StreamHealth


class Predictor:
    def __init__(self, predictions):
        self.predictions = iter(predictions)

    def predict(self, window):
        return next(self.predictions)


class OnlineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.project = Project.create(Path(self.directory.name) / "study", "Study")
        artifact = self.project.root / "model.bin"
        artifact.write_bytes(b"test")
        model_id = self.project.register_model("test", "01", None, "model.bin",
                                               {"channels": ["C3", "C4"], "sampling_rate": 100,
                                                "artifact_sha256": hashlib.sha256(b"test").hexdigest()}, {})
        self.project.activate_model(model_id, "01", ["C3", "C4"], 100)
        self.clock = [0.0]
        self.device = VirtualSTM32()

    def tearDown(self):
        self.project.close()
        self.directory.cleanup()

    def session(self, predictions, **config):
        return OnlineSession(self.project, self.device,
                             predictor_factory=lambda *args: Predictor(predictions),
                             config=OnlineConfig(**config), clock=lambda: self.clock[0])

    def feed(self, session, count, start=0, health=StreamHealth.GOOD):
        for offset in range(0, count, 25):
            index = start + offset
            session.feed(EEGBlock(tuple((0.1, 0.2) for _ in range(25)),
                                  tuple((index + point) / 100 for point in range(25)),
                                  (Channel("C3"), Channel("C4")), 100, health=health))
            self.clock[0] += 0.25

    def test_three_predictions_cooldown_and_low_confidence_stop(self):
        session = self.session([("LEFT", .9), ("LEFT", .91), ("LEFT", .92),
                                ("RIGHT", .9), ("RIGHT", .9), ("RIGHT", .9), ("LEFT", .5)],
                               cooldown_seconds=.5)
        session.start(["C3", "C4"], 100)
        self.feed(session, 150)
        self.assertEqual(self.device.history, ["STOP", "LEFT"])
        self.feed(session, 75, start=150)
        self.assertEqual(self.device.history, ["STOP", "LEFT", "RIGHT"])
        self.feed(session, 25, start=225)
        self.assertEqual(self.device.history[-1], "STOP")
        self.assertEqual(session.streak, 0)
        self.assertEqual(len([event for event in session.events if event["kind"] == "prediction"]), 7)
        self.assertTrue(all("confidence" in event for event in session.events if event["kind"] == "prediction"))

    def test_timeout_and_bad_signal_enter_fault_with_stop(self):
        for fault in ("timeout", "health", "gap"):
            self.clock[0] = 0
            self.device = VirtualSTM32()
            session = self.session([("LEFT", .9)] * 4)
            session.start(["C3", "C4"], 100)
            self.feed(session, 150)
            if fault == "timeout":
                self.clock[0] += 2
                session.tick()
            elif fault == "health":
                self.feed(session, 25, start=150, health=StreamHealth.LOST)
            else:
                self.feed(session, 25, start=300)
            self.assertEqual(session.state, "fault")
            self.assertEqual(self.device.command, "STOP")
            self.assertTrue(session.stop_confirmed)

    def test_ack_failure_unconfirmed_stop_and_manual_reset(self):
        session = self.session([("LEFT", .9)] * 4)
        session.start(["C3", "C4"], 100)
        self.device.ack_enabled = False
        self.feed(session, 150)
        self.assertEqual(session.state, "fault")
        self.assertFalse(session.stop_confirmed)
        session.stop()
        self.assertEqual(session.state, "fault")
        self.device.ack_enabled = True
        session.reset()
        self.assertEqual(session.state, "idle")

    def test_duplicate_reconnect_and_idempotent_stop(self):
        self.device.connect()
        self.device.exchange(3, "LEFT", .3)
        self.device.exchange(3, "LEFT", .3)
        self.assertEqual(self.device.history, ["LEFT"])
        with self.assertRaises(ValueError):
            self.device.exchange(3, "RIGHT", .3)
        self.device.close()
        with self.assertRaises(ConnectionError):
            self.device.exchange(4, "RIGHT", .3)
        self.device.connect()
        self.assertEqual(self.device.command, "STOP")
        session = self.session([])
        session.start(["C3", "C4"], 100)
        session.stop()
        count = len(self.device.history)
        session.stop()
        self.assertEqual(len(self.device.history), count)

    def test_tcp_gateway_requires_matching_ack_and_times_out(self):
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]

        def device():
            connection, _ = listener.accept()
            with connection:
                frame = connection.recv(80)
                self.assertEqual(frame, b"CMD 1 LEFT\n")
                connection.sendall(b"ACK 8 LEFT\nACK 1 LEFT\n")
                self.assertEqual(connection.recv(80), b"CMD 2 STOP\n")
            listener.close()

        thread = threading.Thread(target=device)
        thread.start()
        transport = LineTransport("tcp", f"127.0.0.1:{port}")
        try:
            transport.connect()
            self.assertTrue(transport.exchange(1, "LEFT", .3))
            with self.assertRaises(ConnectionError):
                transport.exchange(2, "STOP", .3)
        finally:
            transport.close()
            thread.join(2)
        self.assertFalse(thread.is_alive())

    def test_usb_and_bluetooth_serial_share_ack_protocol(self):
        class FakeSerial:
            def __init__(self, *args, **kwargs):
                self.writes = []
                self.timeout = .3

            def write(self, frame):
                self.writes.append(frame)

            def readline(self):
                return b"ACK 1 STOP\n"

            def close(self):
                pass

        import sys
        from types import SimpleNamespace
        with patch.dict(sys.modules, {"serial": SimpleNamespace(Serial=FakeSerial)}):
            for kind in ("usb", "bluetooth"):
                transport = LineTransport(kind, "COM3")
                transport.connect()
                self.assertTrue(transport.exchange(1, "STOP", .3))
                self.assertEqual(transport.connection.writes, [b"CMD 1 STOP\n"])
                with self.assertRaises(TimeoutError):
                    transport.exchange(2, "LEFT", .01)
                transport.close()

    def test_saved_fbcsp_model_produces_finite_online_confidence(self):
        import numpy as np

        path = self.project.root / "fbcsp.npz"
        np.savez(path, bands=np.array([[8, 12]]), filters=np.array([[[1., 0.], [0., 1.]]]),
                 selected_features=np.array([0, 1]), coef=np.array([[1., -1.]]),
                 intercept=np.array([0.]))
        provenance = {"channels": ["C3", "C4"], "sampling_rate": 100,
                      "preprocessing": {"bandpass_hz": [8, 30]},
                      "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        model = {"subject": "01", "algorithm": "FBCSP+LDA", "artifact": "fbcsp.npz",
                 "provenance": provenance}
        predictor = ModelPredictor(self.project, model, ["C3", "C4"], 100, 100)
        window = [(float(np.sin(index * .6)), float(np.sin(index * .9))) for index in range(100)]
        label, confidence = predictor.predict(window)
        self.assertIn(label, {"LEFT", "RIGHT"})
        self.assertTrue(.5 <= confidence <= 1)


if __name__ == "__main__":
    unittest.main()
