import io
import importlib.util
import json
import socket
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mi_bci import (BluetoothSerialSource, FileSource, JsonLineSource, LSLSource,
                    Project, ReplaySource, SerialSource, StreamState, TCPSource, import_file)


def drain(source):
    source.start()
    result = []
    while (block := source.read()) is not None:
        result.append(block)
    return result


def packet(index, **extra):
    return json.dumps({"channels": ["C3", "C4"], "sampling_rate": 250,
                       "timestamp": index / 250, "sample": [index, -index], **extra}) + "\n"


class AcquisitionTests(unittest.TestCase):
    def test_csv_import_preserves_source_events_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "signal.csv"
            path.write_text("timestamp,C3,C4,event\n0,1,-1,left\n0.004,2,-2,\n0.008,3,-3,right\n", encoding="utf-8")
            project = Project.create(Path(directory) / "project", "Study")
            identifier = import_file(project, path, block_size=2)
            blocks = drain(ReplaySource(project, identifier))
            self.assertEqual([event.code for block in blocks for event in block.events], ["left", "right"])
            self.assertEqual([sample for block in blocks for sample in block.samples], [(1, -1), (2, -2), (3, -3)])
            row = project.db.execute("SELECT source_kind, source_path, source_sha256 FROM recordings").fetchone()
            self.assertEqual(row["source_kind"], "csv")
            self.assertEqual((project.root / row["source_path"]).read_bytes(), path.read_bytes())
            self.assertEqual(len(row["source_sha256"]), 64)
            self.assertTrue(next((project.root / "derivatives").rglob("*_eeg.json")).is_file())
            project.close()

    def test_csv_rejects_missing_rate_and_bad_timestamps(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            path.write_text("C3\n1\n2\n", encoding="utf-8")
            source = FileSource(path)
            source.start()
            with self.assertRaises(ValueError):
                source.read()
            self.assertEqual(source.state, StreamState.FAULT)
            path.write_text("timestamp,C3\n1,1\n1,2\n", encoding="utf-8")
            source = FileSource(path)
            source.start()
            with self.assertRaises(ValueError):
                source.read()

    def test_json_line_conformance_and_disconnect(self):
        source = JsonLineSource(lambda: io.BytesIO((packet(0, event="left") + packet(1)).encode()), block_size=2)
        source.start()
        block = source.read()
        self.assertEqual(block.channels[0].name, "C3")
        self.assertEqual(block.events[0].code, "left")
        with self.assertRaises(ConnectionError):
            source.read()
        self.assertEqual(source.state, StreamState.FAULT)

    def test_json_line_rejects_malformed_and_nonmonotonic_packets(self):
        for contents in ("not-json\n", packet(0) + packet(0),
                         json.dumps({"timestamp": 0, "sample": [1]}) + "\n"):
            with self.subTest(contents=contents):
                source = JsonLineSource(lambda: io.BytesIO(contents.encode()), block_size=2 if contents.count("\n") > 1 else 1)
                source.start()
                with self.assertRaises((ValueError, KeyError, json.JSONDecodeError)):
                    source.read()
                self.assertEqual(source.state, StreamState.FAULT)

    def test_tcp_adapter_uses_same_contract(self):
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        def serve():
            connection, _ = listener.accept()
            with connection:
                connection.sendall((packet(0) + packet(1)).encode())
            listener.close()
        thread = threading.Thread(target=serve)
        thread.start()
        source = TCPSource("127.0.0.1", port, block_size=2)
        source.start()
        self.assertEqual(source.read().samples, ((0.0, 0.0), (1.0, -1.0)))
        with self.assertRaises(ConnectionError):
            source.read()
        thread.join()

    def test_serial_and_bluetooth_share_line_protocol(self):
        self.assertTrue(issubclass(BluetoothSerialSource, SerialSource))
        self.assertTrue(issubclass(SerialSource, JsonLineSource))
        self.assertTrue(hasattr(LSLSource, "read"))

    def test_serial_and_bluetooth_read_and_fault_without_hardware(self):
        fake = SimpleNamespace(Serial=lambda *args, **kwargs: io.BytesIO(packet(0).encode()))
        with patch.dict("sys.modules", {"serial": fake}):
            for cls in (SerialSource, BluetoothSerialSource):
                source = cls("COM99", block_size=1)
                source.start()
                self.assertEqual(source.read().channels[0].name, "C3")
                with self.assertRaises(ConnectionError):
                    source.read()
                self.assertEqual(source.state, StreamState.FAULT)

    def test_lsl_metadata_samples_and_timeout_without_hardware(self):
        class Node:
            def child(self, name):
                return self
            def child_value(self, name):
                return "C3"
            def next_sibling(self):
                return self
        info = SimpleNamespace(nominal_srate=lambda: 250, channel_count=lambda: 1, desc=Node)
        class Inlet:
            def __init__(self, *args, **kwargs):
                self.calls = 0
            def pull_chunk(self, **kwargs):
                self.calls += 1
                return ([[1.0]], [0.1]) if self.calls == 1 else ([], [])
            def close_stream(self):
                pass
        fake = SimpleNamespace(resolve_byprop=lambda *args, **kwargs: [info], StreamInlet=Inlet)
        with patch.dict("sys.modules", {"pylsl": fake}):
            source = LSLSource(timeout=0.01)
            source.start()
            self.assertEqual(source.read().samples, ((1.0,),))
            with self.assertRaises(ConnectionError):
                source.read()
            self.assertEqual(source.state, StreamState.FAULT)

    def test_lsl_rejects_missing_rate_and_malformed_samples(self):
        class Node:
            def child(self, name):
                return self
            def child_value(self, name):
                return "C3"
            def next_sibling(self):
                return self
        class Inlet:
            def __init__(self, *args, **kwargs):
                pass
            def pull_chunk(self, **kwargs):
                return ([[float("nan")]], [0.1])
            def close_stream(self):
                pass
        rate = [0]
        info = SimpleNamespace(nominal_srate=lambda: rate[0], channel_count=lambda: 1, desc=Node)
        fake = SimpleNamespace(resolve_byprop=lambda *args, **kwargs: [info], StreamInlet=Inlet)
        with patch.dict("sys.modules", {"pylsl": fake}):
            source = LSLSource()
            with self.assertRaises(ValueError):
                source.start()
            self.assertEqual(source.state, StreamState.FAULT)
            rate[0] = 250
            source.start()
            with self.assertRaises(ValueError):
                source.read()
            self.assertEqual(source.state, StreamState.FAULT)

    def test_edf_and_bdf_use_matching_mne_readers(self):
        class Raw:
            info = {"sfreq": 250}
            ch_names = ["C3"]
            n_times = 2
            annotations = SimpleNamespace(onset=[0.004], description=["left"])
            def get_channel_types(self, picks):
                return ["eeg"]
            def time_as_index(self, onset):
                return [1]
            def get_data(self, start, stop):
                return [[1e-6, 2e-6][start:stop]]
            def close(self):
                pass
        calls = []
        def reader(kind):
            def read(path, **kwargs):
                calls.append(kind)
                return Raw()
            return read
        fake = SimpleNamespace(io=SimpleNamespace(read_raw_edf=reader("edf"), read_raw_bdf=reader("bdf")))
        with patch.dict("sys.modules", {"mne": fake}):
            for suffix in ("edf", "bdf"):
                blocks = drain(FileSource(f"record.{suffix}"))
                self.assertEqual(blocks[0].samples, ((1.0,), (2.0,)))
                self.assertEqual(blocks[0].events[0].code, "left")
        self.assertEqual(calls, ["edf", "bdf"])

    @unittest.skipUnless(importlib.util.find_spec("edfio") and importlib.util.find_spec("mne"),
                         "optional EDF/BDF integration packages are not installed")
    def test_real_edf_and_bdf_roundtrip(self):
        import edfio
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            values = np.full(250, 5.0)
            for suffix, document, signal in (("edf", edfio.Edf, edfio.EdfSignal),
                                             ("bdf", edfio.Bdf, edfio.BdfSignal)):
                path = Path(directory) / f"record.{suffix}"
                document([signal(values, 250, label="C3", physical_dimension="uV")],
                         annotations=[edfio.EdfAnnotation(0.2, 0, "left")]).write(path)
                blocks = drain(FileSource(path, block_size=100))
                self.assertEqual(len(blocks), 3)
                self.assertAlmostEqual(blocks[0].samples[0][0], 5.0, places=2)
                self.assertEqual([e.code for b in blocks for e in b.events], ["left"])


if __name__ == "__main__":
    unittest.main()
