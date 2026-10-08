"""File and live EEG adapters for the normalized stream contract."""

import csv
import json
import socket
from math import isfinite
from pathlib import Path

from .stream import Channel, EEGBlock, EEGEvent, StreamHealth, StreamState


def _channels(names, units=None, kinds=None):
    names = tuple(names)
    if not names or any(not str(name).strip() for name in names):
        raise ValueError("channel metadata is required")
    return tuple(Channel(str(name), str((units or {}).get(name, "uV")),
                         str((kinds or {}).get(name, "EEG"))) for name in names)


class FileSource:
    """Read CSV or EDF/BDF in bounded blocks without changing the original file."""

    def __init__(self, path: str | Path, sampling_rate: float | None = None, block_size: int = 250):
        self.path = Path(path)
        if self.path.suffix.lower() not in {".csv", ".edf", ".bdf"}:
            raise ValueError("supported file formats: CSV, EDF, BDF")
        if block_size <= 0 or (sampling_rate is not None and (not isfinite(sampling_rate) or sampling_rate <= 0)):
            raise ValueError("invalid sampling rate or block size")
        self.sampling_rate = sampling_rate
        self.block_size = block_size
        self.state = StreamState.IDLE
        self._blocks = None

    def start(self):
        if self.state == StreamState.RUNNING:
            raise RuntimeError("source already running")
        try:
            self._blocks = self._read_csv() if self.path.suffix.lower() == ".csv" else self._read_edf()
            self.state = StreamState.RUNNING
        except Exception:
            self.state = StreamState.FAULT
            raise

    def read(self):
        if self.state != StreamState.RUNNING:
            raise RuntimeError("source is not running")
        try:
            block = next(self._blocks, None)
        except Exception:
            self.state = StreamState.FAULT
            self.stop(close_only=True)
            raise
        if block is None:
            self.stop()
        return block

    def stop(self, close_only=False):
        if self._blocks is not None:
            self._blocks.close()
            self._blocks = None
        if not close_only:
            self.state = StreamState.STOPPED

    def _read_csv(self):
        with self.path.open(newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            headers = reader.fieldnames or []
            names = [name for name in headers if name not in {"timestamp", "event"}]
            channels = _channels(names)
            rows = list(reader)
            if not rows:
                raise ValueError("CSV contains no samples")
            if "timestamp" in headers:
                timestamps = [float(row["timestamp"]) for row in rows]
                if any(a >= b for a, b in zip(timestamps, timestamps[1:])):
                    raise ValueError("CSV timestamps must increase strictly")
                rate = self.sampling_rate or (1 / (timestamps[1] - timestamps[0]) if len(rows) > 1 else None)
                if rate is None:
                    raise ValueError("sampling rate is required for a single sample")
            else:
                if self.sampling_rate is None:
                    raise ValueError("CSV without timestamps requires a sampling rate")
                rate = self.sampling_rate
                timestamps = [index / rate for index in range(len(rows))]
            for start in range(0, len(rows), self.block_size):
                chunk = rows[start:start + self.block_size]
                times = tuple(timestamps[start:start + len(chunk)])
                samples = tuple(tuple(float(row[name]) for name in names) for row in chunk)
                events = tuple(EEGEvent(index, row["event"].strip(), times[index])
                               for index, row in enumerate(chunk) if row.get("event", "").strip())
                yield EEGBlock(samples, times, channels, rate, events)

    def _read_edf(self):
        try:
            import mne
        except ImportError as exc:
            raise RuntimeError("EDF/BDF import requires mne; install mi-bci-platform[acquisition]") from exc
        reader = mne.io.read_raw_bdf if self.path.suffix.lower() == ".bdf" else mne.io.read_raw_edf
        raw = reader(self.path, preload=False, verbose="ERROR")
        try:
            rate = float(raw.info["sfreq"])
            names = raw.ch_names
            channels = tuple(Channel(name, "uV" if raw.get_channel_types(picks=[index])[0] == "eeg" else "V",
                                     raw.get_channel_types(picks=[index])[0].upper())
                             for index, name in enumerate(names))
            event_map = {}
            for onset, description in zip(raw.annotations.onset, raw.annotations.description):
                index = raw.time_as_index(onset)[0]
                if 0 <= index < raw.n_times:
                    event_map.setdefault(index, []).append(description)
            for start in range(0, raw.n_times, self.block_size):
                end = min(start + self.block_size, raw.n_times)
                data = raw.get_data(start=start, stop=end)
                samples = tuple(tuple(float(data[ch][index]) * (1e6 if channels[ch].unit == "uV" else 1)
                                      for ch in range(len(channels))) for index in range(end - start))
                times = tuple(float(index / rate) for index in range(start, end))
                events = tuple(EEGEvent(index - start, code, times[index - start])
                               for index in range(start, end) for code in event_map.get(index, []))
                yield EEGBlock(samples, times, channels, rate, events)
        finally:
            raw.close()


class JsonLineSource:
    """One JSON object per line: rate, channels, timestamp and sample."""

    def __init__(self, opener, block_size=50):
        if block_size <= 0:
            raise ValueError("block size must be positive")
        self._opener = opener
        self.block_size = block_size
        self.state = StreamState.IDLE
        self.health = StreamHealth.LOST
        self._connection = None
        self._channels = None
        self._rate = None
        self._last_time = None

    def start(self):
        if self.state == StreamState.RUNNING:
            raise RuntimeError("source already running")
        try:
            self._connection = self._opener()
            self._channels = self._rate = self._last_time = None
            self.state = StreamState.RUNNING
            self.health = StreamHealth.GOOD
        except Exception:
            self.state = StreamState.FAULT
            self.health = StreamHealth.LOST
            raise

    def read(self):
        if self.state != StreamState.RUNNING:
            raise RuntimeError("source is not running")
        samples, times, events = [], [], []
        try:
            for _ in range(self.block_size):
                line = self._connection.readline()
                if not line:
                    raise ConnectionError("EEG source disconnected")
                packet = json.loads(line)
                if self._channels is None:
                    self._channels = _channels(packet["channels"])
                    self._rate = float(packet["sampling_rate"])
                    if not isfinite(self._rate) or self._rate <= 0:
                        raise ValueError("invalid sampling rate")
                elif "channels" in packet or "sampling_rate" in packet:
                    if tuple(packet.get("channels", [c.name for c in self._channels])) != tuple(c.name for c in self._channels) or float(packet.get("sampling_rate", self._rate)) != self._rate:
                        raise ValueError("stream metadata changed")
                timestamp = float(packet["timestamp"])
                sample = tuple(float(value) for value in packet["sample"])
                if self._last_time is not None and timestamp <= self._last_time:
                    raise ValueError("timestamps must increase across blocks")
                self._last_time = timestamp
                if packet.get("event"):
                    events.append(EEGEvent(len(samples), str(packet["event"]), timestamp))
                samples.append(sample)
                times.append(timestamp)
            return EEGBlock(tuple(samples), tuple(times), self._channels, self._rate, tuple(events))
        except Exception:
            self.state = StreamState.FAULT
            self.health = StreamHealth.LOST
            self.stop(close_only=True)
            raise

    def stop(self, close_only=False):
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        if not close_only:
            self.state = StreamState.STOPPED
            self.health = StreamHealth.LOST


class TCPSource(JsonLineSource):
    def __init__(self, host: str, port: int, timeout=2.0, block_size=50):
        def open_socket():
            connection = socket.create_connection((host, port), timeout=timeout)
            return _SocketLines(connection)
        super().__init__(open_socket, block_size)


class _SocketLines:
    def __init__(self, connection):
        self.connection = connection
        self.file = connection.makefile("rb")

    def readline(self):
        return self.file.readline()

    def close(self):
        self.file.close()
        self.connection.close()


class SerialSource(JsonLineSource):
    def __init__(self, port: str, baudrate=115200, timeout=2.0, block_size=50):
        def open_serial():
            try:
                import serial
            except ImportError as exc:
                raise RuntimeError("serial acquisition requires pyserial; install mi-bci-platform[acquisition]") from exc
            return serial.Serial(port, baudrate=baudrate, timeout=timeout)
        super().__init__(open_serial, block_size)


class BluetoothSerialSource(SerialSource):
    """Use the COM port paired by Windows for Bluetooth SPP."""


class LSLSource:
    def __init__(self, name: str | None = None, timeout=2.0, block_size=50):
        self.name, self.timeout, self.block_size = name, timeout, block_size
        self.state = StreamState.IDLE
        self.health = StreamHealth.LOST
        self._inlet = None
        self._last_time = None

    def start(self):
        if self.state == StreamState.RUNNING:
            raise RuntimeError("source already running")
        try:
            import pylsl
        except ImportError as exc:
            raise RuntimeError("LSL acquisition requires pylsl; install mi-bci-platform[acquisition]") from exc
        try:
            streams = pylsl.resolve_byprop("name", self.name, timeout=self.timeout) if self.name else pylsl.resolve_byprop("type", "EEG", timeout=self.timeout)
            if not streams:
                raise ConnectionError("LSL EEG stream not found")
            info = streams[0]
            self._rate = float(info.nominal_srate())
            if not isfinite(self._rate) or self._rate <= 0:
                raise ValueError("LSL stream requires a nominal sampling rate")
            labels = []
            channel = info.desc().child("channels").child("channel")
            for index in range(info.channel_count()):
                labels.append(channel.child_value("label") or f"ch{index + 1}")
                channel = channel.next_sibling()
            self._channels = _channels(labels)
            self._inlet = pylsl.StreamInlet(info, max_buflen=10)
            self._last_time = None
            self.state, self.health = StreamState.RUNNING, StreamHealth.GOOD
        except Exception:
            self.state, self.health = StreamState.FAULT, StreamHealth.LOST
            raise

    def read(self):
        if self.state != StreamState.RUNNING:
            raise RuntimeError("source is not running")
        try:
            samples, times = self._inlet.pull_chunk(timeout=self.timeout, max_samples=self.block_size)
            if not samples:
                raise ConnectionError("LSL EEG stream timed out")
            if self._last_time is not None and times[0] <= self._last_time:
                raise ValueError("LSL timestamps must increase across blocks")
            block = EEGBlock(tuple(tuple(float(v) for v in row) for row in samples),
                             tuple(float(t) for t in times), self._channels, self._rate)
            self._last_time = block.timestamps[-1]
            return block
        except Exception:
            self.state, self.health = StreamState.FAULT, StreamHealth.LOST
            self.stop(close_only=True)
            raise

    def stop(self, close_only=False):
        if self._inlet is not None:
            self._inlet.close_stream()
            self._inlet = None
        if not close_only:
            self.state, self.health = StreamState.STOPPED, StreamHealth.LOST


def import_file(project, path, sampling_rate=None, block_size=250):
    source = FileSource(path, sampling_rate, block_size)
    source.start()
    try:
        def blocks():
            while (block := source.read()) is not None:
                yield block
        return project.record(source.path.suffix.lower().lstrip("."), blocks(), raw_file=source.path)
    finally:
        source.stop()
