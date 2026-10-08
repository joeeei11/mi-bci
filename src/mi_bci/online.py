"""Online inference, acknowledged STM32 commands, and fail-safe session control."""

import hashlib
import math
import socket
import time
from collections import deque
from dataclasses import dataclass

from .stream import StreamHealth, StreamState


@dataclass(frozen=True)
class OnlineConfig:
    window_seconds: float = 1.0
    hop_seconds: float = 0.25
    confidence: float = 0.8
    consecutive: int = 3
    cooldown_seconds: float = 1.0
    stream_timeout: float = 1.0
    ack_timeout: float = 0.3

    def validate(self):
        if not (0 < self.hop_seconds <= self.window_seconds and
                0.5 < self.confidence <= 1 and self.consecutive >= 1 and
                self.cooldown_seconds >= 0 and self.stream_timeout > 0 and self.ack_timeout > 0):
            raise ValueError("invalid online control configuration")


class ModelPredictor:
    def __init__(self, project, model, channels, rate, window_samples):
        import numpy as np

        provenance = model["provenance"]
        if (model["subject"] != project.subject or provenance["channels"] != channels or
                provenance["sampling_rate"] != rate):
            raise ValueError("active model is incompatible with EEG metadata")
        path = project.root / model["artifact"]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != provenance["artifact_sha256"]:
            raise ValueError("active model artifact is missing or modified")
        self.algorithm = model["algorithm"]
        self.rate = rate
        self.band = tuple(provenance["preprocessing"]["bandpass_hz"])
        if not 0 < self.band[0] < self.band[1] < rate / 2:
            raise ValueError("model filter is incompatible with EEG rate")
        if self.algorithm == "FBCSP+LDA":
            with np.load(path, allow_pickle=False) as saved:
                self.bands = saved["bands"]
                self.filters = saved["filters"]
                self.selected = saved["selected_features"]
                self.coef = saved["coef"]
                self.intercept = saved["intercept"]
        elif self.algorithm == "EEGNet":
            import torch
            from .research import compute_device, eegnet_network

            self.device = compute_device()
            saved = torch.load(path, map_location=self.device, weights_only=True)
            if saved["channels"] != len(channels) or saved["samples"] != window_samples:
                raise ValueError("EEGNet requires its trained window length")
            self.network = eegnet_network(len(channels), window_samples).to(self.device)
            self.network.load_state_dict(saved["state_dict"])
            self.network.eval()
            self.scale = saved["scale"]
        else:
            raise ValueError("unsupported active model")

    def predict(self, window):
        import numpy as np
        from .research import _filter

        data = np.asarray(window, dtype=float).T[None]
        if not np.isfinite(data).all():
            raise ValueError("nonfinite online signal")
        if self.algorithm == "FBCSP+LDA":
            features = []
            for band, matrix in zip(self.bands, self.filters):
                filtered = _filter(data, self.rate, tuple(band))
                projected = np.einsum("kc,nct->nkt", matrix, filtered)
                variance = np.var(projected, axis=-1) + 1e-12
                features.append(np.log(variance / variance.sum(axis=1, keepdims=True)))
            selected = np.concatenate(features, axis=1)[:, self.selected]
            score = float((selected @ self.coef.T + self.intercept)[0, 0])
            probability = 1 / (1 + math.exp(-max(-700, min(700, score))))
        else:
            import torch

            filtered = _filter(data, self.rate, self.band)
            with torch.no_grad():
                inputs = torch.tensor((filtered / self.scale)[:, None], dtype=torch.float32,
                                      device=self.device)
                probability = float(torch.softmax(self.network(inputs), dim=1)[0, 1])
        return ("RIGHT", probability) if probability >= 0.5 else ("LEFT", 1 - probability)


class VirtualSTM32:
    """Firmware-equivalent endpoint; fault switches support deterministic tests."""

    def __init__(self):
        self.connected = False
        self.command = "STOP"
        self.history = []
        self.ack_enabled = True
        self.seen = {}

    def connect(self):
        self.connected = True
        self.command = "STOP"
        self.seen.clear()

    def close(self):
        self.command = "STOP"
        self.connected = False

    def exchange(self, sequence, command, timeout):
        if not self.connected:
            raise ConnectionError("STM32 disconnected")
        if self.seen and sequence < max(self.seen):
            raise ValueError("stale sequence")
        if sequence in self.seen:
            if self.seen[sequence] != command:
                raise ValueError("sequence reused for another command")
        else:
            self.seen[sequence] = command
            self.command = command
            self.history.append(command)
        if not self.ack_enabled:
            raise TimeoutError("STM32 ACK timeout")
        return True


class LineTransport:
    def __init__(self, kind, target, baudrate=115200):
        if kind not in {"usb", "bluetooth", "tcp"}:
            raise ValueError("unsupported STM32 transport")
        self.kind, self.target, self.baudrate = kind, target, baudrate
        self.connection = None

    def connect(self):
        if self.kind == "tcp":
            host, port = self.target.rsplit(":", 1)
            self.connection = socket.create_connection((host, int(port)), timeout=2)
        else:
            import serial
            self.connection = serial.Serial(self.target, self.baudrate, timeout=0.3, write_timeout=0.3)

    def close(self):
        if self.connection:
            self.connection.close()
            self.connection = None

    def exchange(self, sequence, command, timeout):
        if self.connection is None:
            raise ConnectionError("STM32 disconnected")
        frame = f"CMD {sequence} {command}\n".encode("ascii")
        deadline = time.monotonic() + timeout
        self.connection.settimeout(timeout) if self.kind == "tcp" else setattr(self.connection, "timeout", timeout)
        self.connection.sendall(frame) if self.kind == "tcp" else self.connection.write(frame)
        while time.monotonic() < deadline:
            try:
                line = self._readline()
            except (socket.timeout, TimeoutError):
                break
            if line == f"ACK {sequence} {command}":
                return True
            if line.startswith(f"ERR {sequence} "):
                raise ValueError(line)
        raise TimeoutError(f"STM32 {command} ACK timeout")

    def _readline(self):
        if self.kind != "tcp":
            return self.connection.readline().decode("ascii", errors="replace").strip()
        line = bytearray()
        while len(line) < 80:
            byte = self.connection.recv(1)
            if not byte:
                raise ConnectionError("STM32 disconnected")
            if byte == b"\n":
                break
            line.extend(byte)
        return line.decode("ascii", errors="replace").strip()


class OnlineSession:
    def __init__(self, project, transport, predictor_factory=ModelPredictor, config=None, clock=time.monotonic):
        self.project, self.transport, self.predictor_factory = project, transport, predictor_factory
        self.config = config or OnlineConfig()
        self.config.validate()
        self.clock = clock
        self.state = "idle"
        self.stop_confirmed = False
        self.sequence = 0
        self.events = []
        self.samples = deque()
        self.previous_timestamp = None
        self.last_received = None
        self.next_window_end = None
        self.last_command_at = float("-inf")
        self.streak_label = None
        self.streak = 0
        self.last_command = "STOP"
        self.predictor = None

    def _event(self, kind, **details):
        item = {"kind": kind, "time": self.clock(), **details}
        self.events.append(item)
        with self.project.db:
            self.project._audit("online_" + kind, details)

    def _send(self, command):
        if command == "STOP" and self.last_command == "STOP" and self.stop_confirmed:
            return
        self.sequence += 1
        try:
            self.transport.exchange(self.sequence, command, self.config.ack_timeout)
            self.last_command = command
            self.stop_confirmed = command == "STOP"
            self._event("command", command=command, sequence=self.sequence, acknowledged=True)
        except Exception as exc:
            self.stop_confirmed = False
            self._event("command", command=command, sequence=self.sequence,
                        acknowledged=False, error=str(exc))
            raise

    def start(self, channels, rate):
        if self.state != "idle":
            raise RuntimeError("session requires manual reset")
        model = self.project.active_model()
        if model is None:
            raise ValueError("activate a model before online control")
        samples = round(self.config.window_seconds * rate)
        if samples < 2 or round(self.config.hop_seconds * rate) < 1:
            raise ValueError("online window or hop is too short")
        self.predictor = self.predictor_factory(self.project, model, channels, rate, samples)
        try:
            self.transport.connect()
            self.stop_confirmed = False
            self._send("STOP")
        except Exception:
            self.state = "fault"
            self._event("fault", reason="device_start")
            raise
        self.channels, self.rate = channels, rate
        self.window_samples = samples
        self.hop_samples = round(self.config.hop_seconds * rate)
        self.state = "running"
        self.last_received = self.clock()
        self._event("start", model_id=model["id"], channels=channels, rate=rate,
                    config=self.config.__dict__)

    def feed(self, block):
        if self.state != "running":
            return
        try:
            if block.health != StreamHealth.GOOD or block.state != StreamState.RUNNING:
                raise ValueError("EEG stream is degraded or stopped")
            if [channel.name for channel in block.channels] != self.channels or block.sampling_rate != self.rate:
                raise ValueError("EEG metadata changed")
            if self.previous_timestamp is not None:
                gap = block.timestamps[0] - self.previous_timestamp
                if gap <= 0 or gap > max(2 / self.rate, self.config.hop_seconds):
                    raise ValueError("EEG timestamp gap or reversal")
            for sample, timestamp in zip(block.samples, block.timestamps):
                self.samples.append(sample)
                if len(self.samples) > self.window_samples:
                    self.samples.popleft()
                if len(self.samples) == self.window_samples and (self.next_window_end is None or timestamp >= self.next_window_end):
                    self.next_window_end = timestamp + self.hop_samples / self.rate - 1 / (self.rate * 10)
                    self._predict(timestamp)
            self.previous_timestamp = block.timestamps[-1]
            self.last_received = self.clock()
        except Exception as exc:
            self.fault(str(exc))

    def _predict(self, timestamp):
        label, confidence = self.predictor.predict(tuple(self.samples))
        if label not in {"LEFT", "RIGHT"} or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("invalid model prediction")
        self._event("prediction", label=label, confidence=confidence, eeg_timestamp=timestamp)
        if confidence < self.config.confidence:
            self.streak = 0
            self.streak_label = None
            self._send("STOP")
            return
        self.streak = self.streak + 1 if label == self.streak_label else 1
        self.streak_label = label
        if self.streak < self.config.consecutive or self.clock() - self.last_command_at < self.config.cooldown_seconds:
            return
        if self.last_command != label:
            self._send(label)
            self.last_command_at = self.clock()

    def tick(self):
        if self.state == "running" and self.clock() - self.last_received > self.config.stream_timeout:
            self.fault("EEG stream timeout")

    def fault(self, reason):
        if self.state == "fault":
            return
        self.state = "fault"
        self.streak = 0
        self._event("fault", reason=reason)
        try:
            self._send("STOP")
        except Exception:
            pass

    def stop(self):
        if self.state == "idle":
            return
        was_fault = self.state == "fault"
        self.state = "fault" if was_fault else "stopping"
        try:
            self._send("STOP")
            if not was_fault:
                self.state = "idle"
                self._event("stopped")
        except Exception:
            self.state = "fault"
            self._event("fault", reason="STOP unconfirmed")

    def reset(self):
        if self.state != "fault":
            raise RuntimeError("only a fault can be reset")
        self.transport.close()
        self.state = "idle"
        self.last_command = "STOP"
        self.stop_confirmed = False
        self.samples.clear()
        self.previous_timestamp = None
        self.next_window_end = None
        self._event("reset")
