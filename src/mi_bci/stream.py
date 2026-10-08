"""Normalized, hardware-independent EEG stream contract."""

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from typing import Protocol


class StreamState(StrEnum):
    IDLE = "idle"
    READY = "ready"
    RUNNING = "running"
    STOPPED = "stopped"
    FAULT = "fault"


class StreamHealth(StrEnum):
    GOOD = "good"
    DEGRADED = "degraded"
    LOST = "lost"


@dataclass(frozen=True)
class Channel:
    name: str
    unit: str = "uV"
    kind: str = "EEG"

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("channel name is required")


@dataclass(frozen=True)
class EEGEvent:
    sample_index: int
    code: str
    timestamp: float

    def __post_init__(self) -> None:
        if self.sample_index < 0 or not self.code.strip():
            raise ValueError("event requires a nonnegative index and code")


@dataclass(frozen=True)
class EEGBlock:
    samples: tuple[tuple[float, ...], ...]
    timestamps: tuple[float, ...]
    channels: tuple[Channel, ...]
    sampling_rate: float
    events: tuple[EEGEvent, ...] = ()
    health: StreamHealth = StreamHealth.GOOD
    state: StreamState = StreamState.RUNNING

    def __post_init__(self) -> None:
        if not isfinite(self.sampling_rate) or self.sampling_rate <= 0 or not self.channels or not self.samples:
            raise ValueError("positive sampling rate, channels and samples are required")
        if len({channel.name for channel in self.channels}) != len(self.channels):
            raise ValueError("channel names must be unique")
        if len(self.timestamps) != len(self.samples):
            raise ValueError("each sample requires a timestamp")
        if any(len(sample) != len(self.channels) for sample in self.samples):
            raise ValueError("sample width must match channels")
        if any(not isfinite(value) for sample in self.samples for value in sample):
            raise ValueError("samples must be finite")
        if any(not isfinite(timestamp) for timestamp in self.timestamps):
            raise ValueError("timestamps must be finite")
        if any(a >= b for a, b in zip(self.timestamps, self.timestamps[1:])):
            raise ValueError("timestamps must increase strictly")
        if any(event.sample_index >= len(self.samples) or event.timestamp != self.timestamps[event.sample_index] for event in self.events):
            raise ValueError("events must refer to samples in the block")
        if not isinstance(self.health, StreamHealth) or not isinstance(self.state, StreamState):
            raise ValueError("health and state must use stream enums")


class EEGSource(Protocol):
    @property
    def state(self) -> StreamState: ...

    def start(self) -> None: ...

    def read(self) -> EEGBlock | None: ...

    def stop(self) -> None: ...
