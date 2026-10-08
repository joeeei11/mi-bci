"""Hardware-free sources implementing the normalized stream contract."""

import math
import random

from .stream import Channel, EEGBlock, EEGEvent, StreamState


class SyntheticSource:
    def __init__(self, sampling_rate: int = 250, channels: tuple[Channel, ...] = (Channel("C3"), Channel("C4")),
                 block_size: int = 50, total_samples: int = 500, seed: int = 0,
                 event_samples: dict[int, str] | None = None) -> None:
        if sampling_rate <= 0 or block_size <= 0 or total_samples < 0 or not channels:
            raise ValueError("invalid synthetic source configuration")
        self.sampling_rate = sampling_rate
        self.channels = channels
        self.block_size = block_size
        self.total_samples = total_samples
        self.seed = seed
        self.event_samples = event_samples or {}
        if any(index < 0 or index >= total_samples for index in self.event_samples):
            raise ValueError("event sample outside synthetic recording")
        self.state = StreamState.IDLE
        self._position = 0
        self._random = random.Random(seed)

    def start(self) -> None:
        if self.state == StreamState.RUNNING:
            raise RuntimeError("source already running")
        self._position = 0
        self._random = random.Random(self.seed)
        self.state = StreamState.RUNNING

    def read(self) -> EEGBlock | None:
        if self.state != StreamState.RUNNING:
            raise RuntimeError("source is not running")
        if self._position == self.total_samples:
            self.state = StreamState.STOPPED
            return None
        end = min(self._position + self.block_size, self.total_samples)
        indexes = range(self._position, end)
        timestamps = tuple(index / self.sampling_rate for index in indexes)
        samples = tuple(tuple(12.0 * math.sin(2 * math.pi * (10 + channel_index) * timestamp)
                              + self._random.gauss(0, 0.5) for channel_index in range(len(self.channels)))
                        for timestamp in timestamps)
        events = tuple(EEGEvent(index - self._position, self.event_samples[index], index / self.sampling_rate)
                       for index in indexes if index in self.event_samples)
        self._position = end
        return EEGBlock(samples, timestamps, self.channels, self.sampling_rate, events)

    def stop(self) -> None:
        self.state = StreamState.STOPPED


class ReplaySource:
    def __init__(self, project: "Project", recording_id: str) -> None:
        self.project = project
        self.recording_id = recording_id
        self.state = StreamState.IDLE
        self._blocks: iter | None = None

    def start(self) -> None:
        if self.state == StreamState.RUNNING:
            raise RuntimeError("source already running")
        self._blocks = iter(self.project.blocks(self.recording_id))
        self.state = StreamState.RUNNING

    def read(self) -> EEGBlock | None:
        if self.state != StreamState.RUNNING or self._blocks is None:
            raise RuntimeError("source is not running")
        block = next(self._blocks, None)
        if block is None:
            self.state = StreamState.STOPPED
        return block

    def stop(self) -> None:
        self._blocks = None
        self.state = StreamState.STOPPED


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .project import Project
