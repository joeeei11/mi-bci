"""Recoverable local project storage and de-identified BIDS derivative export."""

import csv
import hashlib
import io
import json
import shutil
import sqlite3
import uuid
import zipfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .stream import Channel, EEGBlock, EEGEvent, StreamHealth, StreamState


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tsv(rows: list[list[object]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.writer(output, delimiter="\t", lineterminator="\n")
    writer.writerows(rows)
    return output.getvalue()


class Project:
    def __init__(self, root: Path, connection: sqlite3.Connection) -> None:
        self.root = root
        self.db = connection
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS protocol_events (
                id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, created_at TEXT NOT NULL,
                elapsed REAL NOT NULL, code TEXT NOT NULL, config_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS calibration_sessions (
                recording_id TEXT PRIMARY KEY REFERENCES recordings(id), session_id TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS models (
                id TEXT PRIMARY KEY, created_at TEXT NOT NULL, algorithm TEXT NOT NULL,
                subject TEXT NOT NULL, recording_id TEXT, artifact TEXT NOT NULL,
                provenance_json TEXT NOT NULL, metrics_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS model_activation (
                id INTEGER PRIMARY KEY, created_at TEXT NOT NULL, model_id TEXT REFERENCES models(id),
                action TEXT NOT NULL
            );
        """)

    @classmethod
    def create(cls, root: str | Path, name: str, subject: str = "01") -> "Project":
        path = Path(root)
        if path.exists() and any(path.iterdir()):
            raise FileExistsError("project directory is not empty")
        if not name.strip() or not subject.isalnum():
            raise ValueError("project name and alphanumeric subject pseudonym are required")
        path.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(path / "metadata.sqlite")
        db.execute("PRAGMA foreign_keys = ON")
        db.executescript("""
            CREATE TABLE project (name TEXT NOT NULL, subject TEXT NOT NULL, created_at TEXT NOT NULL);
            CREATE TABLE config_history (id INTEGER PRIMARY KEY, created_at TEXT NOT NULL, config_json TEXT NOT NULL);
            CREATE TABLE recordings (
                id TEXT PRIMARY KEY, source_kind TEXT NOT NULL, source_path TEXT, source_sha256 TEXT,
                created_at TEXT NOT NULL, status TEXT NOT NULL, sampling_rate REAL, channels_json TEXT
            );
            CREATE TABLE blocks (
                recording_id TEXT NOT NULL REFERENCES recordings(id), sequence INTEGER NOT NULL,
                payload_json TEXT NOT NULL, PRIMARY KEY (recording_id, sequence)
            );
            CREATE TABLE audit (
                id INTEGER PRIMARY KEY, created_at TEXT NOT NULL, action TEXT NOT NULL,
                details_json TEXT NOT NULL
            );
        """)
        with db:
            db.execute("INSERT INTO project VALUES (?, ?, ?)", (name, subject, _now()))
            db.execute("INSERT INTO audit (created_at, action, details_json) VALUES (?, ?, ?)",
                       (_now(), "project_created", "{}"))
        project = cls(path, db)
        project._write_dataset_description()
        return project

    @classmethod
    def open(cls, root: str | Path) -> "Project":
        path = Path(root)
        if not (path / "metadata.sqlite").is_file():
            raise FileNotFoundError("project metadata.sqlite not found")
        db = sqlite3.connect(path / "metadata.sqlite")
        db.execute("PRAGMA foreign_keys = ON")
        project = cls(path, db)
        project.db.execute("SELECT name FROM project").fetchone()
        for row in project.db.execute("SELECT id FROM recordings WHERE status = 'recording'").fetchall():
            with project.db:
                project.db.execute("UPDATE recordings SET status = 'recovered' WHERE id = ?", (row["id"],))
                project._audit("recording_recovered", {"recording_id": row["id"]})
        with project.db:
            project._audit("project_opened", {})
        project._write_dataset_description()
        for row in project.db.execute("SELECT id FROM recordings WHERE status != 'recording'"):
            project._materialize(row["id"])
        return project

    @property
    def name(self) -> str:
        return self.db.execute("SELECT name FROM project").fetchone()["name"]

    @property
    def subject(self) -> str:
        return self.db.execute("SELECT subject FROM project").fetchone()["subject"]

    def close(self) -> None:
        self.db.close()

    def _audit(self, action: str, details: dict) -> None:
        self.db.execute("INSERT INTO audit (created_at, action, details_json) VALUES (?, ?, ?)",
                        (_now(), action, json.dumps(details, sort_keys=True)))

    def audit_records(self) -> list[dict]:
        return [dict(row) | {"details": json.loads(row["details_json"])}
                for row in self.db.execute("SELECT * FROM audit ORDER BY id")]

    def set_config(self, config: dict) -> None:
        encoded = json.dumps(config, sort_keys=True)
        with self.db:
            self.db.execute("INSERT INTO config_history (created_at, config_json) VALUES (?, ?)", (_now(), encoded))
            self._audit("config_saved", {"revision": self.db.execute("SELECT last_insert_rowid()").fetchone()[0]})

    def config_history(self) -> list[dict]:
        return [json.loads(row[0]) for row in self.db.execute("SELECT config_json FROM config_history ORDER BY id")]

    def begin_live_recording(self, source_kind: str, session_id: str | None = None) -> "LiveRecording":
        return LiveRecording(self, source_kind, session_id)

    def record_protocol_event(self, session_id: str, elapsed: float, code: str, config: dict) -> None:
        if not session_id or not code or elapsed < 0:
            raise ValueError("invalid protocol event")
        with self.db:
            self.db.execute("INSERT INTO protocol_events (session_id, created_at, elapsed, code, config_json) VALUES (?, ?, ?, ?, ?)",
                            (session_id, _now(), elapsed, code, json.dumps(config, sort_keys=True)))
            self._audit("protocol_event", {"session_id": session_id, "code": code, "elapsed": elapsed})

    def protocol_events(self, session_id: str | None = None) -> list[dict]:
        query = "SELECT * FROM protocol_events"
        rows = self.db.execute(query + (" WHERE session_id = ?" if session_id else "") + " ORDER BY id",
                               (session_id,) if session_id else ())
        return [dict(row) | {"config": json.loads(row["config_json"])} for row in rows]

    def register_model(self, algorithm: str, subject: str, recording_id: str | None,
                       artifact: str, provenance: dict, metrics: dict) -> str:
        if not algorithm or not subject or not (self.root / artifact).is_file():
            raise ValueError("model metadata or artifact is missing")
        if recording_id and self.db.execute("SELECT 1 FROM recordings WHERE id = ?", (recording_id,)).fetchone() is None:
            raise KeyError(recording_id)
        identifier = uuid.uuid4().hex[:12]
        with self.db:
            self.db.execute("INSERT INTO models VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                            (identifier, _now(), algorithm, subject, recording_id, artifact,
                             json.dumps(provenance, sort_keys=True), json.dumps(metrics, sort_keys=True)))
            self._audit("model_registered", {"model_id": identifier})
        return identifier

    def models(self) -> list[dict]:
        return [dict(row) | {"provenance": json.loads(row["provenance_json"]),
                              "metrics": json.loads(row["metrics_json"])}
                for row in self.db.execute("SELECT * FROM models ORDER BY created_at, id")]

    def active_model(self) -> dict | None:
        row = self.db.execute("SELECT model_id FROM model_activation ORDER BY id DESC LIMIT 1").fetchone()
        return next((model for model in self.models() if model["id"] == row["model_id"]), None) if row else None

    def activate_model(self, model_id: str, subject: str, channels: list[str], sampling_rate: float) -> None:
        model = next((item for item in self.models() if item["id"] == model_id), None)
        if model is None:
            raise KeyError(model_id)
        source = model["provenance"]
        if model["subject"] != subject or source["channels"] != channels or source["sampling_rate"] != sampling_rate:
            raise ValueError("model subject or EEG metadata is incompatible")
        if not (self.root / model["artifact"]).is_file():
            raise FileNotFoundError(model["artifact"])
        artifact_hash = hashlib.sha256((self.root / model["artifact"]).read_bytes()).hexdigest()
        if artifact_hash != source["artifact_sha256"]:
            raise ValueError("model artifact hash does not match its registered version")
        with self.db:
            self.db.execute("INSERT INTO model_activation (created_at, model_id, action) VALUES (?, ?, ?)",
                            (_now(), model_id, "activate"))
            self._audit("model_activated", {"model_id": model_id})

    def activation_history(self) -> list[dict]:
        return [dict(row) for row in self.db.execute("SELECT * FROM model_activation ORDER BY id")]

    def record(self, source_kind: str, blocks: Iterable[EEGBlock], raw_file: str | Path | None = None) -> str:
        if not source_kind.strip():
            raise ValueError("source kind is required")
        recording_id = uuid.uuid4().hex[:12]
        source_path = None
        source_sha256 = None
        if raw_file is not None:
            source = Path(raw_file)
            digest = hashlib.sha256()
            with source.open("rb") as input_file:
                for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
                    digest.update(chunk)
            source_sha256 = digest.hexdigest()
            target = self.root / "sourcedata" / f"{recording_id}{source.suffix.lower()}"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            source_path = str(target.relative_to(self.root))
        with self.db:
            self.db.execute("INSERT INTO recordings VALUES (?, ?, ?, ?, ?, ?, NULL, NULL)",
                            (recording_id, source_kind, source_path, source_sha256, _now(), "recording"))
            self._audit("recording_started", {"recording_id": recording_id, "source_kind": source_kind})
        previous_end = None
        reference = None
        count = 0
        for block in blocks:
            if reference is None:
                reference = (block.channels, block.sampling_rate)
            if (block.channels, block.sampling_rate) != reference:
                raise ValueError("channel metadata or sampling rate changed within recording")
            if previous_end is not None and block.timestamps[0] <= previous_end:
                raise ValueError("timestamps must increase across blocks")
            previous_end = block.timestamps[-1]
            with self.db:
                self.db.execute("INSERT INTO blocks VALUES (?, ?, ?)",
                                (recording_id, count, json.dumps(asdict(block))))
            count += 1
        if count == 0:
            raise ValueError("recording contains no blocks")
        with self.db:
            self.db.execute("UPDATE recordings SET status = 'complete', sampling_rate = ?, channels_json = ? WHERE id = ?",
                            (reference[1], json.dumps([asdict(channel) for channel in reference[0]]), recording_id))
            self._audit("recording_completed", {"recording_id": recording_id, "blocks": count})
        self._materialize(recording_id)
        return recording_id

    def blocks(self, recording_id: str) -> list[EEGBlock]:
        if self.db.execute("SELECT 1 FROM recordings WHERE id = ?", (recording_id,)).fetchone() is None:
            raise KeyError(recording_id)
        result = []
        for row in self.db.execute("SELECT payload_json FROM blocks WHERE recording_id = ? ORDER BY sequence", (recording_id,)):
            payload = json.loads(row[0])
            result.append(EEGBlock(
                tuple(tuple(sample) for sample in payload["samples"]), tuple(payload["timestamps"]),
                tuple(Channel(**channel) for channel in payload["channels"]), payload["sampling_rate"],
                tuple(EEGEvent(**event) for event in payload["events"]),
                StreamHealth(payload["health"]), StreamState(payload["state"])))
        return result

    def _write_dataset_description(self) -> None:
        folder = self.root / "derivatives" / "mi_bci"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "dataset_description.json").write_text(json.dumps({
            "Name": self.name, "BIDSVersion": "1.10.1", "DatasetType": "derivative",
            "GeneratedBy": [{"Name": "mi-bci-platform", "Version": "0.1.0"}]}, indent=2), encoding="utf-8")

    def _artifact_files(self, recording_id: str, subject: str, deidentified: bool = False) -> dict[str, str]:
        row = self.db.execute("SELECT * FROM recordings WHERE id = ?", (recording_id,)).fetchone()
        blocks = self.blocks(recording_id)
        if not blocks:
            return {}
        prefix = f"sub-{subject}_task-mi_run-{recording_id}"
        folder = f"sub-{subject}/eeg/"
        channels = blocks[0].channels
        standard_channels = {"C3", "C4", "Cz", "F3", "F4", "Fz", "P3", "P4", "Pz",
                             "O1", "O2", "Oz", "Fp1", "Fp2"}
        channel_names = ([channel.name if channel.name in standard_channels
                          else f"ch{index + 1}" for index, channel in enumerate(channels)]
                         if deidentified else [channel.name for channel in channels])
        origin = blocks[0].timestamps[0] if deidentified else 0
        data_rows: list[list[object]] = [["timestamp", *channel_names]]
        event_rows: list[list[object]] = [["onset", "duration", "trial_type", "sample"]]
        sample_offset = 0
        for block in blocks:
            data_rows.extend([[timestamp - origin, *sample] for timestamp, sample in zip(block.timestamps, block.samples)])
            event_rows.extend([[event.timestamp - origin, 0,
                                event.code if not deidentified or event.code.lower() in
                                {"left", "right", "cue", "rest", "prepare", "start", "stop"} else "other",
                                sample_offset + event.sample_index]
                               for event in block.events])
            sample_offset += len(block.samples)
        sidecar = {
            "SamplingFrequency": blocks[0].sampling_rate, "EEGChannelCount": len(channels),
            "Channels": ([{"name": name,
                           "unit": channel.unit if not deidentified or channel.unit in {"uV", "V", "mV"} else "unknown",
                           "kind": channel.kind if not deidentified or channel.kind in {"EEG", "EOG", "EMG", "ECG"} else "other"}
                          for name, channel in zip(channel_names, channels)]),
            "SourceKind": (row["source_kind"] if not deidentified else
                           row["source_kind"] if row["source_kind"] in
                           {"synthetic", "replay", "csv", "edf", "bdf", "lsl", "serial", "tcp", "bluetooth"}
                           else "other"),
            "SourceSHA256": row["source_sha256"], "RecordingStatus": row["status"],
        }
        return {
            folder + prefix + "_eeg.tsv": _tsv(data_rows),
            folder + prefix + "_events.tsv": _tsv(event_rows),
            folder + prefix + "_eeg.json": json.dumps(sidecar, indent=2),
        }

    def _materialize(self, recording_id: str) -> None:
        base = self.root / "derivatives" / "mi_bci"
        for relative, content in self._artifact_files(recording_id, self.subject).items():
            target = base / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".tmp")
            temporary.write_text(content, encoding="utf-8")
            temporary.replace(target)

    def export_deidentified(self, destination: str | Path) -> Path:
        destination = Path(destination)
        if destination.resolve().is_relative_to(self.root.resolve()):
            raise ValueError("export destination must be outside the project")
        base = "derivatives/mi_bci/"
        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(base + "dataset_description.json", json.dumps({
                "Name": "De-identified MI-BCI export", "BIDSVersion": "1.10.1",
                "DatasetType": "derivative", "GeneratedBy": [{"Name": "mi-bci-platform", "Version": "0.1.0"}]}))
            archive.writestr("participants.tsv", _tsv([["participant_id"], ["sub-01"]]))
            for row in self.db.execute("SELECT id FROM recordings ORDER BY created_at, id"):
                for relative, content in self._artifact_files(row["id"], "01", deidentified=True).items():
                    archive.writestr(base + relative, content)
            # Export an audit timeline without internal paths, names, or free-form details.
            audit = [{"sequence": index, "action": row["action"]}
                     for index, row in enumerate(self.db.execute("SELECT action FROM audit ORDER BY id"), 1)]
            archive.writestr("audit.json", json.dumps(audit, indent=2))
        with self.db:
            self._audit("project_exported", {"deidentified": True})
        return destination


class LiveRecording:
    """Append acquisition blocks as they arrive; completed runs use normal derivatives."""

    def __init__(self, project: Project, source_kind: str, session_id: str | None):
        if not source_kind.strip():
            raise ValueError("source kind is required")
        self.project = project
        self.id = uuid.uuid4().hex[:12]
        self.reference = None
        self.previous_end = None
        self.count = 0
        self.finished = False
        with project.db:
            project.db.execute("INSERT INTO recordings VALUES (?, ?, NULL, NULL, ?, 'recording', NULL, NULL)",
                               (self.id, source_kind, _now()))
            if session_id:
                project.db.execute("INSERT INTO calibration_sessions VALUES (?, ?)", (self.id, session_id))
            project._audit("recording_started", {"recording_id": self.id, "source_kind": source_kind,
                                                 "session_id": session_id})

    def append(self, block: EEGBlock) -> None:
        if self.finished:
            raise RuntimeError("recording is finished")
        reference = (block.channels, block.sampling_rate)
        if self.reference is not None and reference != self.reference:
            raise ValueError("channel metadata or sampling rate changed within recording")
        if self.previous_end is not None and block.timestamps[0] <= self.previous_end:
            raise ValueError("timestamps must increase across blocks")
        with self.project.db:
            self.project.db.execute("INSERT INTO blocks VALUES (?, ?, ?)",
                                    (self.id, self.count, json.dumps(asdict(block))))
            if self.reference is None:
                self.project.db.execute("UPDATE recordings SET sampling_rate = ?, channels_json = ? WHERE id = ?",
                                        (block.sampling_rate, json.dumps([asdict(channel) for channel in block.channels]), self.id))
        self.reference = reference
        self.previous_end = block.timestamps[-1]
        self.count += 1

    def finish(self, completed: bool = True) -> str | None:
        if self.finished:
            return self.id if self.count else None
        self.finished = True
        if not self.count:
            with self.project.db:
                self.project.db.execute("DELETE FROM calibration_sessions WHERE recording_id = ?", (self.id,))
                self.project.db.execute("DELETE FROM recordings WHERE id = ?", (self.id,))
            return None
        with self.project.db:
            self.project.db.execute("UPDATE recordings SET status = ? WHERE id = ?",
                                    ("complete" if completed else "interrupted", self.id))
            self.project._audit("recording_completed" if completed else "recording_interrupted",
                                {"recording_id": self.id, "blocks": self.count})
        self.project._materialize(self.id)
        return self.id
