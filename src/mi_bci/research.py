"""Reproducible left/right motor-imagery research workflow."""

import hashlib
import html
import json
import os
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProtocolConfig:
    trials_per_side: int = 20
    prepare_seconds: float = 2.0
    cue_seconds: float = 1.0
    imagery_seconds: float = 4.0
    rest_seconds: float = 2.0
    seed: int = 42

    def schedule(self):
        if self.trials_per_side < 1 or min(self.prepare_seconds, self.cue_seconds,
                                           self.imagery_seconds, self.rest_seconds) <= 0:
            raise ValueError("trial count and phase durations must be positive")
        labels = ["left", "right"] * self.trials_per_side
        random.Random(self.seed).shuffle(labels)
        elapsed = 0.0
        for label in labels:
            for code, duration in (("prepare", self.prepare_seconds), (label, self.cue_seconds),
                                   ("imagery_" + label, self.imagery_seconds), ("rest", self.rest_seconds)):
                yield elapsed, code, duration
                elapsed += duration
        yield elapsed, "stop", 0.0


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    test_fraction: float = 0.25
    epoch_start: float = 0.5
    epoch_end: float = 4.0
    low_hz: float = 8.0
    high_hz: float = 30.0
    eegnet_epochs: int = 20

    def validate(self, rate: float):
        if not 0 < self.test_fraction < 0.5 or not 0 <= self.epoch_start < self.epoch_end:
            raise ValueError("invalid split or epoch interval")
        if not 0 < self.low_hz < self.high_hz < rate / 2 or self.eegnet_epochs < 1:
            raise ValueError("invalid filter band or training epochs")


def load_iv2a(path: str | Path, config: TrainingConfig):
    """Load labeled left/right trials from an IV-2a training MAT or GDF."""
    path = Path(path)
    if not re.fullmatch(r"A0[1-9]T\.(mat|gdf)", path.name, re.IGNORECASE):
        raise ValueError("use a labeled IV-2a training file (A01T-A09T.mat/gdf)")
    if path.suffix.lower() == ".mat":
        return _load_iv2a_mat(path, config)
    try:
        import mne
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("IV-2a import requires mi-bci-platform[research]") from exc
    raw = mne.io.read_raw_gdf(path, preload=True, verbose="ERROR")
    try:
        picks = mne.pick_types(raw.info, eeg=True, eog=False, stim=False)
        events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
        codes = {key: event_id[key] for key in ("769", "770") if key in event_id}
        if len(codes) != 2:
            raise ValueError("GDF lacks labeled 769/770 left/right events")
        selected = events[np.isin(events[:, 2], list(codes.values()))]
        rate = float(raw.info["sfreq"])
        config.validate(rate)
        start = int(round(config.epoch_start * rate))
        length = int(round((config.epoch_end - config.epoch_start) * rate))
        data = raw.get_data(picks=picks)
        trials, labels = [], []
        for sample, _, code in selected:
            window = data[:, sample + start:sample + start + length]
            if window.shape[1] == length:
                trials.append(window * 1e6)
                labels.append(0 if code == codes["769"] else 1)
        if not trials:
            raise ValueError("no complete left/right epochs")
        return np.asarray(trials), np.asarray(labels), [raw.ch_names[i] for i in picks], rate
    finally:
        raw.close()


def _load_iv2a_mat(path, config):
    import numpy as np
    from scipy.io import loadmat

    runs = loadmat(path, squeeze_me=True, struct_as_record=False)["data"]
    names = ["Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2",
             "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz"]
    trials, labels = [], []
    rate = None
    for run in np.atleast_1d(runs):
        if not len(np.atleast_1d(run.trial)):
            continue
        current_rate = float(run.fs)
        config.validate(current_rate)
        if rate is not None and current_rate != rate:
            raise ValueError("IV-2a runs have inconsistent sampling rates")
        rate = current_rate
        if run.X.shape[1] < len(names):
            raise ValueError("IV-2a run has fewer than 22 EEG channels")
        start_offset = int(round(config.epoch_start * rate))
        length = int(round((config.epoch_end - config.epoch_start) * rate))
        for onset, label, artifact in zip(np.atleast_1d(run.trial), np.atleast_1d(run.y),
                                          np.atleast_1d(run.artifacts)):
            if int(label) not in (1, 2) or int(artifact):
                continue
            start = int(onset) - 1 + start_offset
            window = run.X[start:start + length, :22]
            if window.shape[0] == length:
                trials.append(window.T)
                labels.append(int(label) - 1)
    if not trials:
        raise ValueError("no complete artifact-free left/right epochs")
    return np.asarray(trials), np.asarray(labels), names, rate


def load_recording(project, recording_id: str, config: TrainingConfig):
    import numpy as np

    blocks = project.blocks(recording_id)
    if not blocks:
        raise ValueError("recording has no data")
    rate = blocks[0].sampling_rate
    config.validate(rate)
    channels = blocks[0].channels
    factors = np.asarray([{"uV": 1, "mV": 1000, "V": 1e6}[channel.unit] for channel in channels])
    samples = np.asarray([sample for block in blocks for sample in block.samples], dtype=float).T * factors[:, None]
    timestamps = np.asarray([stamp for block in blocks for stamp in block.timestamps])
    events = [(event.timestamp, event.code.lower()) for block in blocks for event in block.events]
    length = int(round((config.epoch_end - config.epoch_start) * rate))
    trials, labels = [], []
    for onset, code in events:
        if code not in {"left", "right"}:
            continue
        start = int(np.searchsorted(timestamps, onset + config.epoch_start))
        window = samples[:, start:start + length]
        if window.shape[1] == length and timestamps[start + length - 1] - timestamps[start] <= length / rate * 1.05:
            trials.append(window)
            labels.append(0 if code == "left" else 1)
    if not trials:
        raise ValueError("recording needs complete left/right cue epochs")
    return np.asarray(trials), np.asarray(labels), [channel.name for channel in channels], rate


def _filter(trials, rate, band):
    from scipy.signal import butter, sosfiltfilt

    sos = butter(4, band, btype="bandpass", fs=rate, output="sos")
    return sosfiltfilt(sos, trials, axis=-1)


def _csp_features(train, test, labels, bands, rate):
    import numpy as np
    from scipy.linalg import eigh

    features_train, features_test, filters = [], [], []
    for band in bands:
        x_train, x_test = _filter(train, rate, band), _filter(test, rate, band)
        covariances = []
        for label in (0, 1):
            group = x_train[labels == label]
            covariances.append(np.mean([np.cov(trial) for trial in group], axis=0))
        _, vectors = eigh(covariances[0], covariances[0] + covariances[1] + np.eye(train.shape[1]) * 1e-8)
        matrix = vectors[:, [0, 1, -2, -1]].T if train.shape[1] >= 4 else vectors.T
        filters.append(matrix)
        for data, output in ((x_train, features_train), (x_test, features_test)):
            projected = np.einsum("kc,nct->nkt", matrix, data)
            variance = np.var(projected, axis=-1) + 1e-12
            output.append(np.log(variance / variance.sum(axis=1, keepdims=True)))
    return np.concatenate(features_train, axis=1), np.concatenate(features_test, axis=1), filters


class FBCSPLDA:
    name = "FBCSP+LDA"

    def fit_predict(self, train, test, labels, rate, config, artifact):
        import numpy as np
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.feature_selection import mutual_info_classif
        from sklearn.metrics import balanced_accuracy_score
        from sklearn.model_selection import StratifiedKFold

        bands = [(8, 12), (12, 16), (16, 20), (20, 24), (24, 28)]
        bands = [band for band in bands if band[1] < rate / 2 and band[0] >= config.low_hz and band[1] <= config.high_hz]
        if not bands:
            bands = [(config.low_hz, config.high_hz)]
        x_train, x_test, filters = _csp_features(train, test, labels, bands, rate)
        choices = [count for count in (4, 8, 12, 16, 20) if count <= x_train.shape[1]]
        scores_by_count = {count: [] for count in choices}
        folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=config.seed)
        for inner_train, inner_test in folds.split(train, labels):
            fit, validation, _ = _csp_features(train[inner_train], train[inner_test],
                                                labels[inner_train], bands, rate)
            ranking = np.argsort(mutual_info_classif(fit, labels[inner_train], random_state=config.seed))
            for count in choices:
                columns = ranking[-count:]
                candidate = LinearDiscriminantAnalysis().fit(fit[:, columns], labels[inner_train])
                scores_by_count[count].append(balanced_accuracy_score(labels[inner_test],
                                                                        candidate.predict(validation[:, columns])))
        count = max(choices, key=lambda value: (np.mean(scores_by_count[value]), -value))
        scores = mutual_info_classif(x_train, labels, random_state=config.seed)
        selected = np.argsort(scores)[-count:]
        model = LinearDiscriminantAnalysis()
        model.fit(x_train[:, selected], labels)
        np.savez(artifact, bands=np.asarray(bands), filters=np.stack(filters),
                 selected_features=selected, selected_count=count,
                 classes=model.classes_, means=model.means_, priors=model.priors_,
                 scalings=model.scalings_, intercept=model.intercept_, coef=model.coef_)
        return model.predict(x_test[:, selected])


class EEGNet:
    name = "EEGNet"

    def fit_predict(self, train, test, labels, rate, config, artifact):
        import numpy as np
        import torch
        from torch import nn

        torch.manual_seed(config.seed)
        torch.set_num_threads(1)
        channels = train.shape[1]
        device = compute_device()
        network = eegnet_network(channels, train.shape[-1]).to(device)
        optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
        loss = nn.CrossEntropyLoss()
        x_train = _filter(train, rate, (config.low_hz, config.high_hz))
        x_test = _filter(test, rate, (config.low_hz, config.high_hz))
        scale = max(float(np.std(x_train)), 1e-8)
        inputs = torch.tensor((x_train / scale)[:, None], dtype=torch.float32, device=device)
        targets = torch.tensor(labels, dtype=torch.long, device=device)
        network.train()
        for _ in range(config.eegnet_epochs):
            for indices in torch.randperm(len(inputs)).split(32):
                optimizer.zero_grad()
                result = network(inputs[indices])
                loss(result, targets[indices]).backward()
                optimizer.step()
        network.eval()
        with torch.no_grad():
            prediction = network(torch.tensor((x_test / scale)[:, None], dtype=torch.float32,
                                              device=device)).argmax(1).cpu().numpy()
        network.to("cpu")
        torch.save({"state_dict": network.state_dict(), "scale": scale, "channels": channels,
                    "samples": train.shape[-1]}, artifact)
        return prediction


def compute_device():
    import torch

    return "cuda" if os.environ.get("MI_BCI_FORCE_CPU") != "1" and torch.cuda.is_available() else "cpu"


def eegnet_network(channels, samples):
    import torch
    from torch import nn

    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 8, (1, 31), padding=(0, 15), bias=False), nn.BatchNorm2d(8),
                nn.Conv2d(8, 16, (channels, 1), groups=8, bias=False), nn.BatchNorm2d(16),
                nn.ELU(), nn.AvgPool2d((1, 4)),
                nn.Conv2d(16, 16, (1, 15), padding=(0, 7), groups=16, bias=False),
                nn.Conv2d(16, 16, 1, bias=False), nn.BatchNorm2d(16), nn.ELU(),
                nn.AvgPool2d((1, 8)), nn.Flatten())
            with torch.no_grad():
                width = self.features(torch.zeros(1, 1, channels, samples)).shape[1]
            self.head = nn.Linear(width, 2)

        def forward(self, data):
            return self.head(self.features(data))

    return Network()


PROVIDERS = {provider.name: provider for provider in (FBCSPLDA(), EEGNet())}


def project_training_config(project):
    history = project.config_history()
    for revision in reversed(history):
        if "research_training" in revision:
            return TrainingConfig(**revision["research_training"])
    return TrainingConfig()


def train_and_evaluate(project, trials, labels, channels, rate, algorithm, config=None,
                       recording_id=None, source=None, subject=None):
    import numpy as np
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
    from sklearn.model_selection import train_test_split

    config = config or TrainingConfig()
    config.validate(rate)
    if algorithm not in PROVIDERS:
        raise ValueError("unknown algorithm")
    trials, labels = np.asarray(trials, dtype=float), np.asarray(labels, dtype=int)
    if trials.ndim != 3 or len(trials) != len(labels) or len(channels) != trials.shape[1]:
        raise ValueError("trials must have shape (trial, channel, sample)")
    if set(labels.tolist()) != {0, 1} or min(np.bincount(labels)) < 4:
        raise ValueError("at least four left and four right trials are required")
    if not np.isfinite(trials).all():
        raise ValueError("trials contain nonfinite values")
    indices = np.arange(len(labels))
    train_idx, test_idx = train_test_split(indices, test_size=config.test_fraction,
                                           random_state=config.seed, stratify=labels)
    if source is None and recording_id:
        row = project.db.execute("SELECT source_kind, source_sha256 FROM recordings WHERE id = ?",
                                 (recording_id,)).fetchone()
        source = {"kind": row["source_kind"], "recording_id": recording_id,
                  "sha256": row["source_sha256"]}
        session = project.db.execute("SELECT session_id FROM calibration_sessions WHERE recording_id = ?",
                                     (recording_id,)).fetchone()
        if session:
            events = project.protocol_events(session["session_id"])
            source["calibration"] = {"session_id": session["session_id"],
                                     "protocol": events[0]["config"] if events else None,
                                     "event_count": len(events)}
    source = source or {"kind": "in_memory"}
    artifacts = project.root / "models"
    artifacts.mkdir(exist_ok=True)
    digest = hashlib.sha256(np.ascontiguousarray(trials).tobytes() + labels.tobytes()).hexdigest()
    import uuid
    suffix = ".npz" if algorithm == "FBCSP+LDA" else ".pt"
    artifact = artifacts / (uuid.uuid4().hex + suffix)
    try:
        prediction = PROVIDERS[algorithm].fit_predict(trials[train_idx], trials[test_idx],
                                                        labels[train_idx], rate, config, artifact)
        metrics = {"accuracy": float(accuracy_score(labels[test_idx], prediction)),
                   "balanced_accuracy": float(balanced_accuracy_score(labels[test_idx], prediction)),
                   "confusion_matrix": confusion_matrix(labels[test_idx], prediction, labels=[0, 1]).tolist(),
                   "train_trials": len(train_idx), "test_trials": len(test_idx),
                   "train_indices": train_idx.tolist(), "test_indices": test_idx.tolist()}
        provenance = {"source": source, "data_sha256": digest, "channels": list(channels),
                      "sampling_rate": rate, "preprocessing": {"bandpass_hz": [config.low_hz, config.high_hz],
                      "epoch_seconds": [config.epoch_start, config.epoch_end]},
                      "hyperparameters": asdict(config), "evaluation": "stratified_holdout",
                      "labels": {"0": "left", "1": "right"},
                      "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()}
        for row in project.db.execute("SELECT id, config_json FROM config_history ORDER BY id DESC"):
            if "research_training" in json.loads(row["config_json"]):
                provenance["project_config_revision"] = row["id"]
                break
        return project.register_model(algorithm, subject or project.subject, recording_id,
                                      str(artifact.relative_to(project.root)), provenance, metrics)
    except Exception:
        artifact.unlink(missing_ok=True)
        raise


def generate_report(project, destination: str | Path):
    """Write a self-contained HTML report with provenance and a confusion-matrix figure."""
    destination = Path(destination)
    escape = lambda value: html.escape(str(value))
    rows = project.db.execute("SELECT id, source_kind, source_sha256, sampling_rate, channels_json FROM recordings ORDER BY created_at").fetchall()
    parts = ["<!doctype html><html lang='zh'><meta charset='utf-8'><title>运动想象实验报告</title>",
             "<style>body{font:15px sans-serif;max-width:920px;margin:32px auto;color:#273b39}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:8px;text-align:left}h2{margin-top:32px}</style>",
             f"<h1>运动想象实验报告</h1><p>项目：{escape(project.name)}；受试者代号：{escape(project.subject)}</p>",
             "<h2>设备与数据来源</h2><table><tr><th>记录</th><th>来源</th><th>采样率</th><th>SHA-256</th></tr>"]
    device = next((revision["device"] for revision in reversed(project.config_history()) if "device" in revision), None)
    parts.insert(-1, f"<p>设备：{escape(json.dumps(device, ensure_ascii=False) if device else '未记录设备型号')}</p>")
    for row in rows:
        parts.append("<tr>" + "".join(f"<td>{escape(value)}</td>" for value in
                    (row["id"], row["source_kind"], row["sampling_rate"], row["source_sha256"] or "无原文件")) + "</tr>")
    parts.append("</table><h2>实验方案与事件</h2>")
    events = project.protocol_events()
    parts.append(f"<p>已记录事件：{len(events)}；会话：{len(set(event['session_id'] for event in events))}</p>")
    for event in events:
        parts.append(f"<p>{escape(event['session_id'])} / {event['elapsed']:.2f}s / {escape(event['code'])} / {escape(event['config_json'])}</p>")
    parts.append("<h2>模型、预处理与评估</h2>")
    for model in project.models():
        metrics, provenance = model["metrics"], model["provenance"]
        parts.append(f"<h3>{escape(model['algorithm'])} · {escape(model['id'])}</h3><p>适用受试者：{escape(model['subject'])}；准确率：{metrics['accuracy']:.3f}；平衡准确率：{metrics['balanced_accuracy']:.3f}</p>")
        parts.append(f"<pre>{escape(json.dumps(provenance, ensure_ascii=False, indent=2))}</pre>")
        matrix = metrics["confusion_matrix"]
        parts.append("<figure><figcaption>混淆矩阵（行：真实左/右；列：预测左/右）</figcaption><svg width='320' height='230' role='img' aria-label='混淆矩阵'>")
        for row in range(2):
            for col in range(2):
                value = int(matrix[row][col])
                parts.append(f"<rect x='{50 + col * 100}' y='{10 + row * 100}' width='96' height='96' fill='{('#b8ddd8' if row == col else '#f2d4bf')}'/><text x='{98 + col * 100}' y='{64 + row * 100}' text-anchor='middle' font-size='22'>{value}</text>")
        parts.append("</svg></figure>")
    parts.append("<h2>模型激活历史</h2><pre>" + escape(json.dumps(project.activation_history(), ensure_ascii=False, indent=2)) + "</pre>")
    parts.append("<h2>控制日志</h2>")
    control_events = [record for record in project.audit_records() if record["action"].startswith("online_")]
    if control_events:
        parts.append("<table><tr><th>时间</th><th>事件</th><th>详情</th></tr>")
        for record in control_events:
            parts.append("<tr>" + "".join(f"<td>{escape(value)}</td>" for value in
                         (record["created_at"], record["action"],
                          json.dumps(record["details"], ensure_ascii=False))) + "</tr>")
        parts.append("</table>")
    else:
        parts.append("<p>尚无在线控制记录。</p>")
    parts.append("</html>")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(parts), encoding="utf-8")
    with project.db:
        project._audit("report_generated", {"path": str(destination.relative_to(project.root)) if destination.is_relative_to(project.root) else destination.name})
    return destination
