import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from mi_bci import Channel, EEGBlock, EEGEvent, Project
from mi_bci.research import (ProtocolConfig, TrainingConfig, generate_report,
                             compute_device, load_iv2a, load_recording, train_and_evaluate)


class ResearchTests(unittest.TestCase):
    def test_compute_device_requires_cuda_detection_and_supports_cpu_override(self):
        with patch("torch.cuda.is_available", return_value=True):
            with patch.dict("os.environ", {"MI_BCI_FORCE_CPU": "1"}):
                self.assertEqual(compute_device(), "cpu")
            with patch.dict("os.environ", {"MI_BCI_FORCE_CPU": "0"}):
                self.assertEqual(compute_device(), "cuda")
        with patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(compute_device(), "cpu")

    def test_protocol_events_survive_reopen(self):
        config = ProtocolConfig(trials_per_side=2, prepare_seconds=0.1, cue_seconds=0.1,
                                imagery_seconds=0.1, rest_seconds=0.1)
        schedule = list(config.schedule())
        self.assertEqual(list(ProtocolConfig().schedule())[-1][0], 360)
        self.assertEqual([code for _, code, _ in schedule].count("left"), 2)
        self.assertEqual([code for _, code, _ in schedule].count("right"), 2)
        self.assertEqual([code for _, code, _ in schedule].count("rest"), 4)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "project"
            project = Project.create(root, "Study")
            for elapsed, code, _ in schedule:
                project.record_protocol_event("session", elapsed, code, config.__dict__)
            project.close()
            project = Project.open(root)
            self.assertEqual(len(project.protocol_events("session")), len(schedule))
            self.assertEqual(project.protocol_events("session")[1]["config"]["cue_seconds"], 0.1)
            project.close()

    def test_official_mat_labels_and_recording_calibration(self):
        import numpy as np
        from scipy.io import savemat

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "A01T.mat"
            run = {"X": np.arange(3000 * 25, dtype=float).reshape(3000, 25),
                   "trial": np.asarray([1, 251, 501, 751, 1001, 1251]),
                   "y": np.asarray([1, 2, 3, 4, 1, 2]), "artifacts": np.asarray([0, 0, 0, 0, 1, 0]),
                   "fs": 250, "classes": np.asarray(["left hand", "right hand", "feet", "tongue"])}
            savemat(path, {"data": np.asarray([run], dtype=object)})
            config = TrainingConfig(epoch_start=0, epoch_end=0.5)
            trials, labels, channels, rate = load_iv2a(path, config)
            self.assertEqual(trials.shape, (3, 22, 125))
            self.assertEqual(labels.tolist(), [0, 1, 1])
            self.assertEqual(channels[7], "C3")
            self.assertEqual(rate, 250)

            project = Project.create(Path(directory) / "study", "Study")
            blocks = []
            for index in range(8):
                timestamps = tuple((index * 200 + sample) / 100 for sample in range(200))
                blocks.append(EEGBlock(tuple((0.000001 * (index + 1), 0.0) for _ in range(200)),
                                       timestamps, (Channel("C3", "V"), Channel("C4", "V")), 100,
                                       (EEGEvent(0, "left" if index % 2 == 0 else "right", timestamps[0]),)))
            identifier = project.record("synthetic", blocks)
            epochs, labels, channels, rate = load_recording(project, identifier,
                                                            TrainingConfig(epoch_start=0, epoch_end=1))
            self.assertEqual(epochs.shape, (8, 2, 100))
            self.assertEqual(epochs[0, 0, 0], 1)
            self.assertEqual(labels.tolist(), [0, 1] * 4)
            project.close()

    def test_providers_registry_and_report_share_evaluation(self):
        import numpy as np

        rate = 100
        rng = np.random.default_rng(4)
        time = np.arange(200) / rate
        trials = rng.normal(0, 0.15, (24, 4, 200))
        labels = np.asarray([0, 1] * 12)
        for index, label in enumerate(labels):
            trials[index, label * 2] += np.sin(2 * np.pi * 10 * time)
        config = TrainingConfig(epoch_start=0, epoch_end=2, eegnet_epochs=2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "project"
            project = Project.create(root, "Study")
            first = train_and_evaluate(project, trials, labels, ["C3", "C4", "Cz", "Pz"],
                                       rate, "FBCSP+LDA", config)
            second = train_and_evaluate(project, trials, labels, ["C3", "C4", "Cz", "Pz"],
                                        rate, "EEGNet", config)
            models = project.models()
            self.assertEqual({model["algorithm"] for model in models}, {"FBCSP+LDA", "EEGNet"})
            self.assertEqual(models[0]["metrics"]["test_indices"], models[1]["metrics"]["test_indices"])
            self.assertEqual(models[0]["provenance"]["data_sha256"], models[1]["provenance"]["data_sha256"])
            for model in models:
                self.assertTrue((root / model["artifact"]).is_file())
                self.assertGreaterEqual(model["metrics"]["accuracy"], 0)
            with self.assertRaises(ValueError):
                project.activate_model(first, "other", ["C3", "C4", "Cz", "Pz"], rate)
            with self.assertRaises(ValueError):
                project.activate_model(first, "01", ["C4", "C3", "Cz", "Pz"], rate)
            project.activate_model(first, "01", ["C3", "C4", "Cz", "Pz"], rate)
            project.activate_model(second, "01", ["C3", "C4", "Cz", "Pz"], rate)
            project.activate_model(first, "01", ["C3", "C4", "Cz", "Pz"], rate)
            self.assertEqual(project.active_model()["id"], first)
            report = generate_report(project, root / "reports" / "report.html").read_text(encoding="utf-8")
            for required in ("运动想象实验报告", "数据来源", "FBCSP+LDA", "EEGNet", "混淆矩阵", "模型激活历史"):
                self.assertIn(required, report)
            with project.db:
                project._audit("online_command", {"command": "STOP", "acknowledged": True})
            report = generate_report(project, root / "reports" / "report.html").read_text(encoding="utf-8")
            self.assertIn("online_command", report)
            self.assertIn("STOP", report)
            project.close()
            project = Project.open(root)
            self.assertEqual(project.active_model()["id"], first)
            self.assertEqual(len(project.activation_history()), 3)
            (root / models[0]["artifact"]).write_bytes(b"changed")
            with self.assertRaises(ValueError):
                project.activate_model(first, "01", ["C3", "C4", "Cz", "Pz"], rate)
            project.close()

    def test_calibration_model_retains_protocol_session(self):
        import math
        import random

        with tempfile.TemporaryDirectory() as directory:
            project = Project.create(Path(directory) / "study", "Study")
            protocol = ProtocolConfig(trials_per_side=12)
            project.record_protocol_event("session-1", 0, "prepare", protocol.__dict__)
            capture = project.begin_live_recording("calibration_synthetic", "session-1")
            for index in range(24):
                label = "left" if index % 2 == 0 else "right"
                noise = random.Random(index)
                timestamps = tuple((index * 200 + sample) / 100 for sample in range(200))
                samples = tuple(((math.sin(2 * math.pi * 10 * sample / 100) if label == "left" else 0.1)
                                 + noise.gauss(0, 0.05),
                                 (math.sin(2 * math.pi * 20 * sample / 100) if label == "right" else 0.1)
                                 + noise.gauss(0, 0.05))
                                for sample in range(200))
                capture.append(EEGBlock(samples, timestamps, (Channel("C3"), Channel("C4")), 100,
                                        (EEGEvent(0, label, timestamps[0]),)))
            identifier = capture.finish()
            config = TrainingConfig(epoch_start=0, epoch_end=2)
            trials, labels, channels, rate = load_recording(project, identifier, config)
            model_id = train_and_evaluate(project, trials, labels, channels, rate, "FBCSP+LDA",
                                          config, recording_id=identifier)
            model = next(item for item in project.models() if item["id"] == model_id)
            self.assertEqual(model["provenance"]["source"]["calibration"]["session_id"], "session-1")
            self.assertEqual(model["provenance"]["source"]["calibration"]["protocol"]["trials_per_side"], 12)
            project.close()


if __name__ == "__main__":
    unittest.main()
