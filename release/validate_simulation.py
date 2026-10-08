"""Deterministic closed-loop timing and command checks; not physical validation."""

import hashlib
import json
import statistics
import tempfile
import time
from pathlib import Path

from mi_bci import Channel, EEGBlock, Project
from mi_bci.online import OnlineSession, VirtualSTM32


class Predictor:
    def predict(self, window):
        return ("LEFT", .95) if window[-1][0] > 0 else ("RIGHT", .95)


def run():
    with tempfile.TemporaryDirectory() as directory:
        project = Project.create(Path(directory) / "study", "Simulated validation")
        try:
            artifact = project.root / "model.bin"
            artifact.write_bytes(b"simulated predictor")
            model_id = project.register_model("simulated", project.subject, None, "model.bin",
                                              {"channels": ["C3", "C4"], "sampling_rate": 100,
                                               "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()}, {})
            project.activate_model(model_id, project.subject, ["C3", "C4"], 100)
            device = VirtualSTM32()
            simulated_time = [0.0]
            session = OnlineSession(project, device, predictor_factory=lambda *args: Predictor(),
                                    clock=lambda: simulated_time[0])
            session.start(["C3", "C4"], 100)
            latencies = []
            results = []
            sample_index = 0
            for trial in range(10):
                target = "LEFT" if trial % 2 == 0 else "RIGHT"
                amplitude = 1.0 if target == "LEFT" else -1.0
                for _ in range(16):
                    block = EEGBlock(tuple((amplitude, 0.0) for _ in range(25)),
                                     tuple((sample_index + n) / 100 for n in range(25)),
                                     (Channel("C3"), Channel("C4")), 100)
                    started = time.perf_counter()
                    session.feed(block)
                    latencies.append((time.perf_counter() - started) * 1000)
                    sample_index += 25
                    simulated_time[0] += .25
                    if session.state != "running":
                        raise RuntimeError("simulation entered fault state")
                results.append(device.command == target)
            session.stop()
            return {"mode": "simulated", "trials": len(results),
                    "success_rate": sum(results) / len(results),
                    "max_processing_ms": max(latencies),
                    "median_processing_ms": statistics.median(latencies),
                    "final_stop_confirmed": session.stop_confirmed,
                    "passed_simulated_targets": sum(results) / len(results) >= .8 and
                                                max(latencies) < 300 and session.stop_confirmed}
        finally:
            project.close()


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["passed_simulated_targets"] else 1)
