import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from mi_bci import Channel, EEGBlock, EEGEvent, Project, ReplaySource, StreamState, SyntheticSource


def drain(source):
    source.start()
    blocks = []
    while (block := source.read()) is not None:
        blocks.append(block)
    return blocks


class FoundationTests(unittest.TestCase):
    def test_stream_contract_rejects_malformed_blocks(self):
        channel = (Channel("C3"),)
        with self.assertRaises(ValueError):
            EEGBlock(((1.0,),), (), channel, 250)
        with self.assertRaises(ValueError):
            EEGBlock(((1.0,), (2.0,)), (2.0, 1.0), channel, 250)
        with self.assertRaises(ValueError):
            EEGBlock(((1.0,),), (0.0,), channel, 250, (EEGEvent(1, "left", 0.0),))
        with self.assertRaises(ValueError):
            EEGBlock(((float("nan"),),), (0.0,), channel, 250)

    def test_synthetic_and_replay_preserve_samples_events_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Project.create(Path(directory) / "project", "Study")
            source = SyntheticSource(total_samples=8, block_size=3, seed=7,
                                     event_samples={2: "left", 5: "right"})
            generated = drain(source)
            self.assertEqual(source.state, StreamState.STOPPED)
            self.assertEqual(generated, drain(SyntheticSource(total_samples=8, block_size=3, seed=7,
                                                               event_samples={2: "left", 5: "right"})))
            recording_id = project.record("synthetic", generated)
            project.close()

            reopened = Project.open(Path(directory) / "project")
            self.assertEqual(generated, drain(ReplaySource(reopened, recording_id)))
            artifact = next((reopened.root / "derivatives").rglob("*_events.tsv"))
            self.assertIn("left", artifact.read_text(encoding="utf-8"))
            sidecar = json.loads(next((reopened.root / "derivatives").rglob("*_eeg.json")).read_text())
            self.assertEqual(sidecar["SourceKind"], "synthetic")
            reopened.close()

    def test_project_recovery_and_config_history(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "project"
            project = Project.create(path, "Study")
            project.set_config({"rate": 250})
            project.set_config({"rate": 500})
            block = drain(SyntheticSource(total_samples=2, block_size=2))[0]
            with self.assertRaises(ValueError):
                project.record("synthetic", [block, block])
            project.close()

            reopened = Project.open(path)
            self.assertEqual(reopened.config_history(), [{"rate": 250}, {"rate": 500}])
            recovered = reopened.db.execute("SELECT id, status FROM recordings").fetchone()
            self.assertEqual(recovered["status"], "recovered")
            self.assertEqual(reopened.blocks(recovered["id"]), [block])
            self.assertTrue(any(row["action"] == "recording_recovered" for row in reopened.audit_records()))
            self.assertTrue(any(row["action"] == "project_opened" for row in reopened.audit_records()))
            self.assertTrue(next((path / "derivatives").rglob("*_eeg.tsv")).is_file())
            reopened.close()

    def test_deidentified_export_omits_identity_raw_source_and_absolute_time(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "private-subject-name.csv"
            raw.write_text("private-subject-name", encoding="utf-8")
            project = Project.create(root / "project", "private-subject-name", "secret123")
            block = EEGBlock(((1.0,),), (1_700_000_000.0,), (Channel("private-subject-name"),), 250,
                             (EEGEvent(0, "private-subject-name", 1_700_000_000.0),))
            project.record("private-subject-name", [block], raw)
            output = project.export_deidentified(root / "export.zip")
            with zipfile.ZipFile(output) as archive:
                names = archive.namelist()
                joined = "\n".join(archive.read(name).decode() for name in names)
            self.assertNotIn("private-subject-name", joined)
            self.assertNotIn("secret123", joined)
            self.assertNotIn("1700000000", joined)
            self.assertNotIn("timestamp\":", joined)
            self.assertNotIn("sourcedata", "\n".join(names))
            self.assertIn("other", joined)
            self.assertIn("sub-01", "\n".join(names))
            self.assertTrue(any(row["action"] == "project_exported" for row in project.audit_records()))
            project.close()


if __name__ == "__main__":
    unittest.main()
