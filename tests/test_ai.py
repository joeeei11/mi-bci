import json
import os
import sqlite3
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from mi_bci import Project
from mi_bci.ai_agents import AgentWorkspace
from mi_bci.ai_mcp import ReadOnlyProject, create_server
from mi_bci.ai_providers import ModelProvider
from mi_bci.ai_retrieval import DocumentIndex, evaluate

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class AIWorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.project = Project.create(Path(self.temp.name) / "project", "Study")
        self.addCleanup(self.project.close)

    def test_hybrid_retrieval_cites_persisted_source_and_handles_missing_evidence(self):
        index = DocumentIndex(self.project)
        identifier = index.add_text("sop.md", "设备断线时必须发送 STOP，并等待人工复位。\n\n训练前记录预处理参数。")
        index.add_text("unrelated.md", "实验室每周检查照明设备。")
        index.add_text("sop.md", "设备断线时必须发送 STOP，并等待人工复位。\n\n训练前记录预处理参数。")
        self.assertEqual(self.project.db.execute("SELECT count(*) FROM ai_documents").fetchone()[0], 2)
        answer = index.answer("设备断线后怎么办？")
        self.assertEqual(answer["sources"][0]["title"], "sop.md")
        self.assertIn("[1]", answer["answer"])
        self.assertTrue(answer["sources"][0]["sha256"])
        self.assertEqual(index.search("量子火箭"), [])
        self.assertEqual(index.answer("量子火箭")["sources"], [])
        self.assertEqual(len(evaluate(index)), 3)
        self.project.close()
        self.project = Project.open(Path(self.temp.name) / "project")
        self.assertEqual(DocumentIndex(self.project).search("STOP")[0]["title"], "sop.md")
        self.project.close()

    def test_agent_graph_persists_redacted_trace_and_requires_decision(self):
        workspace = AgentWorkspace(self.project)
        workspace.index.add_text("sop.md", "设备断线时发送 STOP。")
        before = len(self.project.config_history())
        run = workspace.run("device", "设备断线时怎么办？")
        self.assertEqual(run["status"], "pending_approval")
        self.assertEqual([step["node"] for step in run["steps"]], ["retrieve", "device", "approval"])
        self.assertEqual(len(self.project.config_history()), before)
        self.assertFalse(any(row["action"] == "online_command" for row in self.project.audit_records()))
        trace = workspace.traces()[0]
        self.assertEqual(trace["approval"], "pending")
        self.assertNotIn("设备断线时怎么办", json.dumps(trace, ensure_ascii=False))
        workspace.decide(run["id"], True)
        self.assertEqual(workspace.traces()[0]["approval"], "approved")
        with self.assertRaises(ValueError):
            workspace.decide(run["id"], True)

    def test_agent_without_evidence_and_failed_tool_cannot_be_approved(self):
        workspace = AgentWorkspace(self.project)
        empty = workspace.run("training", "没有相关资料")
        self.assertEqual(empty["status"], "insufficient_evidence")
        with self.assertRaises(ValueError):
            workspace.decide(empty["id"], True)
        with patch.object(workspace.index, "search", side_effect=RuntimeError("private data")):
            failed = workspace.run("experiment", "问题")
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(workspace.traces()[0]["steps"][0]["status"], "failed")
        self.assertNotIn("问题", json.dumps(workspace.traces(), ensure_ascii=False))
        self.assertNotIn("private data", json.dumps(workspace.traces(), ensure_ascii=False))

    def test_cloud_provider_confirmation_and_credentials_stay_outside_project(self):
        provider = ModelProvider("openai", "https://example.org/v1", "test")
        with self.assertRaises(PermissionError):
            provider.generate("private")
        with self.assertRaises(ValueError):
            ModelProvider("openai", "http://example.org/v1", "test")
        with patch("keyring.set_password") as save:
            provider.set_credential("secret-value")
            save.assert_called_once_with("mi-bci-ai", "https://example.org/v1", "secret-value")
        self.assertNotIn("secret-value", (self.project.root / "metadata.sqlite").read_bytes().decode("latin1"))
        with patch("keyring.get_password", return_value="secret-value"), patch("urllib.request.urlopen") as send:
            send.return_value.__enter__.return_value = BytesIO(b'{"choices":[{"message":{"content":"answer [1]"}}]}')
            self.assertEqual(provider.generate("question", cloud_confirmed=True), "answer [1]")
            request = send.call_args.args[0]
            self.assertEqual(request.get_header("Authorization"), "Bearer secret-value")
            self.assertEqual(json.loads(request.data)["messages"][0]["content"], "question")
        local = ModelProvider("ollama", "http://localhost:11434", "small")
        with patch("urllib.request.urlopen") as send:
            send.return_value.__enter__.return_value = BytesIO(b'{"response":"local [1]"}')
            self.assertEqual(local.generate("question"), "local [1]")

    def test_mcp_exposes_only_read_operations_and_rejects_path_escape(self):
        reader = ReadOnlyProject(self.project.root)
        self.assertEqual(reader.project()["name"], "Study")
        self.assertEqual(reader.models(), [])
        self.assertEqual(reader.reports(), [])
        with self.assertRaises(ValueError):
            reader.report("../metadata.sqlite")
        with self.assertRaises(sqlite3.OperationalError):
            reader.db.execute("UPDATE project SET name = 'changed'")
        server = create_server(self.project.root)
        tools = {tool.name for tool in __import__("asyncio").run(server.list_tools())}
        self.assertEqual(tools, {"get_project", "list_models", "list_reports", "read_report", "read_logs"})
        with self.assertRaises(Exception):
            __import__("asyncio").run(server.call_tool("activate_model", {"model_id": "x"}))
        reader.close()
        server.read_only_project.close()

    def test_desktop_can_answer_and_record_researcher_review(self):
        from PySide6.QtWidgets import QApplication
        from mi_bci.ai_desktop import AIWindow

        app = QApplication.instance() or QApplication([])
        window = AIWindow(self.project)
        window.workspace.index.add_text("sop.md", "设备断线时发送 STOP。")
        window.question.setPlainText("设备断线时怎么办？")
        window.mode.setCurrentIndex(3)
        window.ask()
        self.assertIn("sop.md", window.citations.toPlainText())
        self.assertTrue(window.approve_button.isEnabled())
        window.decide(False)
        self.assertEqual(window.workspace.traces()[0]["approval"], "rejected")
        window.close()
        app.processEvents()


if __name__ == "__main__":
    unittest.main()
