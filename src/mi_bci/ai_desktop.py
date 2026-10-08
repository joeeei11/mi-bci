"""Researcher-facing AI workspace."""

import json
import re

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (QComboBox, QDialog, QFileDialog, QFormLayout, QHBoxLayout, QInputDialog,
                               QLabel, QLineEdit, QListWidget, QMessageBox, QPlainTextEdit,
                               QPushButton, QVBoxLayout)

from .ai_agents import AgentWorkspace
from .ai_providers import ModelProvider, _local
from .ai_retrieval import evaluate


class ModelWorker(QThread):
    completed = Signal(str)
    failed = Signal(str)

    def __init__(self, provider, prompt, confirmed):
        super().__init__()
        self.provider, self.prompt, self.confirmed = provider, prompt, confirmed

    def run(self):
        try:
            self.completed.emit(self.provider.generate(self.prompt, cloud_confirmed=self.confirmed))
        except Exception as exc:
            self.failed.emit(str(exc))


class AIWindow(QDialog):
    def __init__(self, project, parent=None):
        super().__init__(parent)
        self.project = project
        self.workspace = AgentWorkspace(project)
        self.last_run = None
        self.worker = None
        self.sources = []
        self.setWindowTitle("MI-BCI | AI 辅助研究")
        self.resize(900, 760)
        layout = QVBoxLayout(self)
        heading = QLabel("AI 辅助研究")
        heading.setStyleSheet("font-size: 22px; font-weight: 600")
        layout.addWidget(heading)

        row = QHBoxLayout()
        add = QPushButton("导入资料")
        add.clicked.connect(self.add_document)
        row.addWidget(add)
        evaluation = QPushButton("检索评测")
        evaluation.clicked.connect(self.show_evaluation)
        row.addWidget(evaluation)
        row.addStretch()
        layout.addLayout(row)
        self.documents = QLabel("尚未导入资料")
        layout.addWidget(self.documents)
        self.refresh_documents()

        form = QFormLayout()
        self.mode = QComboBox()
        self.mode.addItem("项目问答", "question")
        self.mode.addItem("实验分析", "experiment")
        self.mode.addItem("训练建议", "training")
        self.mode.addItem("设备诊断", "device")
        form.addRow("工作方式", self.mode)
        self.provider = QComboBox()
        self.provider.addItems(["证据摘录", "Ollama", "OpenAI 兼容"])
        self.provider.currentIndexChanged.connect(self.provider_changed)
        form.addRow("回答方式", self.provider)
        self.endpoint = QLineEdit("http://localhost:11434")
        form.addRow("模型地址", self.endpoint)
        self.model = QLineEdit()
        self.model.setPlaceholderText("例如 qwen2.5:7b")
        form.addRow("模型名称", self.model)
        self.credential = QPushButton("设置访问密钥")
        self.credential.clicked.connect(self.set_credential)
        form.addRow("系统凭据", self.credential)
        layout.addLayout(form)
        self.provider_changed()

        self.question = QPlainTextEdit()
        self.question.setPlaceholderText("输入要核查的问题")
        self.question.setMaximumHeight(90)
        layout.addWidget(self.question)
        self.ask_button = QPushButton("检索并回答")
        self.ask_button.clicked.connect(self.ask)
        layout.addWidget(self.ask_button)
        self.answer = QPlainTextEdit()
        self.answer.setReadOnly(True)
        layout.addWidget(self.answer, 2)
        self.citations = QPlainTextEdit()
        self.citations.setReadOnly(True)
        self.citations.setMaximumHeight(105)
        layout.addWidget(QLabel("来源"))
        layout.addWidget(self.citations)

        approval_row = QHBoxLayout()
        self.approve_button = QPushButton("确认已审核建议")
        self.reject_button = QPushButton("拒绝建议")
        self.approve_button.clicked.connect(lambda: self.decide(True))
        self.reject_button.clicked.connect(lambda: self.decide(False))
        approval_row.addWidget(self.approve_button)
        approval_row.addWidget(self.reject_button)
        approval_row.addStretch()
        layout.addLayout(approval_row)
        self.traces = QListWidget()
        self.traces.currentRowChanged.connect(self.show_trace)
        self.trace_detail = QPlainTextEdit()
        self.trace_detail.setReadOnly(True)
        self.trace_detail.setMaximumHeight(105)
        layout.addWidget(QLabel("运行轨迹"))
        layout.addWidget(self.traces, 1)
        layout.addWidget(self.trace_detail)
        self.refresh_traces()

    def provider_changed(self):
        enabled = self.provider.currentIndex() != 0
        self.endpoint.setEnabled(enabled)
        self.model.setEnabled(enabled)
        self.credential.setEnabled(self.provider.currentIndex() == 2)
        if self.provider.currentIndex() == 1:
            self.endpoint.setText("http://localhost:11434")

    def refresh_documents(self):
        count = self.project.db.execute("SELECT count(*) FROM ai_documents").fetchone()[0]
        self.documents.setText(f"已索引 {count} 份资料")

    def add_document(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "导入研究资料", "", "资料 (*.txt *.md *.pdf)")
        for path in paths:
            try:
                self.workspace.index.add_file(path)
            except Exception as exc:
                QMessageBox.warning(self, "导入失败", str(exc))
        self.refresh_documents()

    def show_evaluation(self):
        result = evaluate(self.workspace.index)
        self.trace_detail.setPlainText("\n".join(
            f"{row['question']}：{'找到来源' if row['has_citation'] else '缺少来源'}；"
            f"{'主题命中' if row['topic_found'] else '主题未命中'}" for row in result))

    def set_credential(self):
        secret, ok = QInputDialog.getText(self, "系统凭据", "访问密钥", QLineEdit.EchoMode.Password)
        if ok and secret:
            try:
                ModelProvider("openai", self.endpoint.text().strip(), self.model.text().strip()).set_credential(secret)
                QMessageBox.information(self, "已保存", "密钥已存入系统凭据库")
            except Exception as exc:
                QMessageBox.warning(self, "保存失败", str(exc))

    def ask(self):
        question = self.question.toPlainText().strip()
        if not question:
            return
        mode = self.mode.currentData()
        try:
            if mode == "question":
                result = self.workspace.index.answer(question)
            else:
                result = self.workspace.run(mode, question)
                self.last_run = result["id"] if result["status"] == "pending_approval" else None
                self.refresh_traces()
            self.sources = result["sources"]
            self.answer.setPlainText(result["answer"] or result.get("failure", ""))
            self.citations.setPlainText("\n".join(
                f"[{n}] {source['title']} · 段落 {source['position']} · SHA-256 {source['sha256'][:12]}"
                for n, source in enumerate(self.sources, 1)))
            if self.provider.currentIndex() and self.sources:
                self.generate_model_answer(question)
        except Exception as exc:
            QMessageBox.warning(self, "运行失败", str(exc))

    def generate_model_answer(self, question):
        kind = "ollama" if self.provider.currentIndex() == 1 else "openai"
        provider = ModelProvider(kind, self.endpoint.text().strip(), self.model.text().strip())
        confirmed = _local(provider.endpoint)
        if not confirmed:
            confirmed = QMessageBox.question(
                self, "发送至云模型", "以下问题和引用资料将发送给远程模型。确认发送？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes
            if not confirmed:
                return
        evidence = "\n".join(f"[{n}] {source['text'][:500]}" for n, source in enumerate(self.sources, 1))
        prompt = f"仅根据以下证据回答，逐项标注 [编号]。没有证据则说明不足。\n问题：{question}\n证据：\n{evidence}"
        self.ask_button.setEnabled(False)
        self.worker = ModelWorker(provider, prompt, confirmed)
        self.worker.completed.connect(self.model_completed)
        self.worker.failed.connect(self.model_failed)
        self.worker.finished.connect(lambda: self.ask_button.setEnabled(True))
        self.worker.start()

    def model_completed(self, text):
        cited = {int(number) for number in re.findall(r"\[(\d+)\]", text)}
        if cited and cited <= set(range(1, len(self.sources) + 1)):
            self.answer.setPlainText("模型草稿（请核查引用）：\n" + text)
        else:
            QMessageBox.warning(self, "引用缺失", "模型回答没有有效来源编号，已保留证据摘录。")

    def model_failed(self, error):
        QMessageBox.warning(self, "模型不可用", error)

    def decide(self, approved):
        if not self.last_run:
            return
        if approved and QMessageBox.question(self, "审核建议", "确认已人工审核这条建议？此操作不会执行建议或控制设备。",
                                            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes:
            return
        self.workspace.decide(self.last_run, approved)
        self.last_run = None
        self.refresh_traces()

    def refresh_traces(self):
        self.trace_rows = self.workspace.traces()
        self.traces.clear()
        for row in self.trace_rows:
            self.traces.addItem(f"{row['created_at']} · {row['mode']} · {row['status']} · {row['id']}")
        pending = self.last_run is not None
        self.approve_button.setEnabled(pending)
        self.reject_button.setEnabled(pending)

    def show_trace(self, index):
        if index < 0 or index >= len(self.trace_rows):
            return
        row = self.trace_rows[index]
        self.trace_detail.setPlainText(json.dumps({
            "sources": row["source_ids"], "steps": row["steps"],
            "failure": row["failure"], "approval": row["approval"]}, ensure_ascii=False, indent=2))

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.wait()
        super().closeEvent(event)
