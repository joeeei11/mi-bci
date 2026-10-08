"""Observable, advisory-only LangGraph workflows."""

import hashlib
import json
import time
import uuid
from typing import TypedDict

from .ai_retrieval import DocumentIndex


MODES = {"experiment": "实验分析", "training": "训练建议", "device": "设备诊断"}


class AgentState(TypedDict, total=False):
    mode: str
    question: str
    sources: list[dict]
    answer: str
    steps: list[dict]


class AgentWorkspace:
    def __init__(self, project):
        self.project = project
        self.index = DocumentIndex(project)
        project.db.execute("""CREATE TABLE IF NOT EXISTS ai_runs (
            id TEXT PRIMARY KEY, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            mode TEXT NOT NULL, question_sha256 TEXT NOT NULL, status TEXT NOT NULL,
            source_ids_json TEXT NOT NULL, steps_json TEXT NOT NULL,
            failure TEXT, approval TEXT NOT NULL
        )""")

    def run(self, mode: str, question: str) -> dict:
        if mode not in MODES or not question.strip():
            raise ValueError("choose an agent and enter a question")
        from langgraph.graph import END, StateGraph

        run_id = uuid.uuid4().hex[:12]
        steps = []

        def node(name, operation):
            def invoke(state):
                started = time.perf_counter()
                try:
                    result, summary = operation(state)
                    steps.append({"node": name, "summary": summary,
                                  "milliseconds": round((time.perf_counter() - started) * 1000, 2),
                                  "status": "ok"})
                    return result
                except Exception:
                    steps.append({"node": name, "summary": "工具运行失败",
                                  "milliseconds": round((time.perf_counter() - started) * 1000, 2),
                                  "status": "failed"})
                    raise
            return invoke

        def retrieve(state):
            sources = self.index.search(state["question"])
            return {"sources": sources}, f"检索到 {len(sources)} 条来源"

        def advice(state):
            sources = state["sources"]
            if not sources:
                answer = "缺少可引用资料，无法给出建议。"
            else:
                evidence = "\n".join(f"[{n}] {source['text'][:300]}" for n, source in enumerate(sources, 1))
                answer = f"{MODES[state['mode']]}（仅供研究者核查）：\n{evidence}"
            return {"answer": answer}, f"生成{MODES[state['mode']]}草案"

        def approval(state):
            return {}, "等待研究者确认；未修改配置或发送设备指令"

        graph = StateGraph(AgentState)
        graph.add_node("retrieve", node("retrieve", retrieve))
        for mode in MODES:
            graph.add_node(mode, node(mode, advice))
            graph.add_edge(mode, "approval")
        graph.add_node("approval", node("approval", approval))
        graph.set_entry_point("retrieve")
        graph.add_conditional_edges("retrieve", lambda state: state["mode"], {mode: mode for mode in MODES})
        graph.add_edge("approval", END)
        status, failure, sources, answer = "pending_approval", None, [], ""
        try:
            result = graph.compile().invoke({"mode": mode, "question": question})
            sources, answer = result["sources"], result["answer"]
            if not sources:
                status = "insufficient_evidence"
        except Exception as exc:
            status, failure = "failed", type(exc).__name__
        with self.project.db:
            self.project.db.execute("INSERT INTO ai_runs (id, mode, question_sha256, status, source_ids_json, steps_json, failure, approval) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                                    (run_id, mode, hashlib.sha256(question.encode()).hexdigest(), status,
                                     json.dumps([item["id"] for item in sources]),
                                     json.dumps(steps, ensure_ascii=False), failure,
                                     "pending" if status == "pending_approval" else "none"))
            self.project._audit("ai_run", {"run_id": run_id, "status": status})
        return {"id": run_id, "status": status, "answer": answer, "sources": sources,
                "steps": steps, "failure": failure}

    def decide(self, run_id: str, approved: bool) -> None:
        row = self.project.db.execute("SELECT status FROM ai_runs WHERE id = ?", (run_id,)).fetchone()
        if row is None or row["status"] != "pending_approval":
            raise ValueError("run is not awaiting approval")
        decision = "approved" if approved else "rejected"
        with self.project.db:
            self.project.db.execute("UPDATE ai_runs SET status = ?, approval = ? WHERE id = ?",
                                    (decision, decision, run_id))
            self.project._audit("ai_approval", {"run_id": run_id, "decision": decision})

    def traces(self) -> list[dict]:
        return [dict(row) | {"source_ids": json.loads(row["source_ids_json"]),
                             "steps": json.loads(row["steps_json"])}
                for row in self.project.db.execute("SELECT * FROM ai_runs ORDER BY created_at DESC, rowid DESC")]
