"""Local document snapshots and cited hybrid retrieval."""

import hashlib
import math
import re
import uuid
from pathlib import Path


def _terms(value: str) -> list[str]:
    words = re.findall(r"[a-z0-9_]+|[\u3400-\u9fff]", value.lower())
    return words + [a + b for a, b in zip(words, words[1:]) if len(a) == len(b) == 1]


class DocumentIndex:
    def __init__(self, project):
        self.project = project
        self.db = project.db
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS ai_documents (
                id TEXT PRIMARY KEY, title TEXT NOT NULL, sha256 TEXT NOT NULL,
                source TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS ai_chunks (
                id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES ai_documents(id),
                ordinal INTEGER NOT NULL, text TEXT NOT NULL
            );
        """)

    def add_file(self, path: str | Path) -> str:
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix not in {".txt", ".md", ".pdf"}:
            raise ValueError("only TXT, Markdown, and PDF documents are supported")
        raw = path.read_bytes()
        if len(raw) > 20_000_000:
            raise ValueError("document exceeds 20 MB")
        if suffix == ".pdf":
            from pypdf import PdfReader
            text = "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
        else:
            text = raw.decode("utf-8-sig")
        return self.add_text(path.name, text, hashlib.sha256(raw).hexdigest())

    def add_text(self, title: str, content: str, digest: str | None = None) -> str:
        if not title.strip() or not content.strip():
            raise ValueError("document title and extractable text are required")
        digest = digest or hashlib.sha256(content.encode()).hexdigest()
        previous = self.db.execute("SELECT id FROM ai_documents WHERE title = ? AND sha256 = ?", (title, digest)).fetchone()
        if previous:
            return previous["id"]
        identifier = uuid.uuid4().hex[:12]
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", content) if part.strip()]
        chunks = []
        for paragraph in paragraphs:
            while paragraph:
                chunks.append(paragraph[:900])
                paragraph = paragraph[900:]
        with self.db:
            self.db.execute("INSERT INTO ai_documents (id, title, sha256, source) VALUES (?, ?, ?, ?)",
                            (identifier, title, digest, title))
            self.db.executemany("INSERT INTO ai_chunks VALUES (?, ?, ?, ?)",
                                [(uuid.uuid4().hex[:12], identifier, n + 1, chunk) for n, chunk in enumerate(chunks)])
            self.project._audit("ai_document_added", {"document_id": identifier, "sha256": digest})
        return identifier

    def search(self, question: str, limit: int = 5) -> list[dict]:
        if not question.strip():
            return []
        rows = self.db.execute("""SELECT c.id, c.ordinal, c.text, d.title, d.sha256
                                  FROM ai_chunks c JOIN ai_documents d ON d.id = c.document_id
                                  ORDER BY d.created_at, c.ordinal""").fetchall()
        if not rows:
            return []
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        corpus = [row["text"] for row in rows]
        vectors = TfidfVectorizer(analyzer="char", ngram_range=(2, 3)).fit_transform(corpus + [question])
        semantic = cosine_similarity(vectors[:-1], vectors[-1]).ravel()
        query_terms = set(_terms(question))
        documents = [set(_terms(text)) for text in corpus]
        frequencies = {word: sum(word in terms for terms in documents) for word in query_terms}
        ranked = []
        for row, terms, vector_score in zip(rows, documents, semantic):
            matched = query_terms & terms
            if not matched and vector_score <= 0:
                continue
            lexical = sum(math.log(1 + (len(rows) - frequencies[word] + .5) / (frequencies[word] + .5))
                          for word in matched) / max(1, len(query_terms))
            coverage = len(matched) / max(1, len(query_terms))
            score = .45 * float(vector_score) + .35 * lexical + .20 * coverage
            ranked.append((score, row))
        ranked.sort(key=lambda item: (-item[0], item[1]["title"], item[1]["ordinal"]))
        return [{"id": row["id"], "title": row["title"], "position": row["ordinal"],
                 "sha256": row["sha256"], "text": row["text"], "score": round(score, 5)}
                for score, row in ranked[:limit]]

    def answer(self, question: str) -> dict:
        sources = self.search(question)
        if not sources:
            return {"answer": "未找到可引用的项目资料，无法据此回答。", "sources": []}
        lines = [f"{source['text'][:300]} [{index}]" for index, source in enumerate(sources, 1)]
        return {"answer": "\n\n".join(lines), "sources": sources}


EVALUATION_QUESTIONS = (
    {"question": "模型使用了哪些预处理步骤？", "expected_topic": "预处理"},
    {"question": "实验提示阶段持续多久？", "expected_topic": "提示"},
    {"question": "设备断线后应如何处理？", "expected_topic": "断线"},
)


def evaluate(index: DocumentIndex, questions=EVALUATION_QUESTIONS) -> list[dict]:
    return [{"question": item["question"], "expected_topic": item["expected_topic"],
             "has_citation": bool(answer["sources"]),
             "topic_found": any(item["expected_topic"] in source["text"] for source in answer["sources"])}
            for item in questions for answer in [index.answer(item["question"])]]
