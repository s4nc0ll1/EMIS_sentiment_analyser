from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

from .models import EnrichedDoc, SearchContext

SQUEMA_SQL = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS documents (
  id TEXT PRIMARY KEY,
  date TEXT,
  title TEXT,
  abstract TEXT,
  company_id TEXT,
  matched_company_name TEXT,
  country_code TEXT,
  industry_code TEXT,
  sentiment TEXT,
  sentiment_score REAL
);

CREATE TABLE IF NOT EXISTS document_topics (
  doc_id TEXT,
  topic_name TEXT,
  PRIMARY KEY (doc_id, topic_name)
);

CREATE TABLE IF NOT EXISTS document_industries (
  doc_id TEXT,
  industry_name TEXT,
  PRIMARY KEY (doc_id, industry_name)
);
"""


class SQLiteStor:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.executescript(SQUEMA_SQL)

    def upsert(self, docs: Iterable[EnrichedDoc], ctx: SearchContext) -> int:
        doc_rows = [
            (
                d.id,
                d.date,
                d.title,
                d.abstract,
                ctx.company_id,
                ctx.matched_company_name,
                ctx.country_code,
                ctx.industry_code,
                d.sentiment,
                d.sentiment_score,
            )
            for d in docs
        ]
        with self.conn:
            self.conn.executemany(
                """
                INSERT INTO documents (
                    id, date, title, abstract, company_id, matched_company_name,
                    country_code, industry_code, sentiment, sentiment_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    date=excluded.date,
                    title=excluded.title,
                    abstract=excluded.abstract,
                    company_id=excluded.company_id,
                    matched_company_name=excluded.matched_company_name,
                    country_code=excluded.country_code,
                    industry_code=excluded.industry_code,
                    sentiment=excluded.sentiment,
                    sentiment_score=excluded.sentiment_score
                ;
                """,
                doc_rows,
            )

            topic_rows = []
            industry_rows = []
            for d in docs:
                for t in d.topics or []:
                    name = t.get("name")
                    if name:
                        topic_rows.append((d.id, name))
                for i in d.industries or []:
                    name = i.get("name")
                    if name:
                        industry_rows.append((d.id, name))

            if topic_rows:
                self.conn.executemany(
                    """
                    INSERT INTO document_topics (doc_id, topic_name)
                    VALUES (?, ?)
                    ON CONFLICT(doc_id, topic_name) DO NOTHING;
                    """,
                    topic_rows,
                )
            if industry_rows:
                self.conn.executemany(
                    """
                    INSERT INTO document_industries (doc_id, industry_name)
                    VALUES (?, ?)
                    ON CONFLICT(doc_id, industry_name) DO NOTHING;
                    """,
                    industry_rows,
                )
        return len(doc_rows)

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
