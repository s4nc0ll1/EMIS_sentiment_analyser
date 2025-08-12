from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(slots=True)
class RawDoc:
    id: str
    date: str
    title: str
    abstract: str
    topics: List[Dict[str, str]]
    industries: List[Dict[str, str]]


@dataclass(slots=True)
class EnrichedDoc(RawDoc):
    sentiment: Optional[str] = None
    sentiment_score: Optional[float] = None


@dataclass(slots=True)
class SearchContext:
    country_code: Optional[str]
    company_id: Optional[str]
    matched_company_name: Optional[str]
    industry_code: Optional[str]
