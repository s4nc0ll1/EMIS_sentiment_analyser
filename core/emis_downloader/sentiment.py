from __future__ import annotations

from typing import Protocol, Tuple

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


class SentimentScorer(Protocol):
    def score(self, text: str) -> Tuple[str, float]: ...


class VaderSentimentScorer:
    """VADER-based sentiment scorer. Lazily ensures lexicon."""

    _analyzer: SentimentIntensityAnalyzer | None = None

    @staticmethod
    def _ensure() -> SentimentIntensityAnalyzer:
        if VaderSentimentScorer._analyzer is not None:
            return VaderSentimentScorer._analyzer
        try:
            analyzer = SentimentIntensityAnalyzer()
            analyzer.polarity_scores("ok")
        except Exception:
            nltk.download("vader_lexicon", quiet=True)
            analyzer = SentimentIntensityAnalyzer()
        VaderSentimentScorer._analyzer = analyzer
        return analyzer

    @staticmethod
    def _classify(compound: float) -> str:
        if compound >= 0.05:
            return "Positive"
        if compound <= -0.05:
            return "Negative"
        return "Neutral"

    def score(self, text: str) -> Tuple[str, float]:
        an = self._ensure()
        s = an.polarity_scores(text or "")
        return self._classify(float(s.get("compound", 0.0))), float(
            s.get("compound", 0.0)
        )
