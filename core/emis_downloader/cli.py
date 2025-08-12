from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import List

from emis_api import EmisDocuments

from .fetcher import fetch_all
from .models import EnrichedDoc
from .resolver import build_search_params
from .sentiment import VaderSentimentScorer
from .storage import SQLiteStore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download historical EMIS documents to SQLite"
    )
    p.add_argument(
        "--api-key",
        default=os.getenv("EMIS_API_KEY"),
        help="EMIS API key (or set EMIS_API_KEY)",
    )
    p.add_argument("--keyword", required=True, help="Keyword to search for (required)")
    p.add_argument("--country", help="Country name (optional)")
    p.add_argument("--company", help="Company name (optional)")
    p.add_argument("--industry", help="Industry name (optional)")
    p.add_argument(
        "--created-from", dest="created_from", help="ISO date YYYY-MM-DD (optional)"
    )
    p.add_argument(
        "--created-to", dest="created_to", help="ISO date YYYY-MM-DD (optional)"
    )
    p.add_argument("--order", default="date:desc", help="Sort order, default date:desc")
    p.add_argument("--batch-size", type=int, default=100, help="API page size (limit)")
    p.add_argument(
        "--max-records", type=int, default=1000, help="Max documents to fetch"
    )
    p.add_argument(
        "--with-sentiment",
        action="store_true",
        help="Compute VADER sentiment for each doc",
    )
    p.add_argument(
        "--db",
        type=Path,
        default=Path("data/emis_docs.sqlite"),
        help="SQLite path (default: data/emis_docs.sqlite)",
    )
    p.add_argument(
        "--dry-run", action="store_true", help="Fetch but do not write to DB"
    )

    args = p.parse_args()
    if not args.api_key:
        p.error("--api-key not provided and EMIS_API_KEY is not set")
    return args


def main() -> None:
    args = parse_args()

    client = EmisDocuments(api_key=args.api_key)

    base_params, ctx = build_search_params(
        client,
        keyword=args.keyword,
        country=args.country,
        company=args.company,
        industry=args.industry,
        created_from=args.created_from,
        created_to=args.created_to,
        order=args.order,
        batch_size=args.batch_size,
    )

    raw_docs = fetch_all(
        client=client,
        base_params=base_params,
        max_records=args.max_records,
        batch_size=args.batch_size,
    )

    if args.with_sentiment:
        scorer = VaderSentimentScorer()
        enriched: List[EnrichedDoc] = []
        for d in raw_docs:
            s, score = scorer.score(f"{d.title}. {d.abstract}")
            enriched.append(
                EnrichedDoc(**asdict(d), sentiment=s, sentiment_score=score)
            )
    else:
        enriched = [EnrichedDoc(**asdict(d)) for d in raw_docs]

    if not args.dry_run:
        with SQLiteStore(args.db) as store:
            store.upsert(enriched, ctx)

    print(
        json.dumps(
            {
                "count": len(enriched),
                "with_sentiment": bool(args.with_sentiment),
                "db": None if args.dry_run else str(args.db),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
