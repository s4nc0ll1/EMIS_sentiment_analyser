from __future__ import annotations

import logging
from typing import Any, Dict, List

from emis_api import APIEndpoint, EmisDocuments

from .models import RawDoc

log = logging.getLogger("emis_downloader.fetcher")


def fetch_all(
    client: EmisDocuments,
    base_params: Dict[str, Any],
    *,
    max_records: int,
    batch_size: int,
) -> List[RawDoc]:
    docs: List[RawDoc] = []
    offset = 0
    while len(docs) < max_records:
        params = dict(base_params)
        params.update(
            {"limit": min(batch_size, max_records - len(docs)), "offset": offset}
        )
        log.info("Fetching batch: offset=%s, limit=%s", offset, params["limit"])
        try:
            response = client.http_client.get(
                APIEndpoint.DOCUMENTS_SEARCH.value, params=params
            )
        except Exception as e:
            log.error("Batch failed at offset=%s: %s", offset, e)
            break
        items = (response or {}).get("data", {}).get("items", [])
        if not items:
            break
        for it in items:
            docs.append(
                RawDoc(
                    id=str(it.get("id", "")),
                    date=str(it.get("creationDate", "")),
                    title=it.get("title", "No title"),
                    abstract=it.get("abstract", ""),
                    topics=it.get("topics", []) or [],
                    industries=it.get("industries", []) or [],
                )
            )
        if len(items) < params["limit"]:
            break
        offset += params["limit"]
    return docs[:max_records]
