from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from emis_api import EmisDocuments

from .models import SearchContext


def build_search_params(
    client: EmisDocuments,
    *,
    keyword: str,
    country: Optional[str] = None,
    company: Optional[str] = None,
    industry: Optional[str] = None,
    created_from: Optional[str] = None,
    created_to: Optional[str] = None,
    order: str = "date:desc",
    batch_size: int = 100,
) -> Tuple[Dict[str, Any], SearchContext]:
    params: Dict[str, Any] = {"limit": batch_size, "order": order, "keyword": keyword}

    country_code = client.lookup_manager.get_country_code(country) if country else None
    if country_code:
        params["country"] = country_code

    company_id = None
    matched_company_name = None
    if company:
        match = client.company_matcher.match_company(company, country)
        if match:
            company_id = match.id
            matched_company_name = match.name
            params["company_ids"] = match.id

    industry_code = (
        client.lookup_manager.get_industry_code(industry) if industry else None
    )
    if industry_code:
        params["industry"] = industry_code

    if created_from:
        params["createdFrom"] = created_from
    if created_to:
        params["createdTo"] = created_to

    ctx = SearchContext(
        country_code=country_code,
        company_id=company_id,
        matched_company_name=matched_company_name,
        industry_code=industry_code,
    )
    return params, ctx
