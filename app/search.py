"""LLM-driven search orchestration logic."""

from __future__ import annotations

import json
import logging
import os
import re
import time
import asyncio
from collections import OrderedDict
from typing import Any

from google import genai

from app.crud import get_products_by_ids
from app.db import DatabaseManager
from app.schemas import ProductRead

logger = logging.getLogger(__name__)

SEARCH_CACHE_SIZE = 128
SEARCH_RESULT_LIMIT = 20
LLM_PRODUCT_LIMIT = 100
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

_search_cache: OrderedDict[str, list[str]] = OrderedDict()
_gemini_client: genai.Client | None = None


class SearchBackendUnavailableError(Exception):
    """Raised when Gemini ranking is unavailable."""


class DatabaseUnavailableError(Exception):
    """Raised when PostgreSQL hydration fails."""


def _normalize_query(query: str) -> str:
    return " ".join(query.lower().split())


def _cache_get(query: str) -> list[str] | None:
    normalized = _normalize_query(query)
    cached = _search_cache.get(normalized)
    if cached is None:
        return None
    _search_cache.move_to_end(normalized)
    return cached


def _cache_set(query: str, ids: list[str]) -> None:
    normalized = _normalize_query(query)
    if normalized in _search_cache:
        _search_cache.move_to_end(normalized)
    _search_cache[normalized] = ids
    if len(_search_cache) > SEARCH_CACHE_SIZE:
        _search_cache.popitem(last=False)


def _tokenize(text: str) -> list[str]:
    raw_tokens = re.findall(r"\w+", text.lower())
    normalized: list[str] = []
    for token in raw_tokens:
        normalized.append(token)
        if len(token) > 4 and token.endswith("ies"):
            normalized.append(f"{token[:-3]}y")
        elif len(token) > 3 and token.endswith("es"):
            normalized.append(token[:-2])
        elif len(token) > 2 and token.endswith("s"):
            normalized.append(token[:-1])
    return normalized


def _serialize_products_for_llm(products: list[dict[str, Any]]) -> str:
    return json.dumps(products, ensure_ascii=True, separators=(",", ":"))


def _extract_json_array(raw_content: str) -> list[Any]:
    content = raw_content.strip()
    if not content:
        raise ValueError("OpenAI returned empty content")

    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\[[\s\S]*\]", content)
    if not match:
        raise ValueError("OpenAI response did not contain a JSON array")

    parsed = json.loads(match.group(0))
    if not isinstance(parsed, list):
        raise ValueError("OpenAI response did not decode to a JSON array")
    return parsed


def _keyword_score(query_tokens: list[str], product: dict[str, Any]) -> int:
    haystack = " ".join(
        [
            str(product["name"]),
            str(product["description"]),
            str(product["category"]),
            str(product["price"]),
        ]
    )
    haystack_tokens = _tokenize(haystack)
    return sum(haystack_tokens.count(token) for token in query_tokens)


def _select_llm_candidates(query: str, products: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(products) <= LLM_PRODUCT_LIMIT:
        return products

    query_lower = query.lower()
    query_tokens = _tokenize(query)
    scored = sorted(
        products,
        key=lambda product: (
            _keyword_score(query_tokens, product),
            -float(product["price"]) if "under" in query_lower or "cheap" in query_lower else 0.0,
        ),
        reverse=True,
    )
    return scored[:LLM_PRODUCT_LIMIT]


def _keyword_fallback_ids(query: str, products: list[dict[str, Any]]) -> list[str]:
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    scored_products: list[tuple[int, str]] = []
    for product in products:
        score = _keyword_score(query_tokens, product)
        if score > 0:
            scored_products.append((score, str(product["id"])))

    scored_products.sort(key=lambda item: item[0], reverse=True)
    return [product_id for _, product_id in scored_products[:SEARCH_RESULT_LIMIT]]


def _get_openai_client() -> AsyncOpenAI:
    global _gemini_client

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SearchBackendUnavailableError("GEMINI_API_KEY is not configured")

    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


async def get_llm_ranked_products(query: str, products: list[dict[str, Any]]) -> list[str]:
    """Return ranked product ids from Gemini for the given query and candidate products."""
    client = _get_openai_client()
    model = os.getenv("GEMINI_SEARCH_MODEL", DEFAULT_GEMINI_MODEL)
    serialized_products = _serialize_products_for_llm(products)

    prompt = (
        "You are a smart shopping assistant.\n\n"
        f"User query:\n{query}\n\n"
        "Here is a list of products:\n"
        f"{serialized_products}\n\n"
        "Return ONLY the top 20 most relevant products in ranked order. "
        "Return as JSON array of product IDs."
    )

    response = await asyncio.to_thread(
        client.models.generate_content,
        model=model,
        contents=prompt,
    )

    content = response.text or ""
    parsed_ids = _extract_json_array(content)

    allowed_ids = {str(product["id"]) for product in products}
    ranked_ids: list[str] = []
    seen_ids: set[str] = set()
    for value in parsed_ids:
        product_id = str(value)
        if product_id in allowed_ids and product_id not in seen_ids:
            ranked_ids.append(product_id)
            seen_ids.add(product_id)
        if len(ranked_ids) >= SEARCH_RESULT_LIMIT:
            break

    return ranked_ids


async def _fetch_all_products(db: DatabaseManager) -> list[dict[str, Any]]:
    try:
        async with db.connection() as connection:
            rows = await connection.fetch(
                """
                SELECT id, name, description, category, price
                FROM products
                ORDER BY created_at DESC
                """
            )
    except Exception as exc:
        raise DatabaseUnavailableError from exc

    return [
        {
            "id": str(row["id"]),
            "name": row["name"],
            "description": row["description"],
            "category": row["category"],
            "price": float(row["price"]),
        }
        for row in rows
    ]


async def search_products(*, query: str, db: DatabaseManager) -> list[ProductRead]:
    """Run PostgreSQL-backed search with OpenAI ranking and keyword fallback."""
    started = time.perf_counter()
    cached_ids = _cache_get(query)

    if cached_ids is None:
        products = await _fetch_all_products(db)
        if not products:
            logger.info("Search query='%s' returned no hits", query)
            return []

        llm_candidates = _select_llm_candidates(query, products)

        try:
            ranked_ids = await get_llm_ranked_products(query, llm_candidates)
        except Exception as exc:
            logger.warning("Gemini ranking failed for query='%s': %s", query, exc)
            ranked_ids = _keyword_fallback_ids(query, products)

        _cache_set(query, ranked_ids)
    else:
        ranked_ids = cached_ids[:SEARCH_RESULT_LIMIT]

    if not ranked_ids:
        logger.info("Search query='%s' returned no hits", query)
        return []

    try:
        async with db.connection() as connection:
            records = await get_products_by_ids(connection, ranked_ids)
    except Exception as exc:
        raise DatabaseUnavailableError from exc

    records_by_id = {str(record.id): record for record in records}
    ordered = [records_by_id[product_id] for product_id in ranked_ids if product_id in records_by_id]

    elapsed_ms = (time.perf_counter() - started) * 1000
    logger.info("Search query='%s' returned %s rows in %.2fms", query, len(ordered), elapsed_ms)
    return [ProductRead.model_validate(record) for record in ordered]


def clear_search_cache() -> None:
    """Invalidate the in-memory search cache."""
    _search_cache.clear()
