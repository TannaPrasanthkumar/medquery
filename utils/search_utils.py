"""
utils/search_utils.py
──────────────────────
Live web search integration using Tavily API.
Tavily is purpose-built for LLM agents — returns clean,
summarized results without scraping noise.

Free tier: 1,000 searches/month
Sign up at: https://tavily.com
"""

import logging
from typing import Optional
from tavily import TavilyClient
from config.config import TAVILY_API_KEY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────
#  Tavily Web Search
# ──────────────────────────────────────────────────────────

def web_search(query: str, max_results: int = 5) -> dict:
    """
    Perform a real-time web search using the Tavily API.

    Args:
        query       : The search query string.
        max_results : Number of results to fetch (max 10).

    Returns:
        dict with keys:
          - "answer"  : Tavily's AI-generated summary answer
          - "results" : List of {"title", "url", "content"} dicts
          - "error"   : Error message string if something went wrong, else None
    """
    try:
        if not TAVILY_API_KEY:
            return {
                "answer":  "",
                "results": [],
                "error":   "Tavily API key not configured. Add it to config/config.py.",
            }

        client  = TavilyClient(api_key=TAVILY_API_KEY)
        response = client.search(
            query=query,
            search_depth="advanced",    # deeper results for medical queries
            max_results=max_results,
            include_answer=True,        # Tavily generates a quick answer
            include_domains=[           # prioritize trusted medical sources
                "nih.gov",
                "mayoclinic.org",
                "medlineplus.gov",
                "cdc.gov",
                "who.int",
                "webmd.com",
                "healthline.com",
                "pubmed.ncbi.nlm.nih.gov",
                "fda.gov",
            ],
        )

        results = []
        for item in response.get("results", []):
            results.append({
                "title":   item.get("title",   "No title"),
                "url":     item.get("url",     "#"),
                "content": item.get("content", ""),
            })

        return {
            "answer":  response.get("answer", ""),
            "results": results,
            "error":   None,
        }

    except Exception as e:
        logger.error("web_search error: %s", e)
        return {
            "answer":  "",
            "results": [],
            "error":   str(e),
        }


def format_search_results(search_data: dict) -> str:
    """
    Format Tavily search results into a clean context string
    that can be injected into the LLM prompt.

    Args:
        search_data: Output dict from web_search().

    Returns:
        Formatted string of search context.
    """
    try:
        if search_data.get("error"):
            return f"Web search unavailable: {search_data['error']}"

        parts = []

        if search_data.get("answer"):
            parts.append(f"🔍 Web Summary:\n{search_data['answer']}")

        for i, result in enumerate(search_data.get("results", []), 1):
            title   = result.get("title",   "")
            url     = result.get("url",     "")
            content = result.get("content", "")[:400]   # cap snippet length
            parts.append(f"[{i}] {title}\nSource: {url}\n{content}")

        return "\n\n".join(parts) if parts else "No relevant web results found."

    except Exception as e:
        logger.error("format_search_results error: %s", e)
        return "Error formatting search results."


def build_search_citations(search_data: dict) -> list[dict]:
    """
    Build a list of citation dicts for display in the Streamlit UI.

    Args:
        search_data: Output dict from web_search().

    Returns:
        List of {"title": str, "url": str} dicts.
    """
    try:
        return [
            {"title": r.get("title", "Source"), "url": r.get("url", "#")}
            for r in search_data.get("results", [])
            if r.get("url")
        ]
    except Exception as e:
        logger.error("build_search_citations error: %s", e)
        return []
