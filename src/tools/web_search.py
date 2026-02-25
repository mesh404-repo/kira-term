"""Web search tool for SuperAgent - searches the web using Firecrawl and Serper APIs."""

from __future__ import annotations

import os
import time
from typing import Optional

from src.tools.base import ToolResult

# Firecrawl API key - set your key here or via environment variable
os.environ["FIRECRAWL_API_KEY"] = "fc-8aed6895a4e8496288a5b6dde405f1d8"

# Serper.dev API key - set your key here or via environment variable
# Get your API key at https://serper.dev/
os.environ.setdefault("SERPER_API_KEY", "798be74f7c7540882f80f7ac93c059b31eefa755")

# Try to import httpx, fall back to requests
try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False
    try:
        import requests
        HAS_REQUESTS = True
    except ImportError:
        HAS_REQUESTS = False


def web_search(
    query: str,
    num_results: int = 5,
    search_type: str = "general",
    provider: str = "firecrawl",
) -> ToolResult:
    """Search the web for information using Firecrawl or Serper API.
    
    Args:
        query: Search query string
        num_results: Number of results to return (default: 5, max: 10)
        search_type: Type of search - 'general', 'code', 'docs', 'news', or 'images'
        provider: Search provider - 'auto', 'firecrawl', or 'serper'
        
    Returns:
        ToolResult with search results or error
    """
    if not query:
        return ToolResult.fail(
            "Missing required parameter 'query'. "
            "Usage: web_search(query: str, num_results?: int, search_type?: str, provider?: str)"
        )
    
    # Clamp num_results
    num_results = max(1, min(10, num_results))
    
    # Try specified provider or auto-detect
    if provider == "serper":
        result = _search_with_serper(query, num_results, search_type)
        if result is not None:
            return result
    elif provider == "firecrawl":
        result = _search_with_firecrawl(query, num_results, search_type)
        if result is not None:
            return result
        # If Firecrawl fails, fall back to Serper even if firecrawl was specified
        result = _search_with_serper(query, num_results, search_type)
        if result is not None:
            return result
    else:
        # Auto mode: try Serper first (faster, more reliable), then Firecrawl
        result = _search_with_serper(query, num_results, search_type)
        if result is not None:
            return result
        
        result = _search_with_firecrawl(query, num_results, search_type)
        if result is not None:
            return result
    
    return ToolResult.fail(
        "Web search unavailable. No search API configured. "
        "Set SERPER_API_KEY or FIRECRAWL_API_KEY environment variable."
    )


def _search_with_firecrawl(
    query: str,
    num_results: int,
    search_type: str,
) -> Optional[ToolResult]:
    """Search using Firecrawl API.
    
    Firecrawl provides web search with optional content scraping.
    API docs: https://docs.firecrawl.dev/features/search
    """
    api_key = os.environ.get("FIRECRAWL_API_KEY")
    if not api_key:
        return None
    
    url = "https://api.firecrawl.dev/v1/search"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    # Build payload
    payload = {
        "query": query,
        "limit": num_results,
    }
    
    # Don't use scrapeOptions by default - it causes timeouts
    # Only use basic search for faster, more reliable results
    # If content scraping is needed, it can be done separately
    
    # Retry up to 3 times with increasing delays
    max_retries = 3
    last_error = None
    
    for attempt in range(1, max_retries + 1):
        try:
            if HAS_HTTPX:
                # Reduced timeout to 30 seconds for faster failure and fallback
                with httpx.Client(timeout=30.0) as client:
                    response = client.post(url, headers=headers, json=payload)
                    response.raise_for_status()
                    data = response.json()
            elif HAS_REQUESTS:
                import requests
                # Reduced timeout to 30 seconds for faster failure and fallback
                response = requests.post(url, headers=headers, json=payload, timeout=30)
                response.raise_for_status()
                data = response.json()
            else:
                # return ToolResult.fail("No HTTP client available. Install httpx or requests.")
                return None
            
            # Success - return formatted results
            return _format_firecrawl_results(data, query, search_type)
            
        except Exception as e:
            last_error = e
            error_msg = str(e)
            
            # Don't retry on certain errors (authentication, bad request, etc.)
            if "401" in error_msg or "403" in error_msg or "400" in error_msg:
                # return ToolResult.fail(f"Firecrawl search failed: {e}")
                return None
            
            # If this is the last attempt, handle the error
            if attempt == max_retries:
                # Return None to allow fallback to Serper on timeout
                if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                    return None  # Allow fallback to Serper
                # return ToolResult.fail(f"Firecrawl search failed after {max_retries} attempts: {e}")
                return None
            
            # Wait before retrying with exponential backoff
            # Delay: 2s, 4s for attempts 1, 2
            delay = 2 ** attempt
            time.sleep(delay)
    
    # Should not reach here, but handle it anyway
    if last_error:
        error_msg = str(last_error)
        if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
            return None  # Allow fallback to Serper
        # return ToolResult.fail(f"Firecrawl search failed: {last_error}")
        return None
    
    return None


def _format_firecrawl_results(data: dict, query: str, search_type: str = "general") -> ToolResult:
    """Format Firecrawl API results."""
    if not data.get("success", False):
        error = data.get("error", "Unknown error")
        return ToolResult.fail(f"Firecrawl search failed: {error}")
    
    results = []
    
    # v1 API returns flat list in 'data' array
    for item in data.get("data", []):
        result = {
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "snippet": item.get("description", ""),
        }
        
        # If markdown content is available, use a truncated version as snippet
        markdown = item.get("markdown", "")
        if markdown and len(markdown) > len(result["snippet"]):
            # Use first 500 chars of markdown as extended snippet
            result["content"] = markdown[:1000] + ("..." if len(markdown) > 1000 else "")
        
        results.append(result)
    
    if not results:
        return ToolResult.ok(f"No results found for: {query}")
    
    return _format_results(results, query, search_type)


def _search_with_serper(
    query: str,
    num_results: int,
    search_type: str,
) -> Optional[ToolResult]:
    """Search using Serper.dev API (Google Search API).
    
    Serper provides fast Google search results via API.
    API docs: https://serper.dev/docs
    """
    api_key = os.environ.get("SERPER_API_KEY")
    if not api_key:
        return None
    
    # Determine endpoint based on search type
    endpoint_map = {
        "general": "https://google.serper.dev/search",
        "code": "https://google.serper.dev/search",
        "docs": "https://google.serper.dev/search",
        "news": "https://google.serper.dev/news",
        "images": "https://google.serper.dev/images",
    }
    url = endpoint_map.get(search_type, "https://google.serper.dev/search")
    
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
    }
    
    # Build payload
    payload = {
        "q": query,
        "num": num_results,
    }
    
    # Add search type modifiers for code/docs
    if search_type == "code":
        payload["q"] = f"{query} site:github.com OR site:stackoverflow.com"
    elif search_type == "docs":
        payload["q"] = f"{query} documentation OR docs OR tutorial"
    
    try:
        if HAS_HTTPX:
            with httpx.Client(timeout=30) as client:
                response = client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
        elif HAS_REQUESTS:
            import requests
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
        else:
            return ToolResult.fail("No HTTP client available. Install httpx or requests.")
        
        return _format_serper_results(data, query, search_type)
        
    except Exception as e:
        return ToolResult.fail(f"Serper search failed: {e}")


def _format_serper_results(data: dict, query: str, search_type: str) -> ToolResult:
    """Format Serper.dev API results."""
    results = []
    
    # Handle different response formats based on search type
    if search_type == "news":
        items = data.get("news", [])
    elif search_type == "images":
        items = data.get("images", [])
    else:
        # Regular search includes organic results, knowledge graph, answer box, etc.
        items = data.get("organic", [])
        
        # Include answer box if present (featured snippet)
        answer_box = data.get("answerBox")
        if answer_box:
            answer_result = {
                "title": answer_box.get("title", "Answer"),
                "url": answer_box.get("link", ""),
                "snippet": answer_box.get("snippet") or answer_box.get("answer", ""),
            }
            if answer_result["snippet"]:
                results.append(answer_result)
        
        # Include knowledge graph if present
        knowledge_graph = data.get("knowledgeGraph")
        if knowledge_graph:
            kg_description = knowledge_graph.get("description", "")
            if kg_description:
                kg_result = {
                    "title": knowledge_graph.get("title", "Knowledge Graph"),
                    "url": knowledge_graph.get("website", ""),
                    "snippet": kg_description,
                }
                results.append(kg_result)
    
    # Process main items
    for item in items:
        if search_type == "images":
            result = {
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": f"Image: {item.get('imageUrl', '')}",
            }
        elif search_type == "news":
            result = {
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", ""),
                "date": item.get("date", ""),
                "source": item.get("source", ""),
            }
        else:
            result = {
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", ""),
            }
            # Include position for reference
            if item.get("position"):
                result["position"] = item.get("position")
        
        results.append(result)
    
    if not results:
        return ToolResult.ok(f"No results found for: {query}")
    
    return _format_results(results, query, search_type)


def _format_results(results: list[dict], query: str, search_type: str = "general") -> ToolResult:
    """Format search results for output."""
    type_label = f" ({search_type})" if search_type != "general" else ""
    lines = [f"Search results{type_label} for: {query}\n"]
    
    for i, result in enumerate(results, 1):
        title = result.get("title", "No title")
        url = result.get("url", "")
        snippet = result.get("snippet", "No description")
        content = result.get("content", "")
        date = result.get("date", "")
        source = result.get("source", "")
        
        lines.append(f"{i}. {title}")
        if url:
            lines.append(f"   URL: {url}")
        if date or source:
            meta_parts = []
            if source:
                meta_parts.append(f"Source: {source}")
            if date:
                meta_parts.append(f"Date: {date}")
            lines.append(f"   {' | '.join(meta_parts)}")
        if snippet:
            lines.append(f"   {snippet}")
        if content:
            lines.append(f"\n   Content preview:\n   {content}\n")
        lines.append("")
    
    return ToolResult.ok("\n".join(lines))
