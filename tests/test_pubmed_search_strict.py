from __future__ import annotations

import json
import time
import urllib.error
import xml.etree.ElementTree as ET

import pytest

from hypoforge.tools import pubmed_search


class _Response:
    def __init__(self, body: str) -> None:
        self.body = body.encode("utf-8")
        self.offset = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = len(self.body) - self.offset
        chunk = self.body[self.offset:self.offset + size]
        self.offset += len(chunk)
        return chunk


@pytest.mark.asyncio
async def test_strict_pubmed_search_fetches_metadata_in_a_thread(monkeypatch) -> None:
    monkeypatch.setattr(pubmed_search, "_esearch", lambda query, limit: ["123"])
    monkeypatch.setattr(
        pubmed_search,
        "_efetch_batch",
        lambda pmids: [{"pmid": "123", "title": "A paper"}],
    )

    result = await pubmed_search.search_pubmed_strict("Hippo YAP", limit=3)

    assert result == [{"pmid": "123", "title": "A paper"}]


@pytest.mark.asyncio
async def test_strict_pubmed_search_propagates_network_failure(monkeypatch) -> None:
    def fail(query: str, limit: int):
        raise OSError("network unavailable")

    monkeypatch.setattr(pubmed_search, "_esearch", fail)

    with pytest.raises(OSError, match="network unavailable"):
        await pubmed_search.search_pubmed_strict("Hippo YAP")


@pytest.mark.asyncio
async def test_legacy_pubmed_tool_still_returns_empty_on_failure(monkeypatch) -> None:
    def fail(query: str, limit: int):
        raise OSError("network unavailable")

    monkeypatch.setattr(pubmed_search, "_esearch", fail)

    assert await pubmed_search.PubMedTool().search("Hippo YAP") == []


@pytest.mark.parametrize(
    "payload",
    [
        {"error": "Invalid request"},
        {"esearchresult": {"ERROR": "Invalid term", "count": "0", "idlist": []}},
        {
            "esearchresult": {
                "errorlist": {"phrasesnotfound": ["bad"]},
                "count": "0",
                "idlist": [],
            }
        },
    ],
)
def test_esearch_rejects_http_200_protocol_error_json(monkeypatch, payload) -> None:
    monkeypatch.setattr(
        pubmed_search.urllib.request,
        "urlopen",
        lambda url, timeout: _Response(json.dumps(payload)),
    )
    monkeypatch.setattr(pubmed_search, "_last_request_time", 0.0)

    with pytest.raises(pubmed_search.PubMedProtocolError):
        pubmed_search._esearch("Hippo", limit=5)


@pytest.mark.parametrize(
    "errorlist",
    [
        {},
        {"phrasesnotfound": [], "fieldsnotfound": []},
        {"phrasesnotfound": "", "nested": {"details": []}},
    ],
)
def test_esearch_accepts_present_but_empty_errorlist(
    monkeypatch, errorlist
) -> None:
    monkeypatch.setattr(
        pubmed_search,
        "_http_get_json",
        lambda url: {
            "esearchresult": {
                "count": "0",
                "idlist": [],
                "errorlist": errorlist,
            }
        },
    )

    assert pubmed_search._esearch("no match", limit=5) == []


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"esearchresult": []},
        {"esearchresult": {"count": "not-a-count", "idlist": []}},
        {"esearchresult": {"count": "0"}},
        {"esearchresult": {"count": "0", "idlist": ["123"]}},
        {"esearchresult": {"count": "1", "idlist": []}},
    ],
)
def test_esearch_rejects_missing_or_malformed_result_structure(
    monkeypatch, payload
) -> None:
    monkeypatch.setattr(
        pubmed_search.urllib.request,
        "urlopen",
        lambda url, timeout: _Response(json.dumps(payload)),
    )
    monkeypatch.setattr(pubmed_search, "_last_request_time", 0.0)

    with pytest.raises(pubmed_search.PubMedProtocolError):
        pubmed_search._esearch("Hippo", limit=5)


@pytest.mark.asyncio
async def test_strict_pubmed_search_accepts_real_zero_result(monkeypatch) -> None:
    calls = []

    def urlopen(url, timeout):
        calls.append(url)
        return _Response(json.dumps({
            "esearchresult": {"count": "0", "idlist": []},
        }))

    monkeypatch.setattr(pubmed_search.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(pubmed_search, "_last_request_time", 0.0)

    assert await pubmed_search.search_pubmed_strict("no match") == []
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_strict_pubmed_search_rejects_http_200_error_xml(monkeypatch) -> None:
    responses = iter([
        _Response(json.dumps({
            "esearchresult": {"count": "1", "idlist": ["123"]},
        })),
        _Response("<PubmedArticleSet><ERROR>Invalid PMID</ERROR></PubmedArticleSet>"),
    ])
    monkeypatch.setattr(
        pubmed_search.urllib.request,
        "urlopen",
        lambda url, timeout: next(responses),
    )
    monkeypatch.setattr(pubmed_search, "_last_request_time", 0.0)

    with pytest.raises(pubmed_search.PubMedProtocolError, match="Invalid PMID"):
        await pubmed_search.search_pubmed_strict("Hippo")


def test_efetch_keeps_xml_parse_errors(monkeypatch) -> None:
    monkeypatch.setattr(pubmed_search, "_http_get_xml_text", lambda url: "<broken")

    with pytest.raises(ET.ParseError):
        pubmed_search._efetch_batch(["123"])


def test_efetch_rejects_empty_article_set_for_requested_pmids(monkeypatch) -> None:
    monkeypatch.setattr(
        pubmed_search,
        "_http_get_xml_text",
        lambda url: "<PubmedArticleSet/>",
    )

    with pytest.raises(pubmed_search.PubMedProtocolError):
        pubmed_search._efetch_batch(["123"])


@pytest.mark.asyncio
async def test_legacy_pubmed_tool_swallows_protocol_errors(monkeypatch) -> None:
    monkeypatch.setattr(
        pubmed_search,
        "_http_get_json",
        lambda url: {"esearchresult": {"ERROR": "bad request"}},
    )
    assert await pubmed_search.PubMedTool().search("Hippo") == []

    monkeypatch.setattr(
        pubmed_search,
        "_http_get_xml_text",
        lambda url: "<PubmedArticleSet><ERROR>bad PMID</ERROR></PubmedArticleSet>",
    )
    assert await pubmed_search.PubMedTool().fetch("123") == {"pmid": "123"}


@pytest.mark.asyncio
async def test_strict_timeout_is_passed_to_sync_http_layer(monkeypatch) -> None:
    observed_timeouts = []

    def urlopen(url, timeout):
        observed_timeouts.append(timeout)
        return _Response(json.dumps({
            "esearchresult": {"count": "0", "idlist": []},
        }))

    monkeypatch.setattr(pubmed_search.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(pubmed_search, "_last_request_time", 0.0)

    assert await pubmed_search.search_pubmed_strict(
        "no match", timeout_seconds=0.25
    ) == []
    assert len(observed_timeouts) == 1
    assert 0 < observed_timeouts[0] <= 0.25


@pytest.mark.asyncio
async def test_strict_detects_deadline_exhausted_after_http_response(
    monkeypatch,
) -> None:
    def urlopen(url, timeout):
        time.sleep(0.03)
        return _Response(json.dumps({
            "esearchresult": {"count": "0", "idlist": []},
        }))

    monkeypatch.setattr(pubmed_search.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(pubmed_search, "_last_request_time", 0.0)

    with pytest.raises(TimeoutError):
        await pubmed_search.search_pubmed_strict(
            "no match", timeout_seconds=0.01
        )


def test_retry_backoff_does_not_outlive_absolute_deadline(monkeypatch) -> None:
    monkeypatch.setattr(pubmed_search, "_last_request_time", 0.0)
    monkeypatch.setattr(
        pubmed_search.urllib.request,
        "urlopen",
        lambda url, timeout: (_ for _ in ()).throw(urllib.error.URLError("offline")),
    )
    started = time.monotonic()

    with pytest.raises(TimeoutError):
        pubmed_search._http_get_json(
            "https://example.invalid", deadline=started + 0.05
        )

    assert time.monotonic() - started < 0.25


def test_rate_limit_wait_does_not_outlive_absolute_deadline(monkeypatch) -> None:
    monkeypatch.setattr(pubmed_search, "_last_request_time", time.monotonic())
    started = time.monotonic()

    with pytest.raises(TimeoutError):
        pubmed_search._rate_limit(deadline=started + 0.05)

    assert time.monotonic() - started < 0.25
