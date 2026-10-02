import numpy as np
import pandas as pd

from models.schemas import ResearchQuestion
from services.literature import (
    build_queries,
    parse_atom,
    rank_papers,
    search_literature,
    search_related,
    snippet,
)

FIXTURE_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2401.0001v1</id>
    <title>  Predicting student
    performance with machine learning </title>
    <summary>Study on final score prediction using attendance rate and study habits across 200 students.</summary>
    <published>2024-01-15T00:00:00Z</published>
    <link href="http://arxiv.org/abs/2401.0001v1" rel="alternate"/>
    <author><name>Alice Wang</name></author>
    <author><name>Bob Li</name></author>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2402.0002v1</id>
    <title>Unrelated quantum gravity survey</title>
    <summary>Cosmology and gravitational waves.</summary>
    <published>2023-02-01T00:00:00Z</published>
    <link href="http://arxiv.org/abs/2402.0002v1" rel="alternate"/>
    <author><name>Carol Zhang</name></author>
  </entry>
</feed>
"""


class FakeResponse:
    def __init__(self, text: str, status_code: int = 200):
        self.text = text
        self.status_code = status_code


def make_profile():
    from services.profiler import profile_dataset

    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "student_id": [f"S{i:04d}" for i in range(60)],
            "attendance_rate": rng.uniform(50, 100, 60).round(1),
            "final_score": rng.normal(70, 10, 60).round(1),
        }
    )
    return profile_dataset(df)


def test_parse_atom_cleans_fields():
    papers = parse_atom(FIXTURE_ATOM)
    assert len(papers) == 2
    first = papers[0]
    assert first.title == "Predicting student performance with machine learning"
    assert first.authors == ["Alice Wang", "Bob Li"]
    assert first.year == "2024"
    assert first.url == "http://arxiv.org/abs/2401.0001v1"
    assert "final score prediction" in first.summary


def test_parse_atom_bad_xml_returns_empty():
    assert parse_atom("not xml at all <") == []


def test_rank_prefers_term_overlap():
    papers = parse_atom(FIXTURE_ATOM)
    queries = ['all:"final score" AND all:"attendance rate"']
    ranked = rank_papers(papers, queries, top=2)
    assert ranked[0].title.startswith("Predicting student")
    assert len(ranked) == 1  # 零词面命中的论文被过滤


def test_build_queries_from_targets_and_questions():
    profile = make_profile()
    question = ResearchQuestion(
        id="RQ1", question="q", variables=["attendance_rate", "final_score"]
    )
    queries = build_queries(profile, [question])
    assert queries
    assert '"final score"' in queries[0]
    assert "attendance" in queries[0]


def test_search_literature_with_injected_fetch():
    calls = []

    def fake_fetch(url, params=None, timeout=None):
        calls.append(params["search_query"])
        return FakeResponse(FIXTURE_ATOM)

    papers = search_literature(['all:"final score"'], fetch=fake_fetch)
    assert calls == ['all:"final score"']
    assert len(papers) == 2


def test_search_literature_network_error_is_silent():
    def broken_fetch(url, params=None, timeout=None):
        import httpx

        raise httpx.ConnectError("boom")

    assert search_literature(['all:"x"'], fetch=broken_fetch) == []


def test_search_literature_non_200_ignored():
    assert search_literature(['all:"x"'], fetch=lambda *a, **k: FakeResponse("", 503)) == []


def test_search_related_caps_top_n():
    profile = make_profile()

    def fake_fetch(url, params=None, timeout=None):
        return FakeResponse(FIXTURE_ATOM)

    papers = search_related(profile, [], fetch=fake_fetch)
    assert 0 < len(papers) <= 3


def test_snippet_truncates():
    assert snippet("x" * 500).endswith("…")
    assert len(snippet("x" * 500)) == 241
    assert snippet("short") == "short"
