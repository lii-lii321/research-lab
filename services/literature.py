# -*- coding: utf-8 -*-
"""文献检索工具：arXiv 关键词检索 + 词面重排（无嵌入、无 Key，网络不可用时优雅降级）。"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import httpx

from models.schemas import PaperRef, ProfileReport, ResearchQuestion

ARXIV_API = "https://export.arxiv.org/api/query"
ATOM_NS = "{http://www.w3.org/2005/Atom}"
TIMEOUT = 10.0
MAX_QUERIES = 2
MAX_PER_QUERY = 4
TOP_N = 3
SUMMARY_SNIPPET = 240

_SUMMARY_CLEAN = re.compile(r"\s+")


def build_queries(
    profile: ProfileReport, questions: list[ResearchQuestion] | None = None
) -> list[str]:
    """从目标候选与研究问题构造 arXiv 检索词（字段名下划线转空格）。

    顺序即精度递降：字段组合 AND → 目标单词（召回兜底）→ 首个 RQ 的变量组合。
    """

    def term(name: str) -> str:
        return name.replace("_", " ").strip().lower()

    def to_query(names: list[str]) -> str:
        return " AND ".join(f'all:"{term(n)}"' for n in names if term(n))

    queries: list[str] = []
    if profile.target_candidates:
        target = profile.target_candidates[0].column
        numerics = [
            c.name for c in profile.columns if c.type == "numeric" and c.name != target
        ][:1]
        q = to_query([target] + numerics)
        if q:
            queries.append(q)
        single = term(target)
        if single:
            queries.append(f'all:"{single}"')
    if questions and questions[0].variables:
        q = to_query(questions[0].variables[:2])
        if q and q not in queries:
            queries.append(q)
    return queries[:MAX_QUERIES + 1]


def parse_atom(xml_text: str) -> list[PaperRef]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    papers: list[PaperRef] = []
    for entry in root.findall(f"{ATOM_NS}entry"):
        title_el = entry.find(f"{ATOM_NS}title")
        summary_el = entry.find(f"{ATOM_NS}summary")
        published_el = entry.find(f"{ATOM_NS}published")
        url = ""
        for link in entry.findall(f"{ATOM_NS}link"):
            href = link.get("href", "")
            if href:
                url = href
                if link.get("rel") == "alternate":
                    break
        papers.append(
            PaperRef(
                title=_SUMMARY_CLEAN.sub(" ", title_el.text or "").strip() if title_el is not None else "",
                authors=[
                    _SUMMARY_CLEAN.sub(" ", (a.text or "")).strip()
                    for a in entry.findall(f"{ATOM_NS}author/{ATOM_NS}name")
                    if a.text
                ],
                year=(published_el.text or "")[:4] if published_el is not None else "",
                summary=_SUMMARY_CLEAN.sub(" ", summary_el.text or "").strip()
                if summary_el is not None
                else "",
                url=url,
            )
        )
    return papers


def _score(paper: PaperRef, terms: set[str]) -> int:
    text = f"{paper.title} {paper.summary}".lower()
    return sum(text.count(t) for t in terms)


def rank_papers(papers: list[PaperRef], queries: list[str], top: int = TOP_N) -> list[PaperRef]:
    terms: set[str] = set()
    for q in queries:
        terms.update(t.strip('"').lower() for t in re.findall(r'"([^"]+)"', q))
    if not terms:
        return papers[:top]
    scored = sorted(papers, key=lambda p: _score(p, terms), reverse=True)
    hits = [p for p in scored if _score(p, terms) > 0]
    return (hits or scored)[:top]


def search_literature(
    queries: list[str],
    max_per_query: int = MAX_PER_QUERY,
    timeout: float = TIMEOUT,
    fetch=None,
) -> list[PaperRef]:
    fetch = fetch or httpx.get
    seen: set[str] = set()
    papers: list[PaperRef] = []
    for q in queries:
        try:
            resp = fetch(
                ARXIV_API,
                params={
                    "search_query": q,
                    "start": 0,
                    "max_results": max_per_query,
                    "sortBy": "relevance",
                },
                timeout=timeout,
            )
        except (httpx.HTTPError, TimeoutError, OSError):
            continue
        if getattr(resp, "status_code", 0) != 200:
            continue
        found = [p for p in parse_atom(resp.text) if p.url and p.url not in seen]
        if not found:
            continue
        seen.update(p.url for p in found)
        papers.extend(found)
        if len(papers) >= max_per_query:
            break
    return papers


def search_related(
    profile: ProfileReport,
    questions: list[ResearchQuestion] | None = None,
    top: int = TOP_N,
    fetch=None,
) -> list[PaperRef]:
    queries = build_queries(profile, questions)
    if not queries:
        return []
    papers = search_literature(queries, fetch=fetch)
    return rank_papers(papers, queries, top=top)


def snippet(text: str, limit: int = SUMMARY_SNIPPET) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"
