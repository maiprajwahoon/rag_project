"""Retrieval evaluation over the golden set."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel

from rag_project.evaluation.golden import GoldenItem
from rag_project.evaluation.metrics import (
    first_hit_rank,
    page_recall,
    reciprocal_rank,
)
from rag_project.ingestion.models import Chunk
from rag_project.retrieval.retriever import Mode

MODES: tuple[Mode, ...] = ("vector", "hybrid", "hybrid_rerank")


class Retriever(Protocol):
    """Anything that can retrieve chunks in a given mode."""

    def retrieve(self, query: str, mode: Mode) -> list[tuple[Chunk, float]]:
        """Return (chunk, score) pairs, best first."""
        ...


class ModeResult(BaseModel):
    """Aggregate retrieval metrics for one mode."""

    mode: str
    n: int
    hit_rate: float
    mean_recall: float
    mrr: float
    ranks: dict[str, int | None]


class EvalReport(BaseModel):
    """Full retrieval evaluation report."""

    created: str
    golden_sha: str
    k: int
    modes: list[ModeResult]
    top_vector_scores_answerable: dict[str, float]
    top_vector_scores_unanswerable: dict[str, float]


def _pages(hits: Sequence[tuple[Chunk, float]], k: int) -> list[str]:
    return [chunk.source_path for chunk, _ in hits[:k]]


def evaluate_retrieval(
    retriever: Retriever, items: Sequence[GoldenItem], k: int, golden_sha: str
) -> EvalReport:
    """Score every retrieval mode on the answerable questions."""
    answerable = [i for i in items if i.answerable]
    if not answerable:
        raise ValueError("golden set has no answerable questions")
    results: list[ModeResult] = []
    for mode in MODES:
        ranks: dict[str, int | None] = {}
        recalls: list[float] = []
        rr: list[float] = []
        for item in answerable:
            pages = _pages(retriever.retrieve(item.question, mode), k)
            ranks[item.id] = first_hit_rank(pages, item.pages)
            recalls.append(page_recall(pages, item.pages))
            rr.append(reciprocal_rank(pages, item.pages))
        n = len(answerable)
        results.append(
            ModeResult(
                mode=mode,
                n=n,
                hit_rate=sum(r is not None for r in ranks.values()) / n,
                mean_recall=sum(recalls) / n,
                mrr=sum(rr) / n,
                ranks=ranks,
            )
        )
    top_ans: dict[str, float] = {}
    top_unans: dict[str, float] = {}
    for item in items:
        hits = retriever.retrieve(item.question, "vector")
        score = hits[0][1] if hits else 0.0
        (top_ans if item.answerable else top_unans)[item.id] = score
    return EvalReport(
        created=datetime.now(UTC).isoformat(timespec="seconds"),
        golden_sha=golden_sha,
        k=k,
        modes=results,
        top_vector_scores_answerable=top_ans,
        top_vector_scores_unanswerable=top_unans,
    )


def format_report(report: EvalReport) -> str:
    """Render a plain-text summary table and diagnostics."""
    lines = [
        f"golden={report.golden_sha} k={report.k} created={report.created}",
        "",
        f"{'mode':<15}{'hit@k':>8}{'recall':>8}{'MRR':>8}",
    ]
    for m in report.modes:
        lines.append(
            f"{m.mode:<15}{m.hit_rate:>8.3f}{m.mean_recall:>8.3f}{m.mrr:>8.3f}"
        )
    lines.append("")
    for m in report.modes:
        misses = [qid for qid, rank in m.ranks.items() if rank is None]
        lines.append(f"{m.mode} misses ({len(misses)}): {', '.join(misses) or 'none'}")
    ans = sorted(report.top_vector_scores_answerable.values())
    unans = sorted(report.top_vector_scores_unanswerable.values(), reverse=True)
    lines.append("")
    lines.append(
        "top-1 cosine, lowest answerable: " + ", ".join(f"{s:.3f}" for s in ans[:5])
    )
    lines.append(
        "top-1 cosine, highest unanswerable: "
        + ", ".join(f"{s:.3f}" for s in unans[:5])
    )
    if ans and unans:
        lines.append(
            f"gap (min answerable - max unanswerable): {ans[0] - unans[0]:.3f}"
        )
    return "\n".join(lines)


def save_results(report: EvalReport, directory: Path) -> Path:
    """Write the report as JSON and return the file path."""
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.fromisoformat(report.created).strftime("%Y%m%d_%H%M%S")
    path = directory / f"retrieval_{stamp}.json"
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return path
