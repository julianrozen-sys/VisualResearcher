"""The read-only HTML routes (CLAUDE.md §15).

Everything here renders a page or serves a file; nothing changes state. The
routes that *do* change something live in :mod:`.api`, which keeps the
"changing a pick rewrites ``selected/``" rule in one place rather than spread
across every handler.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse

from ..logging_setup import get_logger
from .app import ProjectView, templates

__all__ = ["router", "render_segment_card"]

log = get_logger("web.ui")

router = APIRouter()


def _settings(request: Request):
    return request.app.state.settings


def _view(request: Request, name: str) -> ProjectView:
    return ProjectView(name, _settings(request))


def render_segment_card(request: Request, name: str, index: int) -> HTMLResponse:
    """The HTMX partial. Every mutating action re-renders exactly this.

    Shared with :mod:`.api` so an action and a plain reload cannot drift into
    rendering different things.
    """
    project = _view(request, name)
    segment = project.segment(index)
    return templates.TemplateResponse(
        request,
        "_segment_card.html",
        {
            "name": name,
            "segment": segment,
            "images": project.images(segment),
            "videos": project.videos(segment),
            "context": project.context,
            "expanded": True,
        },
    )


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------


@router.get("/", response_class=HTMLResponse)
def jobs_page(request: Request):
    """§15: status, stage, timing, disk used."""
    from ..db import init_db
    from ..jobs.queue import list_jobs
    from ..utils.files import dir_size_bytes

    settings = _settings(request)
    init_db(settings.db_path)
    jobs = list_jobs(settings.db_path, limit=100)

    rows = []
    for job in jobs:
        root = settings.project_dir(job.project)
        rows.append(
            {
                "job": job,
                "exists": root.exists(),
                "size_mb": dir_size_bytes(root) / 1_048_576 if root.exists() else 0,
                "duration": job.duration_s,
            }
        )
    projects = (
        sorted(
            (p.name for p in settings.projects_dir.iterdir() if p.is_dir()),
            reverse=True,
        )
        if settings.projects_dir.exists()
        else []
    )
    return templates.TemplateResponse(request, "jobs.html", {"rows": rows, "projects": projects})


# ---------------------------------------------------------------------------
# Project
# ---------------------------------------------------------------------------


@router.get("/projects/{name}", response_class=HTMLResponse)
def project_page(request: Request, name: str, band: str = "all"):
    from ..pipeline.collect import confidence_band

    project = _view(request, name)
    segments = project.segments
    gaps = {s.index for s in project.gaps()}

    cards = []
    for segment in segments:
        best = max((p.confidence for p in segment.picks if p.use), default=0.0)
        cards.append(
            {
                "segment": segment,
                "band": confidence_band(best, project.settings),
                "confidence": best,
                "is_gap": segment.index in gaps,
                "picks": [p for p in segment.picks if p.use],
            }
        )
    if band != "all":
        cards = [c for c in cards if c["band"] == band]

    return templates.TemplateResponse(
        request,
        "project.html",
        {
            "name": name,
            "context": project.context,
            "cards": cards,
            "gap_count": len(gaps),
            "gap_indexes": sorted(gaps),
            "selected_count": len(project.selected_files()),
            "band": band,
            "total": len(segments),
        },
    )


@router.get("/projects/{name}/segments/{index}", response_class=HTMLResponse)
def segment_card(request: Request, name: str, index: int):
    return render_segment_card(request, name, index)


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


@router.get("/projects/{name}/file/{path:path}")
def project_file(request: Request, name: str, path: str):
    """Serve a file from inside the project, and only from inside it.

    The traversal guard is the same rule as §11's downloader: resolve, then
    check the result is genuinely under the project root. Checking the string
    before resolving is what lets ``..`` through.
    """
    project = _view(request, name)
    target = (project.paths.root / path).resolve()
    root = project.paths.root.resolve()
    if root != target and root not in target.parents:
        raise HTTPException(status_code=403, detail="path escapes the project")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="no such file")
    return FileResponse(target)


@router.get("/healthz")
def healthz():
    return {"ok": True}
