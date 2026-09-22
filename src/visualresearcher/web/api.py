"""The mutating routes (CLAUDE.md §15).

Every route here ends the same way: **re-run collect, then re-render the
card.** That is the whole contract. There is no save button, no dirty state
and no way for the page and ``selected/`` to disagree, because the folder is
rebuilt as part of handling the click rather than later.

Keeping them together makes that visible. Spread across a dozen handlers it
would be one `recollect()` call away from being quietly forgotten in one of
them, and the symptom — a thumbnail marked USED with no file on disk — looks
like a UI bug rather than a missing line.
"""

from __future__ import annotations

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from ..logging_setup import get_logger
from ..schemas import Segment
from .app import ProjectView
from .routes_ui import render_segment_card

__all__ = ["router"]

log = get_logger("web.api")

router = APIRouter()

#: Extensions treated as a clip when a pick is created on the fly.
_VIDEO_SUFFIXES = (".mp4", ".mkv", ".webm", ".mov")


def _view(request: Request, name: str) -> ProjectView:
    return ProjectView(name, request.app.state.settings)


def _find_pick(project: ProjectView, segment: Segment, path: str):
    wanted = project.relative(path)
    return next((p for p in segment.picks if project.relative(p.path) == wanted), None)


def _apply_to_pick(project: ProjectView, segment: Segment, path: str, **changes) -> Segment:
    """Change one pick, creating it if the reviewer is promoting a candidate."""
    from ..schemas import Pick
    from ..utils.files import slugify

    relative = project.relative(path)
    pick = _find_pick(project, segment, path)
    if pick is None:
        pick = Pick(
            kind="clip" if relative.endswith(_VIDEO_SUFFIXES) else "image",
            path=relative,
            rank=len(segment.picks) + 1,
            slug=slugify(
                segment.topic or segment.narration,
                max_len=project.settings.output.slug_max_len,
            ),
            confidence=segment.confidence,
        )
        segment.picks.append(pick)

    for key, value in changes.items():
        setattr(pick, key, value)
    if pick.rejected:
        pick.use = False
    return segment


# ---------------------------------------------------------------------------
# Picks -- these rewrite selected/ (§15)
# ---------------------------------------------------------------------------


@router.post("/projects/{name}/segments/{index}/use", response_class=HTMLResponse)
def toggle_use(request: Request, name: str, index: int, path: str = Form(...)):
    """USE THIS. Rewrites ``selected/`` immediately.

    ``user_set`` marks this as a human decision, which §14's confidence bands
    do not override: the bands govern what the pipeline copies *on its own*.
    """
    project = _view(request, name)
    segment = project.segment(index)
    current = _find_pick(project, segment, path)

    _apply_to_pick(
        project,
        segment,
        path,
        use=not (current and current.use),
        rejected=False,
        user_set=True,
    )
    project.save(segment)
    result = project.recollect()
    log.info("%s segment %03d: selected/ now holds %d file(s)", name, index, result.total)
    return render_segment_card(request, name, index)


@router.post("/projects/{name}/segments/{index}/reject", response_class=HTMLResponse)
def reject(request: Request, name: str, index: int, path: str = Form(...)):
    project = _view(request, name)
    segment = project.segment(index)
    _apply_to_pick(project, segment, path, rejected=True, use=False, user_set=True)
    project.save(segment)
    project.recollect()
    return render_segment_card(request, name, index)


@router.post("/projects/{name}/segments/{index}/favorite", response_class=HTMLResponse)
def toggle_favorite(request: Request, name: str, index: int, path: str = Form(...)):
    """A favourite is a bookmark, not a pick, so it does not touch selected/."""
    project = _view(request, name)
    segment = project.segment(index)
    current = _find_pick(project, segment, path)
    _apply_to_pick(project, segment, path, favorite=not (current and current.favorite))
    project.save(segment)
    return render_segment_card(request, name, index)


# ---------------------------------------------------------------------------
# Segment metadata
# ---------------------------------------------------------------------------


@router.post("/projects/{name}/segments/{index}/note", response_class=HTMLResponse)
def add_note(request: Request, name: str, index: int, note: str = Form("")):
    project = _view(request, name)
    segment = project.segment(index)
    if note.strip():
        segment.notes.append(note.strip())
    project.save(segment)
    return render_segment_card(request, name, index)


@router.post("/projects/{name}/segments/{index}/reviewed", response_class=HTMLResponse)
def mark_reviewed(request: Request, name: str, index: int):
    project = _view(request, name)
    segment = project.segment(index)
    segment.status = "ok"
    if "reviewed" not in segment.notes:
        segment.notes.append("reviewed")
    project.save(segment)
    return render_segment_card(request, name, index)


@router.post("/projects/{name}/segments/{index}/entity", response_class=HTMLResponse)
def edit_entities(request: Request, name: str, index: int, entities: str = Form("")):
    """§15: entities are visible and reversible.

    Reversible means exactly that -- this replaces the list, so typing the old
    value back restores it. The transcript is never touched (§10.2).
    """
    project = _view(request, name)
    segment = project.segment(index)
    segment.entities = [e.strip() for e in entities.split(",") if e.strip()]
    project.save(segment)
    return render_segment_card(request, name, index)


@router.post("/projects/{name}/segments/{index}/query", response_class=HTMLResponse)
def edit_query(
    request: Request,
    name: str,
    index: int,
    kind: str = Form("fallback"),
    text: str = Form(...),
):
    """Add a query. Run it with ``visualresearch rerun NAME --segment N``."""
    from ..schemas import Query

    project = _view(request, name)
    segment = project.segment(index)
    if text.strip():
        segment.queries.append(Query(kind=kind, text=text.strip()))
        segment.notes.append(f"query added in review: {text.strip()!r}")
    project.save(segment)
    return render_segment_card(request, name, index)


# ---------------------------------------------------------------------------
# Whole project
# ---------------------------------------------------------------------------


@router.post("/projects/{name}/collect")
def recollect(request: Request, name: str):
    _view(request, name).recollect()
    return RedirectResponse(f"/projects/{name}", status_code=303)
