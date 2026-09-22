"""The review web app (CLAUDE.md §15).

FastAPI + Jinja2 + HTMX. No client framework, no build step, no bundler --
§4 rules those out and they would be dead weight here anyway: every
interaction is "change one thing, re-render that card".

This module holds the application factory and :class:`ProjectView`, the one
place that knows how to read a project off disk. The routes live next door:
:mod:`.routes_ui` renders, :mod:`.api` changes things.

The rule that shapes the whole design: **changing a pick rewrites
``selected/`` exactly.** Every mutating route re-runs the collect stage before
it returns, so the folder on disk always matches what the page shows. There is
no save button and no way to leave the two out of step.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ..config import Settings, load_settings
from ..logging_setup import get_logger
from ..project import ProjectPaths
from ..schemas import ProjectContext, Segment
from ..utils.files import pad_width
from ..utils.timefmt import timecode

__all__ = ["create_app", "ProjectView", "templates", "TEMPLATES", "STATIC"]

log = get_logger("web.app")

TEMPLATES = Path(__file__).parent / "templates"
STATIC = Path(__file__).parent / "static"

#: One environment for the whole app, so routes in either module render the
#: same templates with the same globals.
templates = Jinja2Templates(directory=str(TEMPLATES))
templates.env.globals["timecode"] = timecode


# ---------------------------------------------------------------------------
# Project access
# ---------------------------------------------------------------------------


class ProjectView:
    """Everything a page needs about one project, loaded from disk each time.

    Deliberately stateless. The worker writes these same files while the UI is
    open, so anything cached here would show the user a project that has since
    moved on.
    """

    def __init__(self, name: str, settings: Settings):
        self.name = name
        self.settings = settings
        self.paths = ProjectPaths(settings.project_dir(name))
        if not self.paths.root.exists():
            raise HTTPException(status_code=404, detail=f"no project named {name!r}")

    # -- reading -----------------------------------------------------------

    @property
    def segments(self) -> list[Segment]:
        out: list[Segment] = []
        for directory in self.paths.existing_segment_dirs():
            path = directory / "segment.json"
            if path.exists():
                out.append(Segment.model_validate_json(path.read_text(encoding="utf-8")))
        return out

    @property
    def width(self) -> int:
        return pad_width(max((s.index for s in self.segments), default=1))

    @property
    def context(self) -> ProjectContext:
        if self.paths.project_context.exists():
            return ProjectContext.model_validate_json(
                self.paths.project_context.read_text(encoding="utf-8")
            )
        return ProjectContext()

    def segment(self, index: int) -> Segment:
        for segment in self.segments:
            if segment.index == index:
                return segment
        raise HTTPException(status_code=404, detail=f"no segment {index}")

    def segment_dir(self, segment: Segment) -> Path:
        return self.paths.segment_dir(segment, width=self.width)

    def relative(self, raw: str) -> str:
        """A project-relative POSIX path, whatever form it arrived in.

        Picks on disk are project-relative, but a form post could carry either
        form, and comparing an absolute path to a relative one silently fails
        to find the pick -- which looks like the click doing nothing.
        """
        path = Path(raw)
        if path.is_absolute():
            try:
                path = path.relative_to(self.paths.root)
            except ValueError:
                return raw.replace("\\", "/")
        return path.as_posix()

    # -- assets ------------------------------------------------------------

    def images(self, segment: Segment) -> list[dict]:
        from ..pipeline.image_search import read_candidate_manifest

        manifest = read_candidate_manifest(
            self.paths.segment_image_manifest(segment, width=self.width), root=self.paths.root
        )
        records = [r for r in manifest["kept"] if r.rank > 0] or manifest["kept"]
        records.sort(key=lambda r: (r.rank or 999, -r.score))
        return [self._as_asset(segment, r) for r in records[:24]]

    def _as_asset(self, segment: Segment, record) -> dict:
        relative = self.relative(record.local_path)
        picked = next((p for p in segment.picks if self.relative(p.path) == relative), None)
        return {
            "path": relative,
            "url": f"/projects/{self.name}/file/{relative}",
            "rank": record.rank,
            "score": record.score,
            "breakdown": record.score_breakdown,
            "source_page": record.source_page,
            "domain": record.domain,
            "license": record.license,
            "creator": record.creator,
            "width": record.width,
            "height": record.height,
            "query": record.query,
            "query_kind": record.query_kind,
            "use": bool(picked and picked.use),
            "favorite": bool(picked and picked.favorite),
            "rejected": bool(picked and picked.rejected),
            "note": picked.note if picked else "",
        }

    def videos(self, segment: Segment) -> list[dict]:
        path = self.paths.segment_youtube_results(segment, width=self.width)
        if not path.exists():
            return []
        out = []
        for item in json.loads(path.read_text(encoding="utf-8")):
            clip = item.get("downloaded_clip_path") or ""
            best = (item.get("timestamp_candidates") or [None])[0]
            out.append(
                {
                    **item,
                    "clip_url": f"/projects/{self.name}/file/{clip}" if clip else "",
                    "best": best,
                    "best_timecode": timecode(best["start_s"]) if best else "",
                }
            )
        return out

    def selected_files(self) -> list[str]:
        if not self.paths.selected_dir.exists():
            return []
        return sorted(p.name for p in self.paths.selected_dir.iterdir() if p.is_file())

    def gaps(self) -> list[Segment]:
        from ..pipeline.collect import plan_selected

        _, gaps = plan_selected(self.segments, self.paths, self.settings, width=self.width)
        return gaps

    # -- writing -----------------------------------------------------------

    def save(self, segment: Segment) -> None:
        from ..utils.files import atomic_write_text

        atomic_write_text(
            self.segment_dir(segment) / "segment.json", segment.model_dump_json(indent=2)
        )

    def recollect(self):
        """Rewrite ``selected/`` to match the current picks (§15)."""
        from ..pipeline.collect import collect

        return collect(self.segments, self.paths, self.settings, width=self.width)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    settings.ensure_dirs()

    app = FastAPI(title="VisualResearcher", docs_url=None, redoc_url=None)
    app.state.settings = settings

    if STATIC.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

    # Imported here rather than at module scope: both route modules import
    # ProjectView from this one, and importing them at the top would be a cycle.
    from . import api, routes_ui

    app.include_router(routes_ui.router)
    app.include_router(api.router)
    return app
