"""Command line interface (CLAUDE.md §17)."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import typer
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import __version__
from .config import Settings, load_env_defaults, load_local_env, load_settings, offline_mode
from .logging_setup import add_job_log, console, get_logger, setup_logging

#: Shown under the all-candidates table.
SELECTED_SUFFIX_HINT = "Files ending _SELECTED are the current pick in selected/."

app = typer.Typer(
    name="visualresearch",
    help="Research and download chronologically-ordered visual assets for a narration.",
    no_args_is_help=True,
    add_completion=False,
)
cache_app = typer.Typer(help="Cache maintenance.", no_args_is_help=True)
app.add_typer(cache_app, name="cache")

log = get_logger("cli")

OK = "[green]OK[/green]"
WARN = "[yellow]WARN[/yellow]"
FAIL = "[red]FAIL[/red]"


def _settings() -> Settings:
    settings = load_settings()
    settings.ensure_dirs()
    return settings


def _db(settings: Settings) -> Path:
    from .db import init_db

    init_db(settings.db_path)
    return settings.db_path


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug-level logging."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Errors only."),
) -> None:
    setup_logging("DEBUG" if verbose else "INFO", quiet=quiet)
    # Do this before anything can import torch or faster-whisper: once those
    # read HF_HOME/TORCH_HOME, setting them is too late (§3).
    applied = load_env_defaults()
    if applied:
        log.debug("cache locations from .env: %s", ", ".join(sorted(applied)))
    local = load_local_env()
    if local:
        # Names only. The values are secrets and personal data by design.
        log.debug("settings from .env.local: %s", ", ".join(sorted(local)))


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------


@app.command()
def doctor() -> None:
    """Report what is installed, what is missing, and what still works.

    Written to be run first when anything is behaving oddly. It never fails the
    process for a missing optional dependency -- it tells you what you lose.
    """
    from .providers import list_providers  # noqa: F401  (registers providers)
    from .providers.registry import get_provider
    from .utils.disk import usage_for

    mode = (
        "[yellow]VR_OFFLINE=1 - fakes only, no network[/yellow]"
        if offline_mode()
        else "online mode"
    )
    console.print(
        Panel.fit(
            f"[bold]VisualResearcher {__version__}[/bold]\n{mode}",
            border_style="cyan",
        )
    )

    problems: list[str] = []
    warnings: list[str] = []

    # -- disk (§3): C: and D: are called out by name -----------------------
    disk = Table(title="Disk", title_justify="left", header_style="bold")
    for col in ("Drive", "Free", "Total", "% free", "Status"):
        disk.add_column(col)

    settings = load_settings()
    checked: set[str] = set()
    targets: list[tuple[str, Path]] = []
    for letter in ("C:", "D:"):
        probe = Path(f"{letter}\\") if os.name == "nt" else Path("/")
        if probe.exists():
            targets.append((letter, probe))
    targets.append(("output", settings.projects_dir))

    for label, path in targets:
        try:
            usage = usage_for(path)
        except Exception as exc:  # noqa: BLE001
            disk.add_row(label, "-", "-", "-", f"{FAIL} {exc}")
            continue
        key = usage.drive
        if label == "output":
            label = f"output ({usage.drive})"
        elif key in checked:
            continue
        checked.add(key)

        if usage.free_gb < settings.output.min_free_gb:
            status = f"{FAIL} below min_free_gb={settings.output.min_free_gb:g}"
            if usage.drive.upper().startswith("D"):
                problems.append(
                    f"drive {usage.drive} has {usage.free_gb:.2f} GB free; bulk stages will abort"
                )
            else:
                warnings.append(
                    f"drive {usage.drive} has {usage.free_gb:.2f} GB free "
                    f"(nothing should be written there, but keep an eye on it)"
                )
        else:
            status = OK
        disk.add_row(
            label,
            f"{usage.free_gb:.2f} GB",
            f"{usage.total_gb:.0f} GB",
            f"{usage.percent_free:.1f}%",
            status,
        )
    console.print(disk)

    # -- python / paths ----------------------------------------------------
    env = Table(title="Environment", title_justify="left", header_style="bold")
    for col in ("Item", "Value", "Status"):
        env.add_column(col)

    py_ok = sys.version_info[:2] == (3, 12)
    env.add_row(
        "python",
        f"{sys.version.split()[0]} ({sys.executable})",
        OK if py_ok else f"{WARN} expected 3.12",
    )
    in_venv = ".venv" in sys.executable.replace("/", "\\")
    env.add_row(
        "venv",
        "in project .venv" if in_venv else "NOT the project venv",
        OK if in_venv else f"{WARN} use `uv run`",
    )
    if not in_venv:
        warnings.append("not running inside the project .venv; use `uv run visualresearch ...`")

    for var in ("HF_HOME", "TORCH_HOME", "UV_CACHE_DIR", "PIP_CACHE_DIR"):
        value = os.environ.get(var, "")
        if not value:
            status = f"{WARN} unset - caches may land on C:"
            warnings.append(f"{var} is unset; model/package caches may be written to C:")
        elif value.upper().startswith("C:"):
            status = f"{FAIL} points at C:"
            problems.append(f"{var}={value} points at C:, which §3 forbids")
        else:
            status = OK
        env.add_row(var, value or "(unset)", status)

    on_c = []
    for label, path in (
        ("projects", settings.projects_dir),
        ("data", settings.data_dir),
        ("cache", settings.cache_dir),
        ("tmp", settings.tmp_dir),
    ):
        drive = str(path.resolve())[:2].upper()
        bad = drive == "C:"
        if bad:
            on_c.append(f"{label}={path}")
        env.add_row(f"path: {label}", str(path), FAIL if bad else OK)
    if on_c:
        problems.append("these paths are on C:, which §3 forbids: " + ", ".join(on_c))
    console.print(env)

    # -- external tools ----------------------------------------------------
    tools = Table(title="External tools", title_justify="left", header_style="bold")
    for col in ("Tool", "Path", "Needed for", "Status"):
        tools.add_column(col)
    for name, needed, required in (
        ("ffmpeg", "audio decode, clip trimming", True),
        ("ffprobe", "media probing", True),
        ("yt-dlp", "YouTube search and sectioned clip download (P5)", False),
        ("git", "version control", False),
    ):
        found = shutil.which(name)
        if found:
            tools.add_row(name, found, needed, OK)
        elif required:
            tools.add_row(name, "-", needed, FAIL)
            problems.append(f"{name} is not on PATH but is required for {needed}")
        else:
            tools.add_row(name, "-", needed, f"{WARN} missing")
            warnings.append(f"{name} is not on PATH; {needed} will not work")
    console.print(tools)

    # -- providers ---------------------------------------------------------
    provs = Table(title="Providers", title_justify="left", header_style="bold")
    for col in ("Kind", "Name", "Offline-safe", "Status", "Detail"):
        provs.add_column(col)
    for kind, name in list_providers():
        try:
            provider = get_provider(kind, name)
            avail = provider.availability()
        except Exception as exc:  # noqa: BLE001
            provs.add_row(kind, name, "-", FAIL, str(exc))
            continue
        status = OK if avail.ok else f"{WARN} unavailable"
        detail = avail.detail
        if avail.missing:
            detail += "  -> " + "; ".join(avail.missing)
        provs.add_row(
            kind,
            name + (" (fake)" if provider.is_fake else ""),
            "yes" if avail.offline_safe else "no",
            status,
            detail,
        )
        if not avail.ok and not provider.is_fake:
            warnings.append(f"{kind} provider {name!r}: {avail.detail}")
    console.print(provs)

    # -- database ----------------------------------------------------------
    try:
        db_path = _db(settings)
        from .jobs.queue import list_jobs

        jobs = list_jobs(db_path, limit=1000)
        console.print(f"Database  {OK}  {db_path}  ({len(jobs)} jobs)")
    except Exception as exc:  # noqa: BLE001
        console.print(f"Database  {FAIL}  {exc}")
        problems.append(f"database unusable: {exc}")

    # -- verdict -----------------------------------------------------------
    console.print()
    if problems:
        console.print(
            Panel(
                "\n".join(f"- {p}" for p in problems),
                title="[red]Problems[/red]",
                border_style="red",
            )
        )
    if warnings:
        console.print(
            Panel(
                "\n".join(f"- {w}" for w in warnings),
                title="[yellow]Warnings[/yellow]",
                border_style="yellow",
            )
        )

    from .jobs.worker import _PLANNED, IMPLEMENTED_STAGES

    works = [
        f"all {len(IMPLEMENTED_STAGES)} pipeline stages: "
        + ", ".join(str(s) for s in IMPLEMENTED_STAGES),
        "the whole pipeline offline with VR_OFFLINE=1 (no keys, no network)",
        "`serve` for the review UI; `collect` to rebuild selected/",
    ]
    if _PLANNED:
        warnings.append("not built yet: " + ", ".join(f"{s} ({p})" for s, p in _PLANNED.items()))
    if not shutil.which("yt-dlp"):
        works.append("clips are unavailable without yt-dlp, but everything else runs")
    console.print(
        Panel(
            "\n".join(f"- {w}" for w in works),
            title="[green]What works[/green]",
            border_style="green",
        )
    )
    if problems:
        console.print("[red]doctor: problems found (listed above).[/red]")
        raise typer.Exit(code=1)
    console.print("[green]doctor: no blocking problems.[/green]")


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


@app.command()
def run(
    narration: Path = typer.Argument(..., help="Cleaned narration audio file."),
    project: str | None = typer.Option(None, "--project", help="Project name (default: filename)."),
    no_clips: bool = typer.Option(False, "--no-clips", help="Skip video clip download."),
    force: bool = typer.Option(False, "--force", help="Redo stages that are already checkpointed."),
    batch: bool = typer.Option(
        False,
        "--batch",
        help="Stage-by-stage across all segments, instead of delivering each as it finishes.",
    ),
) -> None:
    """Run the pipeline on a narration file."""
    from .jobs.queue import enqueue
    from .jobs.worker import RunContext, run_pipeline, run_segment_pipelined
    from .pipeline.ingest import derive_project_name

    settings = _settings()
    db_path = _db(settings)

    narration = narration.expanduser().resolve()
    if not narration.exists():
        console.print(f"[red]No such file:[/red] {narration}")
        raise typer.Exit(code=2)

    name = project or derive_project_name(
        narration, settings.projects_dir, slug_max_len=settings.output.slug_max_len
    )
    project_root = settings.project_dir(name)

    job = enqueue(db_path, project=name, source_path=narration, no_clips=no_clips, force=force)
    project_root.mkdir(parents=True, exist_ok=True)
    add_job_log(project_root / "job.log")

    console.print(
        Panel.fit(
            f"project [bold]{name}[/bold]\njob     {job.id}\nsource  {narration}\n"
            f"output  {project_root}"
            + ("\n[yellow]VR_OFFLINE=1[/yellow]" if offline_mode() else ""),
            title="run",
            border_style="cyan",
        )
    )

    ctx = RunContext(
        job_id=job.id,
        project=name,
        project_root=project_root,
        source_path=narration,
        settings=settings,
        db_path=db_path,
        force=force,
        no_clips=no_clips,
    )
    # §21: deliver each segment as it finishes, so selected/ fills front-to-back
    # instead of appearing all at once when the last stage ends.
    if batch:
        results = run_pipeline(ctx)
    else:
        results = run_segment_pipelined(ctx)
    _print_results(results)
    _print_outputs(ctx)

    if any(not r.ok for r in results):
        raise typer.Exit(code=1)


def _print_results(results) -> None:
    table = Table(title="Stages", title_justify="left", header_style="bold")
    for col in ("Stage", "Result", "Time", "Detail"):
        table.add_column(col)
    for r in results:
        if not r.ok:
            verdict = "[red]failed[/red]"
        elif r.skipped:
            verdict = "[dim]skipped[/dim]"
        else:
            verdict = "[green]ok[/green]"
        table.add_row(
            str(r.stage), verdict, f"{r.duration_s:.1f}s" if r.duration_s else "-", r.detail
        )
    console.print(table)


def _print_outputs(ctx) -> None:
    lines: list[str] = []
    for label, path in (
        ("transcript.json", ctx.paths.transcript_json),
        ("transcript.txt", ctx.paths.transcript_txt),
        ("narration.srt", ctx.paths.narration_srt),
        ("timeline.csv", ctx.paths.timeline_csv),
    ):
        mark = "[green]+[/green]" if path.exists() else "[dim]-[/dim]"
        lines.append(f"{mark} {label:<16} {path}")
    seg_dirs = ctx.paths.existing_segment_dirs()
    lines.append(
        f"[green]+[/green] segments/        {len(seg_dirs)} folders"
        + (f"  (first: {seg_dirs[0].name}, last: {seg_dirs[-1].name})" if seg_dirs else "")
    )
    console.print(Panel("\n".join(lines), title="Output", border_style="green"))


# ---------------------------------------------------------------------------
# status / resume
# ---------------------------------------------------------------------------


@app.command()
def status(job_id: str | None = typer.Argument(None, help="Show one job in detail.")) -> None:
    """List jobs, or show one job's stages."""
    from .jobs.queue import (
        JobNotFound,
        completed_stages,
        get_job,
        list_jobs,
        resume_point,
        stale_jobs,
    )
    from .jobs.states import STAGE_ORDER

    settings = _settings()
    db_path = _db(settings)

    if job_id is None:
        jobs = list_jobs(db_path, limit=50)
        if not jobs:
            console.print("[dim]No jobs yet. Run: visualresearch run <narration.wav>[/dim]")
            return
        # A job whose worker was killed keeps its running state forever, so the
        # table shows SEARCHING_IMAGES for a process that no longer exists and
        # there is no way to tell it from a live one. That is how a reader ends
        # up believing two runs are competing for the same project.
        stale_ids = {job.id for job in stale_jobs(db_path)}
        table = Table(title="Jobs", title_justify="left", header_style="bold")
        # The job id is what the user copies into `resume`, so it must never be
        # cropped; the project name is the column that gives way instead.
        table.add_column("Job", no_wrap=True, min_width=12)
        table.add_column("Project", overflow="fold")
        table.add_column("State", no_wrap=True, min_width=15)
        table.add_column("Stage", no_wrap=True)
        table.add_column("Delivered", justify="right", no_wrap=True)
        table.add_column("Size", justify="right", no_wrap=True)
        table.add_column("Created", no_wrap=True, min_width=16)
        for job in jobs:
            table.add_row(
                job.id,
                job.project,
                _state_text(job.state, stale=job.id in stale_ids),
                job.stage or "-",
                _segment_progress(job),
                f"{job.bytes_used / 1_048_576:.0f} MB" if job.bytes_used else "-",
                job.created_at.strftime("%Y-%m-%d %H:%M"),
            )
        console.print(table)
        return

    try:
        job = get_job(db_path, job_id)
    except JobNotFound as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc

    console.print(
        Panel.fit(
            f"project  {job.project}\nstate    {job.state}\nstage    {job.stage or '-'}\n"
            f"segments {_segment_progress(job)} delivered to selected/\n"
            f"source   {job.source_path}\norigin   {job.origin}"
            + (f"\nerror    [red]{job.error}[/red]" if job.error else ""),
            title=f"job {job.id}",
            border_style="cyan",
        )
    )
    done = completed_stages(db_path, job.id)
    resume_at = resume_point(db_path, job.id)
    table = Table(title="Stages", title_justify="left", header_style="bold")
    for col in ("Stage", "State"):
        table.add_column(col)
    for stage in STAGE_ORDER:
        if str(stage) in done:
            mark = "[green]done[/green]"
        elif stage == resume_at:
            mark = "[yellow]next[/yellow]"
        else:
            mark = "[dim]pending[/dim]"
        table.add_row(str(stage), mark)
    console.print(table)


def _segment_progress(job) -> str:
    """"N of 95" while a run is in flight, a plain count once it is not.

    With segment pipelining the number worth showing is how much has actually
    landed in selected/, not how many segments exist. A bare total told the
    reader nothing about progress during the 40 minutes it takes to fill.
    """
    total = job.segments_total or 0
    done = job.segments_done or 0
    if not total:
        return "-"
    if done and done < total:
        return f"{done} of {total}"
    return str(total)


def _state_text(state: str, *, stale: bool = False) -> Text:
    colour = {
        "COMPLETE": "green",
        "FAILED": "red",
        "REVIEW_REQUIRED": "yellow",
        "QUEUED": "dim",
    }.get(str(state), "cyan")
    if stale:
        # Say it in the state column rather than a separate one: the whole
        # point is that the state on its own is misleading.
        return Text(f"{state} (stale)", style="red")
    return Text(str(state), style=colour)


@app.command()
def resume(
    job_id: str = typer.Argument(..., help="Job to resume."),
    batch: bool = typer.Option(False, "--batch", help="Resume stage-by-stage instead."),
) -> None:
    """Pick a job up where it stopped, keeping the segments already delivered."""
    from .jobs.queue import JobNotFound, get_job, resume_point
    from .jobs.states import STAGE_ORDER
    from .jobs.worker import RunContext, run_pipeline, run_segment_pipelined

    settings = _settings()
    db_path = _db(settings)
    try:
        job = get_job(db_path, job_id)
    except JobNotFound as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc

    start = resume_point(db_path, job.id)
    remaining = list(STAGE_ORDER[STAGE_ORDER.index(start) :])
    project_root = settings.project_dir(job.project)
    add_job_log(project_root / "job.log")
    done = job.segments_done or 0
    note = (
        f"\n{done} of {job.segments_total} segment(s) already delivered - "
        "those are kept, not redone"
        if done
        else ""
    )
    console.print(
        f"Resuming job [bold]{job.id}[/bold] ({job.project}) at stage [bold]{start}[/bold]" + note
    )

    ctx = RunContext(
        job_id=job.id,
        project=job.project,
        project_root=project_root,
        source_path=Path(job.source_path),
        settings=settings,
        db_path=db_path,
        force=False,
        no_clips=job.no_clips,
    )
    # Pipelined by default, matching `run`: a resumed run skips the segments
    # whose files are already on disk, so picking up at segment 21 of 170 costs
    # 21 seconds of checking rather than 20 segments of live API calls.
    if batch:
        results = run_pipeline(ctx, stages=remaining)
    else:
        results = run_segment_pipelined(ctx)
    _print_results(results)
    _print_outputs(ctx)
    if any(not r.ok for r in results):
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------


@cache_app.command("clear")
def cache_clear(
    yes: bool = typer.Option(False, "--yes", help="Do not prompt."),
) -> None:
    """Empty the provider response cache. Only touches .cache/ (§2.8)."""
    from .db import session_scope
    from .models import CacheEntry

    settings = _settings()
    db_path = _db(settings)
    if not yes:
        typer.confirm("Clear the provider cache?", abort=True)
    with session_scope(db_path) as session:
        rows = session.query(CacheEntry).delete()
    removed = 0
    for child in settings.cache_dir.glob("*"):
        if child.is_file():
            child.unlink()
            removed += 1
        elif child.is_dir():
            shutil.rmtree(child)
            removed += 1
    console.print(f"Cleared {rows} cache rows and {removed} entries under {settings.cache_dir}")


# ---------------------------------------------------------------------------
# Not yet built (§22). These exist so the CLI surface matches §17 and tells the
# truth about what is not ready, rather than 'command not found'.
# ---------------------------------------------------------------------------


def _not_yet(command: str, phase: str) -> None:
    console.print(
        Panel.fit(
            f"[yellow]`{command}` is not implemented yet.[/yellow]\n"
            f"It is scheduled for [bold]{phase}[/bold] (CLAUDE.md §22).",
            border_style="yellow",
        )
    )
    raise typer.Exit(code=3)


@app.command()
def serve(
    port: int = typer.Option(8765, "--port", help="Port to listen on."),
    host: str = typer.Option("127.0.0.1", "--host", help="Interface to bind."),
    reload: bool = typer.Option(False, "--reload", help="Reload on code changes."),
    watch_folder: bool = typer.Option(
        False, "--watch", help="Also run the folder watcher and a worker pool (§18.7)."
    ),
) -> None:
    """Run the review web UI (CLAUDE.md §15).

    Binds to 127.0.0.1 by default: this serves the contents of your project
    folder with no authentication, so it should not be reachable from the
    network unless you deliberately change that.
    """
    import uvicorn

    settings = _settings()
    _db(settings)
    console.print(
        Panel.fit(
            f"Review UI on [bold]http://{host}:{port}[/bold]\n"
            f"projects: {settings.projects_dir}\n"
            "[dim]Ctrl-C to stop[/dim]",
            border_style="cyan",
        )
    )
    if watch_folder:
        # §18.7: the watcher runs inside serve. Started here rather than in
        # the app factory so --reload does not spawn a second one.
        from .jobs.pool import WorkerPool
        from .jobs.watcher import FolderWatcher

        FolderWatcher(settings, settings.db_path).start_background()
        WorkerPool(settings, settings.db_path, on_finish=_announce).start()
        console.print(f"[dim]watching {settings.watch.dir}[/dim]")

    uvicorn.run(
        "visualresearcher.web.app:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
        log_level="warning",
    )


def _notifier():
    """The notify provider, or None. A missing one never stops anything."""
    from .providers.base import ProviderUnavailable
    from .providers.registry import resolve as resolve_provider

    try:
        return resolve_provider("notify", "desktop")
    except ProviderUnavailable:
        return None


def _announce(outcome) -> None:
    """Tell the user a job finished (§22 P8: zero manual steps)."""
    notifier = _notifier()
    if notifier is None:
        return
    settings = load_settings()
    folder = settings.project_dir(outcome.project) / "selected"
    if outcome.ok and outcome.state != "FAILED":
        notifier.notify(
            f"VisualResearcher: {outcome.project} finished",
            f"{outcome.state} in {outcome.duration_s:.0f}s",
            path=folder,
        )
    else:
        notifier.notify(
            f"VisualResearcher: {outcome.project} FAILED",
            outcome.error or "see job.log",
            path=settings.project_dir(outcome.project),
            urgency="critical",
        )


@app.command()
def worker(
    concurrency: int | None = typer.Option(
        None, "--concurrency", help="Override worker.max_concurrent."
    ),
) -> None:
    """Run a queue worker (CLAUDE.md §19).

    Claims queued jobs and runs them, up to ``worker.max_concurrent`` at once.
    CPU-heavy stages share one global slot, so two jobs do not fight over the
    same cores.
    """
    from .jobs.pool import WorkerPool

    settings = _settings()
    if concurrency:
        settings.worker.max_concurrent = concurrency
    db_path = _db(settings)

    pool = WorkerPool(settings, db_path, on_finish=_announce)
    console.print(
        Panel.fit(
            f"worker pool: [bold]{settings.worker.max_concurrent}[/bold] concurrent job(s)\n"
            f"CPU-heavy slots: {settings.compute.max_concurrent_cpu_heavy}\n"
            f"projects: {settings.projects_dir}\n"
            "[dim]Ctrl-C to stop[/dim]",
            title="worker",
            border_style="cyan",
        )
    )
    pool.run_forever()


@app.command()
def watch(
    once: bool = typer.Option(False, "--once", help="Sweep once and exit."),
    no_worker: bool = typer.Option(
        False, "--no-worker", help="Only enqueue; do not run the jobs here."
    ),
) -> None:
    """Watch the drop folder for new narration files (CLAUDE.md §18).

    Reacts to ``.wav`` files starting with ``watch.filename_prefix``, waits
    until each one stops growing, moves it into a new project and enqueues it
    on the same queue the CLI uses. The same file is never processed twice,
    across restarts.
    """
    from .jobs.pool import WorkerPool
    from .jobs.watcher import FolderWatcher

    settings = _settings()
    db_path = _db(settings)
    watcher = FolderWatcher(settings, db_path)

    console.print(
        Panel.fit(
            f"watching  [bold]{settings.watch.dir}[/bold]\n"
            f"prefix    {settings.watch.filename_prefix}*.wav\n"
            f"stable    {settings.watch.stable_secs:.0f}s before processing\n"
            f"poll      every {settings.watch.poll_interval_s:.0f}s\n"
            + ("[dim]enqueue only[/dim]" if no_worker else "running jobs here too")
            + "\n[dim]Ctrl-C to stop[/dim]",
            title="watch",
            border_style="cyan",
        )
    )

    if once:
        events = watcher.scan_once()
        for event in events:
            console.print(f"[green]queued[/green] {event.source.name} -> {event.project}")
        if not events:
            console.print("[dim]nothing to do[/dim]")
        return

    pool = None
    if not no_worker:
        pool = WorkerPool(settings, db_path, on_finish=_announce)
        pool.start()
    try:
        watcher.run()
    except KeyboardInterrupt:
        console.print("\n[dim]stopping[/dim]")
    finally:
        watcher.stop()
        if pool is not None:
            pool.stop()


@app.command()
def rerun(
    project: str = typer.Argument(..., help="Project name."),
    segment: int = typer.Option(..., "--segment", help="Segment index to re-run."),
    stage: str | None = typer.Option(
        None, "--stage", help="image_search | video_search | clips | rank"
    ),
) -> None:
    """Re-run the research for one segment (CLAUDE.md §17, §20).

    Re-running one segment does not re-run the project. Only this segment's
    folder is touched, and `selected/` is rebuilt afterwards so the
    deliverable still matches the picks.
    """
    from .jobs.rerun import RERUNNABLE_STAGES, rerun_segment

    settings = _settings()
    try:
        result = rerun_segment(project, segment, settings, stage=stage)
    except LookupError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        console.print(f"[dim]valid stages: {', '.join(RERUNNABLE_STAGES)}[/dim]")
        raise typer.Exit(code=2) from exc

    table = Table(title=f"segment {segment:03d}", title_justify="left", header_style="bold")
    for col in ("What", "Count"):
        table.add_column(col)
    table.add_row("stages re-run", ", ".join(result.stages) or "-")
    for label, value in (
        ("candidates found", result.candidates),
        ("images downloaded", result.downloaded),
        ("kept after dedupe", result.kept),
        ("selected", result.selected),
        ("video results", result.videos),
        ("clips", result.clips),
    ):
        if value:
            table.add_row(label, str(value))
    console.print(table)
    for note in result.notes[:8]:
        console.print(f"[yellow]note[/yellow] {note}")


@app.command()
def collect(project: str = typer.Argument(..., help="Project name.")) -> None:
    """Rebuild selected/ from the current picks, and the reports with it (§6, §15, §21)."""
    from .pipeline.all_candidates import dump_all_candidates
    from .pipeline.collect import collect as run_collect
    from .pipeline.report import write_reports
    from .project import ProjectPaths
    from .schemas import ProjectContext, Segment
    from .utils.files import pad_width

    settings = _settings()
    paths = ProjectPaths(settings.project_dir(project))
    if not paths.root.exists():
        console.print(f"[red]No such project:[/red] {paths.root}")
        raise typer.Exit(code=2)

    segments = [
        Segment.model_validate_json((d / "segment.json").read_text(encoding="utf-8"))
        for d in paths.existing_segment_dirs()
        if (d / "segment.json").exists()
    ]
    width = pad_width(max((s.index for s in segments), default=1))
    result = run_collect(segments, paths, settings, width=width)

    # §21: sources.csv is the credits list for selected/, so rebuilding one
    # without the other leaves rows pointing at files that are no longer there.
    # Raising the confidence gate and re-collecting used to empty selected/ and
    # keep all ten credit rows -- a deliverable whose attribution was fiction.
    context_path = paths.root / "project_context.json"
    context = (
        ProjectContext.model_validate_json(context_path.read_text(encoding="utf-8"))
        if context_path.exists()
        else ProjectContext()
    )
    rows = write_reports(result, segments, context, paths, settings, width=width)
    # If this project has an all_candidates/ folder, its SELECTED markers
    # describe selected/ -- which just changed. Leaving them is how the folder
    # starts lying about which image is current.
    refreshed = dump_all_candidates(segments, paths, settings, width=width)

    table = Table(title="selected/", title_justify="left", header_style="bold")
    for col in ("Outcome", "Count"):
        table.add_column(col)
    table.add_row("files in selected/", str(result.total))
    table.add_row("rows in sources.csv", str(rows))
    table.add_row("newly copied", str(len(result.written)))
    table.add_row("unchanged", str(len(result.unchanged)))
    table.add_row("removed (no longer picked)", str(len(result.removed)))
    table.add_row("flagged medium confidence", str(len(result.flagged)))
    table.add_row("gaps (no confident pick)", str(len(result.gaps)))
    if result.foreign:
        table.add_row("left alone (not ours)", str(len(result.foreign)))
    console.print(table)
    if refreshed is not None:  # always, now
        console.print(
            f"[dim]all_candidates/: {refreshed.selected_count} SELECTED marker(s) "
            f"refreshed ({len(refreshed.remarked)} moved)[/dim]"
        )
    console.print(f"[green]{paths.selected_dir}[/green]")


@app.command("all-candidates")
def all_candidates(project: str = typer.Argument(..., help="Project name.")) -> None:
    """Flat, time-ordered folder of every ranked candidate for one project."""
    from .pipeline.all_candidates import DIR_NAME, dump_all_candidates
    from .project import ProjectPaths
    from .schemas import Segment
    from .utils.files import pad_width

    settings = _settings()
    paths = ProjectPaths(settings.project_dir(project))
    if not paths.root.exists():
        console.print(f"[red]No such project:[/red] {paths.root}")
        raise typer.Exit(code=2)

    segments = [
        Segment.model_validate_json((d / "segment.json").read_text(encoding="utf-8"))
        for d in paths.existing_segment_dirs()
        if (d / "segment.json").exists()
    ]
    if not segments:
        console.print(f"[red]No segments in[/red] {paths.segments_dir}")
        raise typer.Exit(code=2)

    width = pad_width(max((s.index for s in segments), default=1))
    result = dump_all_candidates(segments, paths, settings, width=width)

    table = Table(title=f"{DIR_NAME}/", title_justify="left", header_style="bold")
    for col in ("Outcome", "Count"):
        table.add_column(col)
    table.add_row("ranked candidates", str(result.total))
    table.add_row("segments covered", f"{result.segments_covered} of {len(segments)}")
    table.add_row("marked SELECTED", str(result.selected_count))
    table.add_row("newly copied", str(len(result.written)))
    table.add_row("unchanged", str(len(result.unchanged)))
    if result.remarked:
        table.add_row("marker moved", str(len(result.remarked)))
    if result.removed:
        table.add_row("removed (no longer ranked)", str(len(result.removed)))
    if result.missing_sources:
        table.add_row("source missing from disk", str(len(result.missing_sources)))
    if result.foreign:
        table.add_row("left alone (not ours)", str(len(result.foreign)))
    console.print(table)
    console.print(f"[green]{paths.root / DIR_NAME}[/green]")
    console.print(
        "[dim]Sorted alphabetically it is in narration order: each segment's "
        f"candidates form one block, best rank first. {SELECTED_SUFFIX_HINT}[/dim]"
    )


@app.command("open")
def open_project(project: str = typer.Argument(...)) -> None:
    """Open a project folder in the file manager."""
    settings = _settings()
    root = settings.project_dir(project)
    if not root.exists():
        console.print(f"[red]No such project:[/red] {root}")
        raise typer.Exit(code=2)
    if os.name == "nt":
        os.startfile(root)  # noqa: S606
    else:
        import subprocess

        subprocess.run(["xdg-open", str(root)], check=False)
    console.print(f"Opened {root}")


if __name__ == "__main__":  # pragma: no cover
    app()
