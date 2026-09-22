# CHANGELOG

All notable changes, newest first. Dates are the day the work landed.

---

## 2026-09-20 — First live run

Everything up to here was verified with `VR_OFFLINE=1`. This is the first time
the pipeline ran against the real internet, on a real 12m45s narration, and
the real services found a bug the fakes never could.

### Fixed

**Provider searches were not rate-limited** (`utils/politeness.py`,
`providers/images/wikimedia.py`, §13)
- Live symptom: `wikimedia failed on 'galaxy far away Star Wars environment
  wide shot': 429 Too Many Requests`, on the fourth segment of eight.
- Cause: §13's politeness layer was wired into `pipeline/download.py` only.
  Provider *search* calls went straight out through `httpx.get`. So the
  downloader was carefully pacing itself against `commons.wikimedia.org`
  while the search path hammered the same host with no gap at all — and a
  host counts all of our requests together, whatever our internal structure.
- Fix, in two parts:
  - `Politeness.SHARED`, one process-wide instance. Both the search path and
    the download path now go through it, so the per-host interval means what
    it says. A limiter per call site is not a limiter.
  - A 429 is retried (3 attempts, exponential backoff) and `Retry-After` is
    honoured when the server sets it. Previously a single 429 raised
    `ProviderError` and cost that segment an entire provider's worth of
    candidates — a transient "slow down" was being treated as a dead end.
- Tests (`tests/test_politeness.py`): a 429-then-200 returns results, a
  `Retry-After: 7` sleeps 7s, an unparseable header still backs off, a
  permanent 429 raises rather than looping forever, a search passes through
  the gate, and searches and downloads share *one* gate object.

### Added

**The flat `all_candidates/` dump** (`pipeline/all_candidates.py`, `cli.py`)
- `visualresearch all-candidates <project>` copies every **ranked** candidate
  into `projects/<name>/all_candidates/`, flat, named so that sorting
  alphabetically reproduces the narration exactly. `selected/` is the one
  confident pick per segment; this is the folder you scan to swap one out
  without opening `segments/NNN_.../images/` one at a time.
- **The ranked set, not the searched pool.** Ranking keeps
  `images.keep_per_segment` images with ranks 1..N; everything else that
  survived download and dedupe stays in the manifest at `rank: 0`. A first
  attempt at this dumped all of `kept` and numbered the leftovers by position,
  which invented ranks 9-18 for images nothing had ever ranked. On the live
  project that was 1367 files pretending to be candidates where 750 were real.
- Format: `{segment:0Wd}_{MMmSSs}_{rank:02d}_{slug}[_SELECTED].{ext}`. Three
  fields carry the ordering guarantee and each fails quietly:
  - **The padded segment number leads** and is the only field deciding order.
    Width comes from `pad_width()`, never a hardcoded 3.
  - **The timecode is decorative.** Past 99 minutes it gains a digit and
    `100m20s` sorts before `10m30s`, so timecode-first naming breaks every long
    narration.
  - **The rank is padded.** `keep_per_segment` is configuration; above 9 an
    unpadded rank puts `_10_` before `_2_`.
- **`_SELECTED` is a suffix, deliberately.** A leading marker would sort every
  chosen file into one clump at the top and destroy the chronological ordering
  that is the entire point of the folder.
- **The marker cannot go stale.** It is derived from `plan_selected()` -- the
  same function that decides `selected/` -- so the two cannot hold different
  opinions about which image is current. `collect` refreshes it automatically
  for any project that has the folder, so swapping a pick in the review UI
  moves the marker without a second command. A move is a *rename*, not a
  re-copy, so swapping one pick does not rewrite 750 files.
- Per project, always: `projects/<name>/all_candidates/`, never shared.
- Copies only; originals under `segments/` are never moved, and a file a person
  drops into the folder is never deleted (§2.8).
- Tests (`tests/test_all_candidates.py`, 15): ordering at 95 segments and at
  1200, segment 9 before 90, rank 10 after rank 2, narration past 99 minutes,
  the marker not disturbing order, the ranked-set filter, marker correctness
  and movement, idempotence, copies-not-moves, and per-project isolation. Each
  ordering test carries a negative control asserting the broken variant *fails*
  to sort, so none can pass vacuously.

### Calibrated

**Confidence thresholds, against real CLIP output** (`config.yaml`, §14)
- `medium` 0.60 -> **0.55**, `high` 0.85 -> **0.62**.
- These two numbers had never met real data. Every prior run used
  `VR_OFFLINE=1`, whose fake embedder has meaningless similarity by
  construction -- `PROGRESS.md` said as much: *"CLIP relevance is not yet
  verified on real content."* The first live run measured it: min 0.512,
  median 0.581, max 0.648 across 15 segments.
- At 0.60, 6 of 15 segments delivered anything; the rest became gaps with a
  perfectly good top pick sitting at 0.58. At 0.55, 13 of 15 deliver, and
  0.52 delivers the same 13 -- a flat part of the distribution, so the choice
  is robust rather than fitted to noise. Segment 010 (0.512) stays a gap,
  correctly.
- `high` at 0.85 was unreachable against a maximum of 0.648, so every asset
  was flagged for review forever and the band carried no information.
- The reasoning is written into `config.yaml` beside the values, because a
  bare number invites someone to "fix" it back.

### Fixed, round three

**`visualresearch collect` rebuilt the folder and not its credits** (`cli.py`, §21)
- It called `collect()` and stopped. `sources.csv`, `shotlist.md` and
  `research_report.md` were left as they were.
- Demonstrated: raise `confidence.medium` on a finished project, re-collect,
  and `selected/` empties to 0 files while `sources.csv` keeps all 10 rows,
  each naming a file that is no longer there. The deliverable's attribution
  became fiction, silently.
- Found while verifying the recalibration path *before* recommending it --
  which is the only reason it was not shipped as advice.
- Now writes all three reports alongside the folder, and reports the row count
  next to the file count so a mismatch is visible in the output.

### Added

**Segment pipelining** (`jobs/worker.py`, `cli.py`, §21)
- Each segment now flows search -> download -> dedupe -> rank -> video ->
  clips -> delivery on its own, in narration order, and lands in `selected/`
  and `sources.csv` as soon as it is done. Previously every stage ran across
  all 95 segments before the next began, so a 40-minute run showed an empty
  `selected/` until minute 39.
- **Not a second implementation.** `ctx.only_segments` narrows what
  `_load_segments` returns, so each stage function is called exactly as the
  batch runner calls it, handed a list of one. Two parallel implementations
  would drift, and the drift would appear as a delivered folder that differs
  depending on which mode produced it. `--batch` restores the old order.
- Two things had to be hoisted onto the context for this to be a *reordering*
  rather than a rewrite:
  - **CLIP** was built per stage call. Naively per segment that is 95 model
    loads; it is now built once, with a test that asserts it.
  - **The repetition window** (`SegmentHistory`) decides whether a shot repeats
    a recent one. It is causal -- it only looks backwards -- so feeding it
    chronologically produces identical scores. That is the deeper reason
    segments must run in narration order, beyond filling `selected/`
    front-to-back.
- **§21 holds throughout, not just at the end.** `deliver_segment()` calls the
  existing idempotent `collect()` and derives the CSV rows from that same
  `CollectResult`, so counts match by construction; it then verifies
  files == rows and logs an error if they ever diverge. One honest caveat: the
  copy and the CSV write are two operations, so a reader can catch a
  sub-millisecond window between them. The invariant is asserted at every
  segment boundary.
- `status` gained a **Delivered** column showing "N of 95" while a run is in
  flight, since the interesting number during those 40 minutes is how much has
  actually landed.
- Tests (`tests/test_pipelining.py`): pipelined and batch runs produce
  byte-identical `selected/` listings *and* identical `sources.csv` rows;
  `selected/` grows monotonically and is non-empty before the end; delivery is
  chronological; the invariant holds after every segment; progress reaches the
  database; the embedder is built at most once.

**Per-provider query caps** (`config.py`, `pipeline/image_search.py`)
- `images.max_queries_per_provider`, defaulting to `{wikimedia: 2}`.
- Chosen over lowering `candidates_per_segment`, which would not have helped:
  call count is `queries x providers`, and that setting controls `limit=`
  *inside* a call, not how many calls are made. It would have shrunk the
  ranking pool for no rate-limit benefit.
- Commons is capped rather than demoted to a fallback because it is the only
  provider returning a real licence -- ddgs answered 68 of 68 with
  `license=unknown`, from domains like `animalia-life.club`. Making it a
  fallback would have traded provenance, not relevance, and left `sources.csv`
  largely unattributable.

### Fixed, round two (same live run, later stages)

**robots.txt refused the entire internet** (`utils/politeness.py`, §13)
- `RobotFileParser.read()` fetches with urllib, which sends
  `User-Agent: Python-urllib/3.x`. Wikimedia and Cloudflare reject that, and
  `read()` turns a 401/403 into `disallow_all = True`. The tell was
  `entries=0`: it refused every URL on the host without parsing one rule.
- That inverts this module's own documented rule, and it was self-inflicted --
  we were blocked for not identifying ourselves, which is what `USER_AGENT`
  exists for. Caught before the download stage ran; it would have rejected all
  95 segments' candidates as `DISALLOWED` and produced an empty `selected/`.
- Now fetches robots.txt itself with a real User-Agent and treats 403, 404,
  timeout and garbage as *no opinion*. A readable `Disallow:` is still obeyed.

**A 429 was retried but never changed our rate** (`utils/politeness.py`)
- Retrying worked and was still not enough: 32 segments cost **14 refusals and
  10.6 minutes of pure `time.sleep`**, because each retry went back to the same
  interval that had just been refused.
- `HostRateLimiter.penalize()` now doubles that host's interval per refusal,
  capped at 16x, per host. A refusal is the host telling us our rate is wrong.

**The rate-limit log line never said "429"** (`providers/images/wikimedia.py`)
- It read `wikimedia rate-limited; waiting 44.0s`. Every filter watching for
  `429` -- including the one watching this run -- matched nothing, and the run
  was reported as having zero rate limits while sitting through fourteen.
- Now logs at WARNING, names the status code, the attempt, and the new factor.

**`.env.local` was documented but never read** (`config.py`)
- `providers/llm/openai.py` tells people to put `OPENAI_API_KEY` there. Nothing
  loaded it: follow the instruction and auth still fails with no clue why.
- `load_local_env()` now loads it (gitignored, secrets allowed, real
  environment always wins). `VR_CONTACT` goes there and feeds the User-Agent,
  which Wikimedia's policy asks for and throttles harder without.

**`pytest -q` swallowed its own summary** (`pyproject.toml`)
- `addopts` already carried `-q`, so the documented command became `-qq` and
  the "N passed" line disappeared. Removed from `addopts`.

**A killed job looked alive forever** (`jobs/worker.py`, `cli.py`, §18)
- `stale_jobs()` existed, documented, with zero callers, so `status` showed
  `SEARCHING_IMAGES` for a process that no longer existed -- indistinguishable
  from a live one.
- Wiring it in exposed a worse bug: `heartbeat()` fired once per *stage*, so
  the beat period was the stage duration, and `image_search` over a 20-minute
  narration outlasts `STALE_AFTER`. A healthy job would have been libelled, and
  raising the threshold would only have hidden dead ones for longer. The
  heartbeat now ticks on a 30s clock for the life of the job.

### Verified live (no `VR_OFFLINE`)

- **OpenAI** — `gpt-4o-mini`, real completion, 54 tokens.
- **faster-whisper** — real transcription, not the fake. Had to be moved into
  a `uv` dependency-group: as an extra, `uv sync` silently pruned it.
- **Wikimedia Commons** — real results with real licences (`CC BY-SA 2.0`).
- **ddgs** — real results.
- **yt-dlp** — real video results with real durations.

---

## 2026-09-19 — P8 Automation

Dropping a file into a folder now produces a finished project and a
notification, with no manual step in between.

### Added

**Folder watcher** (`jobs/watcher.py`, §18)
- Reacts only to `.wav` files matching `watch.filename_prefix`,
  case-insensitively.
- **Stability guard**: nothing is touched until the file's size has been
  unchanged for `watch.stable_secs`. A drop is not an event, it is the *start*
  of one, and half a narration transcribes into a plausible-looking wrong
  result rather than an obvious failure.
- Claims the filename in SQLite **before** any work, so a crash mid-ingest
  cannot cause a re-run. Survives restarts.
- Moves the file into the project and enqueues on the same queue the CLI
  uses — identical downstream path, no special-casing.
- `watchdog` when available, polling otherwise, and a periodic sweep even in
  watchdog mode: events get missed on network shares, and a missed drop is
  invisible, because the user just waits for a project that never appears.
- Never crashes the loop. One bad file is logged, left untouched, and skipped.

**Worker pool** (`jobs/pool.py`, §19) — `worker.max_concurrent` jobs at once,
each in its own thread, sharing one set of limits.

**Shared limits** (`jobs/limits.py`, §19)
- `ComputeLimiter` — one global slot for transcription and CLIP ranking.
- `DownloadBudget` — a *total* byte budget and request interval, not per-job.
  Three jobs each politely rate-limiting themselves still triple the rate a
  remote host sees, and it is the host's view that gets you blocked.
- `DiskArbiter` — checks free space against the **sum** of active jobs'
  reservations. A per-job check passes twice on the way to filling the drive.

**Notifications** (`providers/notify/`) — Windows toast via PowerShell with a
console line that always happens. Best-effort by design: every failure path
returns False, because a run that crashes on the last line because a toast
could not be shown has failed at the moment it had succeeded.

**Cost and timing report** — `research_report.md` now shows per-stage timings
with their share of the run, the project size, and what the run actually cost:
image and video search need no key, transcription is local, and §20's batching
means an *N*-segment video is `1 + ceil(N/20)` LLM calls rather than `N + 1`.

**CLI** — `worker`, `watch`, `watch --once`, `watch --no-worker`, and
`serve --watch` to run the UI, the watcher and a pool in one process (§18.7).

### Fixed

- **A running job put itself back in the queue.** `STAGE_STATE[Stage.INGEST]`
  mapped to `QUEUED`, which is the state `claim_next` hands out. Single-
  threaded that was invisible; with a pool, the first stage made its own job
  claimable again, a second worker took it, and two threads wrote the same
  `project_context.json` at once — surfacing as a `PermissionError` on a
  `.partial` file and a duplicate `stage_checkpoints` row. Ingest is now part
  of the TRANSCRIBING phase, and a test asserts no stage may report a
  claimable state while running.
- **A transient move failure blacklisted a file forever.** The watcher claims
  before moving; a failed move marked the claim `failed` but kept the row, so
  `claim_filename` refused it ever after. A disk-full or file-locked error
  would have meant the user's narration was silently never processed — the
  worst kind of failure, because nothing reports it. Failed claims are now
  discarded outright; successfully processed ones still keep their row, which
  is what §18.6 actually guarantees.
- **`stage_timings` read SQLModel rows after their session closed**, raising
  `DetachedInstanceError` in the final stage of a run — after all the work was
  done. Caught by the end-to-end watcher run, not by a test.
- **The CLIP ranking slot could leak** if a segment raised mid-ranking,
  wedging every other job behind it. Now held in a `try/finally`.

- **`atomic_write_bytes` did not retry the rename.** On Windows `os.replace`
  onto a just-created file intermittently fails with `WinError 5` while the
  indexer or an antivirus scanner still holds a handle; two concurrent jobs
  make it likely. It appeared as a stage crashing mid-run. Retried with
  backoff for `PermissionError` only — a full disk is still reported at once.
- **`reset_engines()` raced the worker pool**, iterating the SQLAlchemy engine
  cache while another thread inserted into it. Now lock-guarded, including the
  double-create race for the same path.
- **The P8 timing table broke the resume-identity guarantee.** §21 requires a
  resumed run to be byte-identical, and wall-clock durations never repeat. The
  comparison now excludes that section only.

### Tests — see PROGRESS.md for the current count (was 501 at MVP-1)

- `test_watcher.py` (31) — the prefix matched four ways and rejected six;
  a **real background thread writing the file** while the stability guard
  refuses it; the ledger surviving a restart; a permission error on one file
  not stopping the next; a failed move leaving the file retryable.
- `test_concurrency.py` (19) — six threads proving only one CPU-heavy stage
  runs at a time; an exception not leaking a slot; the disk guard refusing
  once another job reserves space; **two real jobs run through the pool** with
  every path checked for cross-contamination.
- `test_notify.py` (9) — the only behaviour that matters is that it never
  raises.
- `test_cache.py` (26) — the TTL either side of the boundary, an expired entry
  removed on read rather than left to be re-checked forever, and an unusable
  database degrading to a miss.
- `test_politeness.py` (15) — a disallowed path refused all the way through
  the downloader, an unreadable robots.txt meaning allowed, and one host's
  rate limit not slowing another's.
- `test_rerun.py` (11) — every *other* segment byte-identical after a re-run.


### Also in this phase

- **`cache/store.py`** — §20's provider cache with its 30-day TTL, which was
  the last unimplemented item on §21's list. Expiry is checked on read rather
  than swept on a timer, so a stale entry cannot be served even if nothing has
  cleaned up, and every failure path degrades to a miss: a cache that can fail
  a job is worse than no cache.
- **The test suite was rebuilding a full pipeline per test.** `completed_run`
  and the rerun fixture were function-scoped, so `test_pipeline_run.py` alone
  ran the whole pipeline eighteen times and `test_rerun.py` twelve more — at
  roughly half a minute each. §2.5 says to run `pytest -q` constantly, which
  a twenty-five-minute suite actively discourages. Both now build once per
  module; the tests that mutate, resume or force a failure still construct
  their own isolated context, so nothing is shared that should not be.
- **`utils/politeness.py`** — robots.txt and per-host rate limiting, which
  §13 asks for ("Respect robots.txt and rate limits") and which the downloader
  had not been doing. Two decisions are worth stating: a robots.txt we cannot
  *read* means allowed, because treating a timeout as a prohibition would make
  the tool refuse most of the web over a transient error; and the rate limit
  is per host rather than global, because politeness is something the remote
  server experiences, and throttling Wikimedia because we just fetched from
  imgur helps nobody. `file://` URLs skip both, so the offline fake is neither
  consulted nor throttled.
- **`jobs/rerun.py`** — `visualresearch rerun PROJECT --segment N [--stage S]`.
  §20 says re-running one segment must not re-run the project, so the test for
  it asserts every *other* segment's files are byte-identical afterwards.
  `selected/` is rebuilt at the end so the deliverable still matches (§15).
- **`web/` split into `app.py`, `routes_ui.py` and `api.py`** to match §5's
  layout. Worth doing beyond tidiness: the mutating routes now sit together,
  where the "re-collect before returning" rule is visible as a pattern rather
  than a line that could be quietly forgotten in one handler out of twelve.

### Decisions worth recording

- **Threads, not processes.** The work is dominated by subprocesses (ffmpeg,
  yt-dlp), network I/O and torch, all of which release the GIL, and the parts
  that do not are already serialised behind the compute semaphore. Processes
  would buy nothing and cost a second SQLite connection and a second copy of
  the model.
- **The periodic sweep stays even in watchdog mode.** It looks redundant until
  the drop folder is a network share.


---

## 2026-09-19 — P6 Review UI and P7 Collection — **MVP-1**

All fifteen stages implemented. A narration file goes in; a flat, ordered,
credited folder of assets comes out.

### Added

**`selected/`** (`pipeline/collect.py`) — the deliverable
- Flat, `{segment:03d}_{pick:d}_{slug}.{ext}`, copies not moves, no symlinks.
- §15's "adds and removes, no orphans" and §2.8's "never destroy user data"
  pull against each other, so removal is narrow: a manifest at the project
  root records the files *this tool* created, and only those are ever deleted.
  A file the user dropped into `selected/` themselves is left alone and
  reported. Without a manifest nothing is ours, so nothing is removed.
- The manifest lives outside `selected/` so the deliverable folder stays pure.

**Review UI** (`web/`) — FastAPI + Jinja2 + HTMX, no client framework, no build
- Jobs page: state, stage, segments, disk used, timing.
- Project page: chronological cards, band filter, gap count at the top, and a
  jump-to-next-gap that *cycles* rather than always returning to the first.
- Segment card: timecode, narration, interpretation, editable and reversible
  entities, queries, image grid with USE THIS / favourite / reject / open
  source, YouTube results with copy-timestamp buttons and inline clip preview,
  confidence badge, notes, mark-reviewed.
- Every mutating route re-runs collect, so there is no save button and no way
  to leave the page and the folder out of step.
- The file server enforces the same traversal guard as §11's downloader.

**Reports** (`pipeline/report.py`)
- `sources.csv` — §13's eleven columns in §13's order, built from the collect
  result rather than a directory listing, so "every asset gets a row" holds by
  construction. A licence that is not known is written `unknown`.
- `shotlist.md` — shots per segment, medium-confidence flags, and **the gaps**
  with the reason each one exists.
- `research_report.md` — entity corrections with reasons and whether they were
  applied, what the packs identified, degraded segments, and how to read the
  bands.

**Job states** — a job with a gap ends REVIEW_REQUIRED; only a gapless run
reaches COMPLETE (§14).

### Fixed

- **`uv sync` silently uninstalled torch.** The ranking dependencies were an
  optional extra, and `uv sync` prunes extras. CLIP ranking would have quietly
  stopped working on any fresh sync. Moved to a dependency group listed in
  `tool.uv.default-groups`.
- **Clips were downloaded and then excluded from the deliverable.** The clip
  pick used its *timestamp* confidence (0.46) as its §14 band, so an
  EXACT_SCENE clip cleared the download gate, cost the bandwidth, and never
  reached `selected/`. Those two numbers answer different questions:
  `clip_confidence()` now weights the classification and lets an uncertain
  timestamp discount it rather than dominate it.
- **USE THIS could mark a thumbnail used while no file appeared.** §14's bands
  govern what the pipeline copies *automatically*; an explicit human choice is
  not automatic. `Pick.user_set` now carries that distinction, so the page and
  the folder cannot disagree — which is precisely what §15 forbids. Found by
  clicking it against the running server, not by a test.
- **A cached clip landed under a different filename than a fresh one.**
- **Segments with no clip recorded no reason**, leaving an unexplained empty
  folder.
- **"How to bake sourdough bread" was classified as commentary** rather than
  simply irrelevant.

### Tests — 501 passing (was 416)

- `test_collect.py` (29) — §21's "changing a pick adds/removes exactly the
  right file, zero orphans", stated literally; a user-added file surviving; a
  missing manifest making removal conservative; 300 segments still sorting.
- `test_web.py` (28) — every §15 element asserted present in the card; path
  traversal blocked three ways; and several tests that assert on **files on
  disk after an HTTP request**, because the HTML is not the product.
- `test_report.py` (21) — §21's "sources.csv row count == selected/ file
  count", including after a pick changes and when metadata is missing.

### Decisions worth recording

- **Deletion in `selected/` is allowed, but only of files we created.** §15
  needs it and §2.8 forbids it in general; the manifest is what lets both be
  true at once.
- **An explicit USE THIS overrides the confidence band.** §14's bands describe
  automatic behaviour. A human who has looked at the image and clicked has
  already made the judgement the band was standing in for.


---

## 2026-09-19 — P5 Clips

A known cutscene segment now yields EXACT_SCENE, a located timestamp with
stated evidence, and a correctly trimmed local mp4.

### Added

**Video providers** (`providers/video/`)
- `ytdlp.py` — `ytsearchN:` search (no key, no quota) and the §12.4 section
  command verbatim: `--download-sections`, `--force-keyframes-at-cuts`, the
  1080p format selector. Driven as a subprocess so the exact arguments are
  loggable and reproducible by hand.
- `fake.py` — synthesises source videos with ffmpeg and **trims them with
  ffmpeg**, so an offline run produces a real, playable mp4 of the right
  duration. Each carries captions, chapters and a description naming its
  subject at a known time, so all four evidence sources are exercised.

**Classification** (`pipeline/video_search.py`) — §7's four buckets, each with
a stated reason naming the evidence. "EXACT_SCENE" with no justification is a
claim the user has to verify by watching, which is the work this removes.

**Timestamp location** (`pipeline/timestamps.py`) — all five §12.3 methods, in
the specified confidence order, each recording `method`, `evidence` and
`confidence`. All five run rather than stopping at the first hit, because the
review UI shows the alternatives.

**Clip download** (`pipeline/clips.py`)
- Only the located span, padded ±`clips.padding_s` and clamped to the video.
- The full-download fallback is the *pipeline's* decision, not the provider's,
  and is refused above `max_full_download_minutes`.
- Cached by `(video_id, start, end)` rounded to 0.1s, with a JSON index that
  survives a restart. The real run showed this working: the clips stage went
  from 15.2s to 0.6s on a re-run.
- Below `MIN_CONFIDENCE` (0.45) nothing is downloaded — §6 forbids fabricating
  a pick, and twelve seconds of an arbitrary video is worse than a visible gap.
  The clickable `&t=` link is kept either way (§12.7).

### Fixed

- **A cached clip landed under a different filename than a fresh one**, so the
  same clip had two possible names and a re-run differed on disk. Caught by
  the resume-identity test.
- **Segments with no clip recorded no reason.** Three of seven segments had
  only CONTEXTUAL hits, so nothing was downloaded — correctly — but the
  reviewer saw an empty `clips/` folder with no explanation. Every segment now
  says why, which is what P7 turns into the gap list.
- **"How to bake sourdough bread" was classified as commentary.** True but
  useless: the real problem is that nothing from the segment appears in it.
  Relevance is now checked before the title markers.
- **A playthrough with partial relevance fell through to a generic reason.**
  It now says it is playthrough footage and whether chapters exist.

### Tests — 416 passing (was 362)

`test_clips.py` (48) — the section command asserted argument by argument; the
trimmed clip's duration measured with **ffprobe** against the located span;
the full-download guard both refusing a 90-minute source and permitting a
5-minute one; a provider that writes a partial file and then fails, with
`clips/` asserted clean afterwards; the cache counted to prove exactly one
fetch across two runs.


---

## 2026-09-19 — P4 Ranking

Every segment now has a ranked, visually varied top-8 with a full score
breakdown and a contact sheet.

### Added

**Embedding providers** (`providers/embedding/`)
- `openclip.py` — ViT-B-32 / laion2b_s34b_b79k per §4, lazily imported.
  Weights follow `HF_HOME`; `device: auto` picks CUDA only when torch reports
  a working CUDA device, because §3 says CPU-only unless confirmed.
- `fake.py` — images embed from actual pixels (a coarse colour signature), so
  visually similar images land near each other and diversity is genuinely
  testable offline. Text embeds from a hash, so image-to-text similarity is
  **arbitrary** — and the provider says so via `meaningful = False`.
- torch 2.14.0+cpu and open-clip-torch 3.3.0 installed, pinned in
  `pyproject.toml` to the CPU wheel index so `uv sync` cannot pull CUDA.

**Ranking** (`pipeline/rank.py`)
- Every term §11 lists: CLIP similarity, entity bonus, exact-event bonus,
  source tier, resolution, Laplacian sharpness, border-edge watermark penalty,
  and a recency-weighted cross-segment repetition penalty.
- Sharpness and the watermark check use numpy gradients, not scipy or OCR —
  §11 asks for border edge-density specifically "no OCR dep".
- Resolution is a saturating curve: 24MP is not twice as useful as 12MP for a
  1080p video, and a linear score lets one enormous image outrank a relevant
  one.
- Deterministic: ties break on sha256, so ranking does not depend on the order
  records arrived in.

**Diversity — selection, not filtering**
- Greedy maximal-marginal-relevance. Each candidate's effective score is
  reduced by how much it resembles what has already been picked, so the second
  copy of a picture loses to a different picture even at a higher raw score.
- This is why ranks 4–8 are not in descending raw-score order in the output:
  rank 4 scored 0.600 while rank 5 scored 0.649. Rank 4 was chosen because it
  was *different*. That is the rule working, not a bug.

**Contact sheets** (§6) — a 4-wide grid per segment, each cell labelled with
rank, score and filename, so a bad ranking is visible at a glance.

**The CLIP weight is dropped when the embedder says so.** Offline, the
similarity is arbitrary; scoring 45% of the result on an arbitrary number
would be worse than not scoring it. The weight is redistributed across the
terms that do mean something and the run logs that it happened.

### Fixed

- **`FakeEmbeddingProvider` took no constructor arguments** while the registry
  passed it four, so `VR_OFFLINE=1` — the path that is supposed to always work
  — failed at the ranking stage. Now every registered provider is tested
  against the full set of kwargs the pipeline passes, and every provider kind
  is checked to have a fake.
- **`test_unimplemented_stages_are_skipped_not_faked` went stale every phase.**
  It now reads the implemented and planned sets from the runner, and asserts
  their union is the whole of `STAGE_ORDER`.

### Tests — 362 passing (was 315)

`test_ranking.py` (39) — the diversity test builds a pool where the top eight
raw scores really are eight near-identical portraits (asserted), then requires
the selection to return at most two of them; sharpness against a
Gaussian-blurred copy; the watermark penalty against a striped border band;
determinism and order-independence; the repetition penalty demoting a reused
image below a fresh one; contact sheets surviving a missing file.


---

## 2026-09-19 — P3 Images

Every segment now has 26–29 validated, deduplicated image candidates with full
§7 metadata, downloaded and verified entirely offline.

### Added

**Image providers** (`providers/images/`)
- `wikimedia.py` — Commons MediaWiki API with `extmetadata`, so licence and
  author arrive with the result. The only source that reliably gives both,
  which makes it the most valuable provider for `sources.csv`.
- `ddgs_provider.py` — DuckDuckGo, no key. Records `license: unknown` rather
  than guessing (§13).
- `fake.py` — writes **real image files** and returns `file://` URLs. A fake
  returning fabricated metadata would skip the download validation, the Pillow
  decode, the hashing and the dedupe pass, which is most of what §11 specifies.

**Download and validation** (`pipeline/download.py`)
- 15 MB cap enforced **while streaming**, not from `Content-Length` — a server
  that lies about its length must not be able to fill the disk.
- Content-type allowlist as a cheap pre-filter; the real decision is made by
  decoding the bytes with Pillow.
- SVG rejected by URL, by sniffed bytes, and by decoded format.
- Saved filenames are built from our own counters. The remote filename is
  never used, and the destination is checked against the images directory
  before writing.
- Written to `.tmp/` and moved on success, so `images/` never holds a partial.

**Dedupe** (`pipeline/dedupe.py`)
- SHA256 first, then pHash **and** dHash at Hamming ≤6. Both are used because
  they fail differently: dHash follows horizontal gradients, pHash works on
  frequency content. Where one is fooled the other usually is not.
- Nothing is deleted. Discards stay in the manifest with `status="duplicate"`
  and a note naming the file they duplicate.
- Deterministic: records are considered best-first, so the same survivor is
  chosen regardless of arrival order.

**Search orchestration** (`pipeline/image_search.py`)
- Budget spread across queries and interleaved, so a segment whose
  exact-event query finds nothing is not left with nothing.
- A provider that raises — `ProviderError` or anything else — is recorded and
  the segment marked `degraded`; the other providers still run.
- `manifest.json` records kept, duplicate **and** rejected entries with
  reasons.

### Fixed

- **Only 9–10 candidates survived per segment**, against §22 P3's ≥15. The
  fake reserved a fixed number of slots for duplicates and invalid files,
  which is realistic at 30 results but swamps a 10-result request — it was
  leaving one usable image per query. Now proportional.
- **Duplicates and rejects were never reached in a real run.** They were
  appended last, and the download stage stops once it has its budget, so
  dedupe and the rejection paths silently never ran end to end. The fake now
  shuffles deterministically; a real search result is not sorted by how
  well-formed it is.
- **`manifest.json` stored absolute paths**, so a project folder could not be
  moved and two identical runs produced different bytes — which was masking
  the resume-identity test. Paths are now project-relative on disk and
  resolved on read.

### Tests — 315 passing (was 260)

`test_images.py` (48) — validation of oversize, undersize, undecodable, empty,
three SVG shapes and HTML-masquerading-as-an-image; hostile remote filenames
including traversal and Windows device names; no partial files; full metadata;
licence never invented; dedupe against exact, resized, cropped and
recompressed copies; six distinct images staying distinct; determinism under
input order; an unhashable file not matching everything; provider failure
containment.

### Decisions worth recording

- **The offline fake writes real bytes.** It is slower and more code than
  returning metadata, and it is the only way `VR_OFFLINE=1` actually tests
  §11 rather than just walking past it.
- **Both perceptual hashes, either one matching.** One hash would be cheaper;
  two catch different failure modes, and false negatives here mean eight
  near-identical portraits in the deliverable, which §11 names as the thing to
  avoid.


---

## 2026-09-19 — P2 Understanding

The pipeline now knows what the narration is *about*. All four known
mishearings from §10 resolve with stated reasons, and every segment gets five
search queries across five distinct kinds.

### Added

**Domain packs** (`domain_packs/`)
- `swtor.yaml` — the four seeded mishearings (Barriss→Darth Baras,
  Valron→Darth Vowrawn, Sanx→Colonel Senks, Drog→Lord Draahg) plus supporting
  cast, places and terminology. Every entry carries a confidence and a reason.
- `star_wars.yaml` — the wider franchise, deliberately shallower.
- Packs activate on trigger words and only when `min_triggers` are present, so
  a cookery narration cannot start renaming people to Darth Vader.
- Active packs are ordered by **trigger-hit count**, so the more specific pack
  leads. Without this, `star_wars` sorted ahead of `swtor` alphabetically and
  the subject tag on every query degraded from SWTOR to "Star Wars".
- `search_tag` — the short qualifier §7's example queries actually use
  ("Darth Baras Dark Council SWTOR", not the full title).

**Entity resolution** (`pipeline/entities.py`)
- §10.4's four passes in order: override → domain pack exact → domain pack
  fuzzy → phonetic → LLM. Each pass sees only what the previous could not
  resolve, so the LLM is asked about the fewest possible names.
- Confidence is discounted by match method: an exact alias is what the pack
  author wrote down; a fuzzy or phonetic hit is inference and scores lower.
- §10.3 honoured both ways: at ≥0.75 the correction is applied to query text;
  below it, `both_spellings()` emits the variant so ranking decides.
- `entity_overrides.yaml` written as a commented template on first run, never
  overwritten afterwards, and always winning at confidence 1.0.
- `apply_corrections()` is whole-word, case-insensitive, longest-original
  first, so "Lord Drog" does not become "Lord Lord Draahg".

**Metaphone** (`utils/phonetics.py`)
- Written out rather than added as a dependency (§2.10). Verified against the
  standard cases (Philip→FLP, Thomas→0MS, School→SKL, Science→SNS) and against
  the project's own: Barriss/Baras→BRS, Sanx/Senks→SNKS, Drog/Draahg→TRK.
- Its limit is documented rather than hidden: Valron→FLRN and Vowrawn→FRN do
  not match, which is exactly why the domain pack is consulted first.

**LLM providers** (`providers/llm/`)
- Interface takes a finished prompt and a machine-readable `task` label;
  providers stay I/O-only (§2.7).
- `FakeLLMProvider` — deterministic, offline. It **invents no entity
  corrections**: a stand-in that guessed would violate §10.1, so it returns an
  empty list and lets the packs and phonetics be the only offline truth.
- `OpenAILLMProvider` — REST over httpx rather than the vendor SDK, temperature
  0, retries on 429/5xx, `json_object` response format.

**Project context** (`pipeline/context.py`)
- `build_context()` fills `project_context.json` from the packs first and asks
  the model only for the gaps. A model answer never overwrites a pack value.
- `analyze_segments()` fills topic, entities, event, location, interpretation,
  visual_intent and sub-shots, **batching 20 segments per call** (§20) — a
  300-segment video is 15 calls, not 300.
- A failed batch degrades only its own segments (§8); they keep a heuristic
  analysis and are marked `degraded`.
- Heuristic analysis works with no model at all, which is what `VR_OFFLINE=1`
  runs on, and caps its own confidence at 0.55 so §14 routes it to review.

**Query generation** (`pipeline/queries.py`)
- Per-intent plans produce queries of *differing kinds*, because four
  rewordings of one phrase return the same thirty images.
- Sub-threshold corrections add the alternative spelling as an extra query.

### Fixed

- **A user override was recorded but not applied.** `_from_override` never set
  `applied`, so it defaulted to False — the exact opposite of §10.5's "always
  wins, permanently". Caught by `test_an_override_beats_the_domain_pack`.
- **Resuming at `queries` would have lost the corrections.** They lived only in
  `RunContext`. Each stage now leaves a complete artifact on disk, and
  `_load_corrections()` reads them back out of `project_context.json`.
  Without this, a resumed job would have searched the misheard spellings and
  quietly returned the wrong images.

### Tests — 260 passing (was 173)

- `test_entities.py` (45) — the four pairs, threshold behaviour either side of
  0.75, overrides winning, `never_correct`, whole-word replacement, LLM
  batching, LLM failure degradation, and a structural test that `transcribe.py`
  never imports entity resolution (§10.2).
- `test_context_and_queries.py` (35) — pack ordering and determinism, §20
  batching at 45 and 300 segments, per-batch degradation, sub-shot clamping,
  and the P2 acceptance criterion stated literally.

### Decisions worth recording

- **`search_tag` added to `ProjectContext`.** Additive to §7's example JSON,
  and justified by it: the example queries read "... SWTOR", which no field in
  the documented shape supplies.
- **The offline LLM resolves no entities.** It would have been easy to have the
  fake return plausible corrections and make the offline output look richer.
  That would be fabrication with a fabricated reason attached, which §10.1
  forbids more clearly than almost anything else in the constitution.


---

## 2026-09-19 — P1 Foundation

First working vertical slice: a narration file in, a chronologically-ordered
folder of segments out. Runs end to end with `VR_OFFLINE=1`, no credentials and
no network.

### Added

**Project skeleton**
- `src/` layout, uv-managed, Python 3.12.14 pinned via `.python-version`.
- `pyproject.toml` with the §4 stack. Dev group: pytest, pytest-asyncio, respx,
  ruff.

**Configuration** (`config.py`)
- Full §16 schema as Pydantic v2 models with the documented defaults.
- `config.yaml` overrides key by key; unnamed keys keep their defaults.
- A relative `output.root` resolves against the install root, never the shell's
  cwd — a relative path must not be able to land on C:.
- `load_env_defaults()` reads cache locations from `.env` for variables that
  are not already set, from a fixed allowlist.

**Contracts** (`schemas.py`)
- `Transcript`, `Word`, `TranscriptSegment`.
- `ProjectContext`, `EntityCorrection`, `Work` — §7 shapes, ready for P2.
- `Segment` with `visual_intent`, `queries`, `sub_shots`, `picks`.
- `ImageRecord` and `YouTubeRecord` with every field §7 lists, ready for P3/P5.

**Output layout** (`project.py`)
- `ProjectPaths` — the §6 folder contract in one place, so fifteen stages
  cannot each invent their own joins.

**Persistence** (`db.py`, `models.py`)
- SQLite with WAL, so the worker can write while the UI reads.
- Tables: `jobs`, `stage_checkpoints`, `cache_entries`, `seen_files`.
- `seen_files` has a unique index on the filename — §18.6's "never process the
  same filename twice", surviving restarts.

**Queue** (`jobs/queue.py`)
- `enqueue` rejects an empty project or source path, and reuses a job id rather
  than duplicating it.
- `claim_next` claims via a conditional UPDATE, so two workers cannot both win
  the same job.
- Checkpoints, `resume_point`, `clear_checkpoints`, heartbeats, stale detection.
- `claim_filename` / `release_filename` for the P8 watcher.

**Runner** (`jobs/worker.py`)
- Walks `STAGE_ORDER`, skipping checkpointed stages unless `--force`.
- Unbuilt stages announce the phase that will build them and are skipped — they
  never write a placeholder to look finished.
- A stage failure marks the job FAILED with the error preserved; a disk-space
  abort is reported separately and names the drive.

**Pipeline stages**
- `ingest` — copies (never moves) from the CLI, checks disk headroom first,
  probes the WAV, `.bak`s before any forced overwrite.
- `transcribe` — writes `transcript.json`, `transcript.txt`, `narration.srt`;
  repairs non-monotonic word timings; skips when the artifact already exists.
- `segment` — see below; also writes `timeline.csv`.

**Segmentation** (`pipeline/segment.py`)
- Dynamic program over word indices rather than a greedy chop, because §9's
  four constraints have to hold simultaneously and greedy strands later windows.
- Cost is `boundary_penalty + 0.35 * ((dur - target)/target)²`. Sentence end
  0.0, clause 0.35, silence ≥250ms 0.70, mid-phrase 1.20. Boundary quality
  dominates duration error by roughly 13×, so a natural boundary always beats
  hitting 8s — §9's rule three as arithmetic.
- Cuts land at the midpoint of the silence between two words, which makes the
  output contiguous and gapless by construction.
- Falls back to grouping the transcriber's own segments when no word timings
  exist, and relaxes only the *final* segment's minimum when no in-bounds
  segmentation exists at all (11s of unbroken speech, for example).

**Providers** (`providers/`)
- `Provider` base with a non-throwing `availability()` so `doctor` can report
  on something that is not installed.
- Registry enforces `VR_OFFLINE=1` in one place; no stage can bypass it.
- `FakeTranscriptionProvider` — replays a `.transcript.json` sidecar when one
  exists, otherwise synthesises a deterministic transcript from the audio's
  real duration. Works on any WAV with no fixture.
- `FasterWhisperProvider` — lazy import, honours `HF_HOME`, and warns when
  `HF_HOME` is unset because on this machine that means weights land on C:.

**CLI** (`cli.py`)
- `doctor` — free space on C: and D: by name, Python and venv, cache variables,
  every derived path checked for being on C:, external tools, provider
  availability, database. Ends with an explicit "what works" panel and exits 1
  only on a real problem.
- `run`, `status`, `status JOB_ID`, `resume`, `open`, `cache clear`.
- `serve`, `worker`, `watch`, `rerun`, `collect` exist and exit 3 saying which
  phase will build them.

**Utilities**
- `utils/disk.py` — the guard names the drive, the free space, the requirement
  and the stage that refused to run.
- `utils/files.py` — filename sanitisation (traversal, control characters,
  Windows reserved device names, trailing dots), `resolve_within`, atomic
  writes via a temp file, `.bak` before overwrite, `pad_width`.
- `utils/hashing.py` — `sha256`, and `cache_key` that is stable under dict
  ordering.
- `utils/timefmt.py` — folder stamps, SRT stamps, `--download-sections` stamps.

**Tests** — 173, all passing
- `test_segmentation.py` — bounds, contiguity, no cut inside a word, sentence
  ends preferred over the target, silence boundaries used when available, word
  boundaries when nothing better exists, awkward remainders, and the
  **filename-ordering** tests at 300/999/1500 segments.
- `test_transcription.py` — monotonic word timings, silent audio, unreadable
  audio, determinism, sidecar replay, **SRT structural validity** (numbering,
  ordering, overlap, comma separator), transcript byte-identical to the
  provider's output.
- `test_jobs.py` — id reuse, malformed rejection, single claim, checkpoint
  semantics, watcher ledger including case-insensitivity.
- `test_disk_and_files.py` — the guard names the drive; 18 hostile filenames;
  path traversal; atomic writes; `.bak`; cache keys.
- `test_pipeline_run.py` — the P1 acceptance criterion end to end, plus
  **resume after a kill reproduces byte-identical output**, no re-run of
  checkpointed work, and failure handling.
- `test_config.py` — §16 defaults, YAML merge, `.env` allowlist, and a test
  that the shipped `.env` points nothing at C: and contains no secrets.
- `test_imports.py` — every module imports cleanly, with `SyntaxWarning`
  promoted to an error.

**Fixtures**
- `sample.wav` (60s), `silence.wav` (3s), `awkward_11s.wav`, all generated with
  ffmpeg. Synthetic; a real narration is still owed (see PROGRESS.md).

### Environment

- Installed uv 0.12.17 to `D:\dev\bin` by direct download, because the official
  installer stages through C: and C: had 140 MB free.
- Installed CPython 3.12.14 to `D:\dev\.pythons`.
- Set `UV_CACHE_DIR`, `UV_PYTHON_INSTALL_DIR`, `HF_HOME`, `TORCH_HOME`,
  `PIP_CACHE_DIR` to `D:\dev\.cache\*` at user scope; prepended `D:\dev\bin` to
  the user PATH.
- Pointed pytest's `basetemp` at `.tmp/pytest`; the default is under the system
  temp directory on C:.

### Fixed

- Invalid escape sequence in `utils/disk.py`'s docstring (`D:\dev\x` unescaped)
  made the module unimportable. It hid because the module is imported lazily;
  `test_imports.py` now walks the whole package so this class of bug cannot
  hide again.
- `status` cropped the 12-character job id, which is the one string a user
  needs to copy into `resume`. Columns now have explicit widths.

### Decisions worth recording

- **`project.py` is not in §5's module list.** Added anyway: every stage needs
  the §6 layout, and duplicating those joins is how `selected/` grows orphans.
- **`pad_width` widens past 999 segments.** §6 says `{segment:03d}`; the stated
  intent is that lexicographic order equals chronological order, which three
  digits stops delivering at 1000. Identical output for every realistic project.
- **`.env` for cache paths.** §3 says caches come from env vars and no cache
  path may be hardcoded. Both hold: code only reads the environment, and the
  machine-specific values sit in an editable file outside the source.
- **No new dependencies beyond §4.** The `.env` parser is fifteen lines rather
  than a python-dotenv dependency (§2.10).
