# VisualResearcher — Project Constitution

Read fully before touching code. If this conflicts with an assumption you were about to
make, this file wins.

## 1. What this is
Takes a cleaned narration `.wav` for a long-form YouTube video essay and produces
**researched, downloaded, chronologically-ordered visual assets** — images and trimmed
video clips — that the user drags into CapCut manually.

**NOT:** an editor integration. No CapCut code, no edit_plan.json, no timeline format, no
rendering, no compositing, no transitions. Do not add these even if they seem natural.

## 2. Operating rules (Claude Code)
1. Inspect before assuming. Every session: list repo, read PROGRESS.md + CHANGELOG.md, run `pytest -q`.
2. Work autonomously. No permission-seeking for ordinary decisions.
3. Ask only when truly blocked — batch into one message.
4. Never claim something works without running it and seeing output.
5. Run `pytest -q` constantly. Actually execute the CLI, not just tests.
6. Maintain PROGRESS.md (phase/done/next/issues/questions) and CHANGELOG.md (dated).
7. Pipeline logic never calls an SDK directly — only provider interfaces. Providers are I/O only.
8. Never destroy user data. No deletes outside `.cache/`/`.tmp/`. `.bak` before overwrite.
9. If a key/API is unavailable, build a fake provider with fixtures and continue.
   `VR_OFFLINE=1` must run the whole pipeline with zero credentials, zero network.
10. Boring code wins. No new dependency without a logged reason.
11. Do not stop after scaffolding.
12. Windows first: `pathlib` everywhere, `encoding="utf-8"` always, no GPU assumed.

## 3. Environment (binding)
- **C: is critically low on space. Nothing may touch C:.** Everything under `D:\dev\visualresearcher`.
- uv-managed Python 3.12 in `.venv`. Never system Python, never a second venv. Use `uv run`.
- CPU-only torch unless user confirms NVIDIA GPU:
  `uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`
- Caches come from `HF_HOME`/`TORCH_HOME` env vars. Never hardcode a cache path.
- **Disk guard:** before any bulk stage, check free space on output drive. Below 5 GB, abort
  that stage with a clear message. `doctor` reports free space on C: and D:.
- ffmpeg and git already installed and on PATH. Never try to install either.

## 4. Stack
Python 3.12, `src/` layout, uv. Typer + Rich. FastAPI + Uvicorn + Jinja2 + HTMX + Tailwind
CDN. Pydantic v2. SQLite via SQLModel. httpx. Pillow, imagehash. open_clip_torch (ViT-B-32,
laion2b_s34b_b79k). faster-whisper (default transcription). Image search: `ddgs` + Wikimedia
Commons (no keys); Brave/SerpAPI optional. Video: `yt-dlp`. LLM: OpenAI behind an interface.
Folder watching: `watchdog`, fall back to 5s polling. pytest, pytest-asyncio, respx, ruff.

**Do not add:** Celery, Redis, Docker, Postgres, React, any build pipeline.

## 5. Layout
```
src/visualresearcher/
  cli.py config.py db.py models.py schemas.py logging_setup.py
  jobs/     queue.py worker.py states.py watcher.py
  pipeline/ ingest.py transcribe.py context.py entities.py segment.py queries.py
            image_search.py download.py dedupe.py rank.py video_search.py
            timestamps.py clips.py collect.py report.py
  providers/ base.py registry.py llm/ transcription/ images/ video/ notify/
  web/      app.py api.py routes_ui.py templates/ static/
  cache/store.py  utils/files.py timefmt.py hashing.py disk.py
domain_packs/ swtor.yaml star_wars.yaml
inbox/ projects/ data/ .cache/ .tmp/ tests/fixtures/
```

## 6. Output structure (this IS the product)
```
projects/<name>/
  input/narration.wav transcript.json transcript.txt narration.srt
  project_context.json timeline.csv shotlist.md sources.csv research_report.md
  segments/031_03m49s-03m57s/
    segment.json images/(+manifest.json) clips/ youtube/results.json contact_sheet.jpg
  selected/                # THE DELIVERABLE — drag into CapCut
    031_1_darth_baras_dark_council.jpg
    031_2_confrontation_clip.mp4
```
`selected/` rules: flat, no subfolders. `{segment:03d}_{pick:d}_{slug}.{ext}` — zero-padded
so lexicographic sort = chronological. Copies, not moves/symlinks; originals stay in
`segments/`. No confident pick → no file, listed as a gap in shotlist.md. Never fabricate a
placeholder.

## 7. Contracts

`project_context.json`:
```json
{"subject":"Star Wars: The Old Republic","franchise":"Star Wars","era":"Old Republic",
 "characters":[],"people":[],"places":[],"events":[],"terminology":["Dark Council"],
 "works":[{"title":"SWTOR Sith Warrior storyline","type":"game"}],
 "entity_corrections":[{"original":"Barriss","resolved":"Darth Baras","confidence":0.93,
   "reason":"Sith Warrior + Dark Council context.","evidence":["segment 12","domain_pack:swtor"]}]}
```

`segment.json`:
```json
{"index":31,"start":229.0,"end":237.0,"duration":8.0,"narration":"...",
 "topic":"Baras unmasked before the Dark Council","entities":["Darth Baras","Dark Council"],
 "event":"...","location":"Korriban","interpretation":"Exact cutscene moment; fall back to portrait + chamber wide.",
 "visual_intent":"EXACT_EVENT",
 "queries":[{"kind":"exact_event","text":"Darth Baras Voice of the Emperor Dark Council"},
            {"kind":"character_setting","text":"Darth Baras Dark Council SWTOR"},
            {"kind":"quest","text":"SWTOR Sith Warrior Retribution Baras"},
            {"kind":"youtube","text":"Darth Baras final confrontation SWTOR cutscene"},
            {"kind":"fallback","text":"SWTOR Dark Council chamber"}],
 "confidence":0.82,"sub_shots":[{"start":229.0,"end":233.0,"topic":"Darth Marr"}],
 "picks":[{"kind":"image","path":"images/031_03.jpg","rank":1}]}
```
`visual_intent` ∈ EXACT_EVENT | CHARACTER | LOCATION | GAME_SCENE | FILM_SCENE | COMIC | ABSTRACT | B_ROLL

Image record: `local_path, image_url, source_page, domain, provider, query, query_kind,
width, height, aspect_ratio, bytes, sha256, phash, dhash, segment_index, rank, score,
score_breakdown, creator, license, license_url, status, notes`

YouTube record: `title, url, video_id, channel, duration, thumbnail, query, reason,
relevance, classification(EXACT_SCENE|LIKELY_EXACT_SCENE|RELATED_FOOTAGE|CONTEXTUAL),
timestamp_candidates:[{start_s,end_s,method,evidence,confidence,url_with_t}], downloaded_clip_path`

## 8. Job lifecycle
`QUEUED, TRANSCRIBING, ANALYZING, SEARCHING_IMAGES, RANKING, SEARCHING_VIDEO,
DOWNLOADING_CLIPS, COLLECTING, REVIEW_REQUIRED, COMPLETE, FAILED`

Stages: ingest → transcribe → context → entities → segment → queries → image_search →
download → dedupe → rank → video_search → timestamps → clips → collect → report.

A failing provider never fails the job — log, mark segment `degraded`, continue. Each stage
checkpoints and is independently re-runnable. `resume JOB_ID` picks up where it died.

## 9. Segmentation
Target 8.0s, bounds 6–10s from config. Semantic boundaries: sentence end > clause end >
word gap ≥250ms, snapped to real silence via word timestamps. **Never split mid-phrase to
hit 8s.** Contiguous and gapless. Folders `031_03m49s-03m57s`, lexicographically sortable.

## 10. Entity resolution (highest-value component)
Seed `domain_packs/swtor.yaml`: Barriss→Darth Baras, Valron→Darth Vowrawn, Sanx→Colonel
Senks, Drog→Lord Draahg.
1. Never blindly replace. Store `original, resolved, confidence, reason`.
2. Transcript on disk keeps ORIGINAL words. Corrections apply only to search queries.
3. Apply at confidence ≥0.75. Below that, search both spellings, let ranking decide.
4. Order: domain pack (exact/fuzzy) → phonetic match (metaphone) → LLM with full transcript context.
5. `projects/<name>/entity_overrides.yaml` is user-edited and always wins, permanently.

## 11. Images
~30 candidates/segment, keep 6–8. Providers behind one interface; one failing never stops a
segment. **Download validation:** 15 MB max, content-type allowlist, Pillow-decodable check,
reject SVG, sanitize filenames, block path traversal, never trust a remote filename.
**Dedupe:** SHA256 exact + pHash/dHash (Hamming ≤6) — catches crops, resizes, recompressions.
Keep best, record discards, never delete silently.
**Rank:** CLIP similarity (prompt = interpretation + entities + event + location) + entity
bonus + exact-event bonus + source tier + resolution + sharpness (Laplacian) + watermark
penalty (border edge-density, no OCR dep) + repetition penalty. Deterministic under fixed seed.
**Diversity:** final 6–8 span different shot types — exact event, close-up, full body, wide
environment, action, secondary character, location, comic panel, game screenshot, object.
Never eight near-identical portraits. Penalize repeats across recent segments.

## 12. Clips
1. Search with `yt-dlp` (`ytsearchN:query`) — no key, no quota.
2. Classify each hit with a stated reason.
3. Locate timestamp, recording which method won: captions → chapters → description →
   metadata → text/entity similarity.
4. **Download only the needed span:**
   `yt-dlp --download-sections "*00:01:42:13-00:01:43:16" --force-keyframes-at-cuts
    -f "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[height<=1080]" -o "<segdir>/clips/%(id)s.%(ext)s" URL`
   Pad ±`clip_padding_s`. Full-download fallback ONLY if source < `max_full_download_minutes`.
5. Limits from config: clips/segment, max project GB, concurrency, rate limiting.
6. Cache by `(video_id, start, end)`. `--no-clips` skips. Write to `.tmp/`, move on success —
   never leave a partial file in `clips/`.
7. Store the clickable `&t=` URL regardless of whether the clip downloaded.

## 13. sources.csv
One row per file in `selected/`: `segment, timecode, file, type, source_url, source_page,
channel_or_creator, timestamp, license, license_url, query`. This is the credits list for
the video description. Every asset gets a row — no silent assets. Mark `license: unknown`
rather than guessing. Respect robots.txt and rate limits; prefer APIs over HTML scraping.

## 14. Confidence
HIGH ≥0.85 → auto-copied to `selected/`. MEDIUM 0.60–0.85 → copied + flagged.
LOW <0.60 → nothing copied, listed as a gap. Job ends REVIEW_REQUIRED, not COMPLETE.

## 15. Web UI
`serve` → http://127.0.0.1:8765. Jobs page: status, stage, timing, disk used. Project page:
chronological cards, filter by confidence, jump-to-next-gap, gap count at top. Segment card:
timecode, narration, interpretation, entities (visible + reversible), queries, image grid,
YouTube results with clickable timestamps, inline clip preview, confidence badge.
Actions: USE THIS, favorite, reject, rerun/edit query, open source, copy timestamp, add note,
mark reviewed. Changing picks **rewrites `selected/` exactly** — adds and removes, no orphans.
HTMX partials, no client framework.

## 16. config.yaml
```yaml
segmentation: {target_s: 8.0, min_s: 6.0, max_s: 10.0}
images: {candidates_per_segment: 30, keep_per_segment: 8, max_bytes: 15728640,
         min_width: 800, phash_distance: 6, providers: [ddgs, wikimedia]}
clips: {enabled: true, max_per_segment: 1, padding_s: 3, max_height: 1080,
        max_full_download_minutes: 15, max_project_gb: 15, concurrency: 2}
ranking: {model: ViT-B-32, pretrained: laion2b_s34b_b79k, device: auto,
          weights: {clip: 0.45, entity: 0.2, source: 0.1, resolution: 0.1,
                    sharpness: 0.1, repetition: -0.15}}
output: {root: "D:/dev/visualresearcher/projects", slug_max_len: 40, write_srt: true, min_free_gb: 5}
confidence: {high: 0.85, medium: 0.60}
watch: {dir: "D:/shared/incoming", filename_prefix: "Tight", stable_secs: 3, poll_interval_s: 5}
worker: {max_concurrent: 2}
compute: {max_concurrent_cpu_heavy: 1}
```

## 17. CLI
```
visualresearch run narration.wav [--project NAME] [--no-clips] [--force]
visualresearch serve [--port 8765]     visualresearch worker    visualresearch watch
visualresearch status [JOB_ID]         visualresearch resume JOB_ID
visualresearch rerun PROJECT --segment 31 [--stage image_search|video_search|clips]
visualresearch collect PROJECT         visualresearch open PROJECT
visualresearch doctor                  visualresearch cache clear
```
Build `doctor` early in Phase 1 — it must state plainly what's missing and what still works.

## 18. Folder watcher (the automation trigger)
VoiceCleaner triggers this by dropping a file. No HTTP call, no JSON from its side.
1. Watch `watch.dir` with `watchdog`; fall back to polling `poll_interval_s`.
2. React only to `.wav` files starting with `watch.filename_prefix` (case-insensitive).
3. **Stability guard:** poll size every 1s; proceed only after unchanged for `stable_secs` —
   VoiceCleaner may still be writing.
4. `project_name` = slugified filename stem, with timestamp suffix on collision.
5. Create job, MOVE the file to `projects/<name>/input/narration.wav`, enqueue on the same
   queue the CLI uses — identical downstream path, no special-casing.
6. **Never process the same filename twice** — track in the job DB, survives restarts.
7. Runs inside `serve`; also standalone as `watch` for testing.
8. On any watcher error: log, leave the source file untouched, keep watching. Never crash
   the loop over one bad file.

## 19. Concurrency
`worker.max_concurrent` jobs at once (default 2). CPU-heavy stages (transcription, CLIP
ranking) obey a separate global semaphore `compute.max_concurrent_cpu_heavy` (default 1) —
they don't parallelize without a GPU. Network stages share a TOTAL download budget across
jobs, not per-job, so 3 jobs don't triple rate-limit exposure and get you blocked. Disk
guard checks the SUM of active jobs' projected usage. `status` shows all jobs.

## 20. Caching and resume
SQLite cache keyed `sha256(provider+endpoint+normalized_params)`: searches, transcripts,
entity resolutions, LLM outputs, image hashes, CLIP embeddings, YouTube metadata, clip
spans. TTL 30 days. Stages skip if artifact+checkpoint exist unless `--force`. Re-running
one segment must not re-run the project. **Batch LLM calls ~20 segments per call** — a
300-segment video must not cost a fortune.

## 21. Tests (real tests that can fail and catch bugs)
Jobs: id reuse doesn't duplicate; malformed job rejected.
Transcription: word timestamps monotonic; silent audio doesn't crash; SRT valid.
Entities: 4 known pairs resolve; <0.75 NOT applied; overrides always win; transcript never overwritten.
Segmentation: all 6–10s; no mid-phrase split; contiguous; **300+ segments sort chronologically as strings**.
Images: ≥15 candidates/segment; oversize/wrong-type/undecodable rejected; filename sanitized;
path traversal blocked; dedupe catches exact/resized/cropped/recompressed; ranking
deterministic; diversity enforced.
Clips: `--download-sections` range matches located timestamp; full-download guard refuses
long sources; no partial file in `clips/`; cache prevents re-download.
Collect: **changing a pick adds/removes exactly the right file in `selected/`, zero orphans**;
no-pick segment → no file + gap listed; sources.csv row count == selected/ file count.
Disk/resume: guard aborts below threshold naming the drive; kill mid-stage → resume gives
identical output, no duplicated work; cache TTL correct.
Watcher: prefix matched case-insensitively; non-matching ignored; waits for stability; never
double-processes; bad file skipped without killing the loop.
Concurrency: 2 simultaneous jobs don't write into each other's folders; CPU-heavy stages
serialize; disk guard sums across jobs.

Fixtures: short real narration WAV + recorded provider responses. `VR_OFFLINE=1` runs it all
with no network, no keys.

## 22. Phases
**P1 Foundation** — skeleton, config, SQLite, queue+worker, CLI, `doctor`, disk guard, ingest,
transcription (faster-whisper + fake), naive segmentation, SRT.
*Done:* `run tests/fixtures/sample.wav` writes transcript, SRT, correctly named folders.

**P2 Understanding** — context, entity resolution, domain packs, overrides, query gen, semantic segmentation.
*Done:* SWTOR fixture resolves all 4 known errors with reasons; ≥4 queries/segment of differing kinds.

**P3 Images** — providers, downloader, manifests, dedupe.
*Done:* ≥15 validated deduplicated candidates/segment with full metadata.

**P4 Ranking** — CLIP, heuristics, diversity, contact sheets.
*Done:* top-8 visually varied and correct, deterministic.

**P5 Clips** — yt-dlp search, classification, timestamp location, sectioned download, guards, cache.
*Done:* a known cutscene segment yields EXACT_SCENE + timestamp + a correctly trimmed local mp4.

**P6 Review UI** — everything in §15.
*Done:* a 100-segment project fully reviewable without the CLI.

**P7 Collection** — `selected/`, sources.csv, shotlist.md, research_report.md.
*Done:* dragging `selected/` into CapCut lands everything in correct order. **MVP-1 — stop and report.**

**P8 Automation** — watcher (§18), concurrency (§19), notifications, full run on a real 20+ min
narration, cost/timing report, README rewrite.
*Done:* dropping `Tight_anything.wav` into `D:/shared/incoming` produces a finished project
and a desktop notification, zero manual steps.

## 23. Done means
Code written · tests written and passing · real command run and output seen · errors fixed
not worked around · PROGRESS.md + CHANGELOG.md updated · no secrets, no hardcoded paths,
nothing on C:, nothing destructive. If you can't tick all of these, say so plainly.