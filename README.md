# VisualResearcher

Takes a cleaned narration `.wav` for a long-form video essay and produces
**researched, downloaded, chronologically-ordered visual assets** — images and
trimmed video clips — that you drag into CapCut by hand.

It is not an editor integration. There is no CapCut code, no edit plan, no
timeline format, no rendering. The deliverable is a folder of ordered files.

> **Status: all eight phases complete.** Every pipeline stage, every command
> and the unattended folder-watcher path are built and working. Dropping a
> `.wav` into the watched folder produces a finished project and a
> notification with no manual steps.
>
> **Most numbers in these docs come from `VR_OFFLINE=1`.** As of 2026-09-20
> the pipeline has also run end to end against the live internet — OpenAI,
> ddgs, Wikimedia Commons, yt-dlp and faster-whisper, on a real 12m45s
> narration — and the live-run section of `PROGRESS.md` says what that
> produced. Treat the offline numbers as the shape of a run and the live ones
> as its cost. `PROGRESS.md` has a table of exactly what is and is not
> verified — read it before trusting a number.

---

## Requirements

- Windows, with **ffmpeg** and **git** on PATH (already installed here).
- **uv** at `D:\dev\bin\uv.exe`, managing Python 3.12 in `.venv`.
- Everything lives on **D:**. C: is critically low on space and nothing may
  touch it — caches included.

If a fresh shell cannot find `uv`, the user-scope PATH entry has not been
picked up; open a new terminal.

---

## Quick start

```powershell
cd D:\dev\visualresearcher

# 1. Check the machine. Run this first whenever anything is odd.
uv run visualresearch doctor

# 2. Run the tests.
uv run pytest -q

# 3. Run the pipeline with no credentials and no network.
$env:VR_OFFLINE = '1'
uv run visualresearch run tests/fixtures/sample.wav --project scratch

# 4. See what happened.
uv run visualresearch status
uv run visualresearch open scratch
```

`VR_OFFLINE=1` forces every provider to its fixture-backed fake. The whole
implemented pipeline runs that way, which is how the test suite runs too.

---

## What you get

```
projects/<name>/
  input/
    narration.wav        the narration, copied (your original is never moved)
    transcript.json      full transcript with word-level timings
    transcript.txt       plain text
    narration.srt        subtitles
  project_context.json   subject, franchise, cast, entity corrections + reasons
  entity_overrides.yaml  yours to edit; always wins, permanently
  timeline.csv           one row per segment
  sources.csv            THE CREDITS LIST — one row per file in selected/
  shotlist.md            what goes on screen, and every gap with its reason
  research_report.md     what the run did and how much to trust it
  segments/
    001_00m00s-00m07s/
      segment.json       timings, narration, entities, queries, picks
      images/            ~30 validated candidates + manifest.json
      clips/             trimmed mp4s, only the span that was needed
      youtube/results.json
      contact_sheet.jpg  the segment's top 8 at a glance
    ...
  selected/              THE DELIVERABLE — flat, ordered, drag into CapCut
  job.log
```

A real `selected/` from the sample fixture:

```
001_1_when_the_sith_warrior_first_walks_into.jpg
002_1_emperor_wrapped_in_that_scene.mp4
002_2_wrapped_in_that_heavy_mask_he_never.jpg
003_1_korriban_darth_vowrawn_watches_full.mp4
...
```

Flat and zero-padded, so a plain name sort is chronological order. If nothing
confident was found for a segment there is **no file** — the gap is listed in
`shotlist.md` with its reason. A placeholder would be worse than an empty
slot.

---

## Commands

| Command | What it does | Status |
|---|---|---|
| `visualresearch doctor` | Free space on C: and D:, Python, caches, tools, providers | works |
| `visualresearch run FILE [--project N] [--no-clips] [--force] [--batch]` | Run the pipeline | works |
| `visualresearch all-candidates PROJECT` | Flat, time-ordered dump of every ranked candidate | works |
| `visualresearch status [JOB_ID]` | List jobs, or show one job's stages | works |
| `visualresearch resume JOB_ID` | Pick up from the first unfinished stage | works |
| `visualresearch open PROJECT` | Open the project folder | works |
| `visualresearch cache clear` | Empty the provider cache | works |
| `visualresearch collect PROJECT` | Rebuild `selected/` from the current picks | works |
| `visualresearch serve [--port 8765]` | Review web UI | works |
| `visualresearch rerun PROJECT --segment N [--stage S]` | Re-run one segment without re-running the project | works |
| `visualresearch worker` | Run queued jobs, `worker.max_concurrent` at a time | works |
| `visualresearch watch` | Watch the drop folder; run the jobs too unless `--no-worker` | works |

Every command in the project constitution's §17 is implemented.


### How a run delivers its output

Segments are processed **one at a time, in narration order**, and each one is
published to `selected/` and `sources.csv` the moment it is finished — so the
folder fills front-to-back while the run is still going, and `visualresearch
status` shows `N of 95` in its **Delivered** column.

`--batch` runs the old order instead: every stage across every segment before
the next stage starts, with nothing in `selected/` until the end. The two
produce identical output; a test asserts the file listings and the CSV rows
match exactly. Pipelining reorders the work, it does not reduce it — the same
number of API calls are made either way.


---

## Reviewing

```powershell
uv run visualresearch serve
```

Then open http://127.0.0.1:8765. The jobs page lists every run; a project page
shows chronological cards with a gap count at the top and a jump-to-next-gap
button that cycles through them.

Each card carries the timecode, narration, interpretation, editable entities,
the queries, an image grid, the YouTube results with copyable timestamps, and
an inline clip preview.

**There is no save button.** USE THIS, reject and favourite rewrite
`selected/` immediately — adding and removing exactly the right files, leaving
no orphans. The folder on disk always matches what the page shows.

It binds to 127.0.0.1 and has no authentication, because it serves the
contents of your project folder.

---

## Unattended runs

VoiceCleaner drops a `.wav` into a folder; a finished project comes out the
other end with no manual step in between.

```powershell
uv run visualresearch watch              # watch + run jobs here
uv run visualresearch watch --no-worker  # only enqueue; a separate worker runs them
uv run visualresearch worker             # run queued jobs
uv run visualresearch serve --watch      # the UI, the watcher and a worker in one process
```

The watcher reacts only to `.wav` files whose name starts with
`watch.filename_prefix` (case-insensitively), and **waits until the file stops
growing** before touching it — VoiceCleaner may still be writing, and half a
narration transcribes into a plausible-looking wrong result rather than an
obvious failure.

Each accepted file is moved into a new project and enqueued on the same queue
the CLI uses, so a watched job and a hand-started job take an identical path.
The same filename is never processed twice, and that survives a restart.

When a job finishes you get a desktop notification with the path to
`selected/`.

### Running several jobs at once

`worker.max_concurrent` jobs run in parallel (default 2), but three resources
are shared rather than per-job:

- **CPU-heavy stages** (transcription, CLIP ranking) take turns through one
  global slot. Without a GPU they do not parallelise, so running two at once
  just makes both slower.
- **The download budget and rate limit** are shared. Three jobs each politely
  rate-limiting themselves still triple the request rate a remote host sees,
  and it is the host's view that gets you blocked.
- **The disk guard sums across jobs.** Two jobs that each individually fit can
  still fill the drive together, and a per-job check passes both times on the
  way to doing exactly that.

---

## How segmentation works

§9 of the project constitution asks for four things at once: every segment
6–10s aiming at 8s, boundaries at semantic joints, never splitting mid-phrase
to hit the target, and output that is contiguous and gapless.

Those constraints conflict under a greedy left-to-right chop — taking the best
boundary available now regularly strands the next window with nothing but
mid-phrase options. So segmentation is a dynamic program over word indices
that minimises

```
boundary_penalty + 0.35 * ((duration - 8) / 8)²
```

where a sentence end costs 0.0, a clause end 0.35, a pause of 250ms or more
0.70, and a mid-phrase cut 1.20. The worst legal duration error costs about
0.09, so boundary quality outweighs duration by roughly thirteen to one and
the optimiser will travel several seconds off target to land on a sentence
end. That ordering is the rule, expressed as arithmetic instead of an `if`.

Cuts are placed at the midpoint of the silence between two words, so segments
are contiguous by construction rather than by a later fix-up pass.

---

## Configuration

`config.yaml` at the repo root. Every key has a default in
`src/visualresearcher/config.py`, so the file only needs what you want to
change.

```yaml
segmentation: {target_s: 8.0, min_s: 6.0, max_s: 10.0}
output:       {root: "D:/dev/visualresearcher/projects", min_free_gb: 5}
confidence:   {high: 0.85, medium: 0.60}
```

A relative `output.root` resolves against the repo, never the shell's working
directory — a relative path must not be able to land on C:.

### Cache locations

`.env` supplies `HF_HOME`, `TORCH_HOME`, `UV_CACHE_DIR` and `PIP_CACHE_DIR`
for variables that are not already set. Only those keys are read, and a value
already in the environment always wins. This exists because a single unguarded
model download would fill what is left of C:.

Credentials do **not** belong in `.env`. Put them in `.env.local`, which is
gitignored and which this loader ignores.

---

## Development

```powershell
uv run pytest -q
uv run ruff check .
uv run ruff format .
```

Tests always run offline, and their temp files go to `.tmp/pytest` on D:
rather than the system temp directory on C:. The current count is in
`PROGRESS.md`; a number in here would go stale every phase.

The suite includes several full end-to-end pipeline runs and a real worker
pool, so it takes several minutes and `pytest -q`'s percentage counter only
advances every 72 tests — a long pause is usually a slow test, not a hang. If
any single test exceeds five minutes, `--faulthandler-timeout` dumps every
thread's stack so you can see exactly where it is stuck.

To watch a run live:

```powershell
uv run pytest -q > .tmp\suite.log 2>&1
Get-Content .tmp\suite.log -Wait     # in another terminal
```

`pytest tests/test_segmentation.py` and friends are the fast way to check one
area while working.

### Fixtures

`tests/fixtures/*.wav` are synthetic, generated with ffmpeg:

```powershell
ffmpeg -y -f lavfi -i "aevalsrc=0.35*sin(2*PI*165*t)*(0.55+0.45*sin(2*PI*1.9*t))*(0.6+0.4*sin(2*PI*0.31*t)):s=16000:d=60" -ac 1 -ar 16000 -sample_fmt s16 tests/fixtures/sample.wav
ffmpeg -y -f lavfi -i "anullsrc=r=16000:cl=mono" -t 3 -sample_fmt s16 tests/fixtures/silence.wav
```

They exercise duration handling and the whole offline path, but they are tones
rather than speech, so the tests still use them: they are deterministic and
they run in milliseconds.

`tests/fixtures/real_sample.wav` is a real 12m45s narration (44.1 kHz stereo,
135 MB, not committed-friendly and not used by the test suite). It is there
for live runs, which is where faster-whisper's real word timings and genuine
prosody actually get exercised.

### Providers

Every provider kind has a fixture-backed fake, which is what `VR_OFFLINE=1`
substitutes. The real ones:

| Kind | Real provider | Needs |
|---|---|---|
| transcription | faster-whisper | installed (the `transcribe` dependency-group) |
| images | ddgs, Wikimedia Commons | nothing — no keys |
| video | yt-dlp | yt-dlp on PATH (it is) |
| embedding | open_clip ViT-B-32 | installed; weights download on first use |
| llm | OpenAI | `OPENAI_API_KEY` in `.env.local` |
| notify | Windows toast | PowerShell |

`doctor` reports each one's state and what you lose without it. Model weights
follow `HF_HOME`, so they land on D:.

The fakes are not stubs. The image fake writes **real JPEGs** and the video
fake **trims real MP4s with ffmpeg**, so an offline run genuinely exercises
the download validation, the perceptual dedupe and the span arithmetic rather
than walking past them.

One thing the offline path cannot do is judge relevance: the fake embedder's
image-to-text similarity is arbitrary by construction. It reports that, and
the ranking stage drops the CLIP weight and redistributes it rather than
scoring 45% of the result on a meaningless number.

---

## Project rules

`CLAUDE.md` is the constitution and wins over anything here. The parts that
bite most often:

- Nothing may touch C:, caches included.
- Pipeline logic never calls an SDK directly — only provider interfaces.
- A failing provider degrades a segment; it never fails the job.
- Never destroy user data. No deletes outside `.cache/` and `.tmp/`.
- `VR_OFFLINE=1` must run everything with zero credentials and zero network.
"# VisualResearcher" 
"# VisualResearcher" 
