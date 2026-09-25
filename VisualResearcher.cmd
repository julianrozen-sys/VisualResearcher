@echo off
setlocal enabledelayedexpansion
title VisualResearcher
cd /d "%~dp0"

set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo.
  echo   Cannot find the Python environment at:
  echo     %PY%
  echo.
  echo   Run this once in the project folder to create it:
  echo     uv sync
  echo.
  pause
  exit /b 1
)

rem ---------------------------------------------------------------------------
rem Keep the machine awake. A 3-hour run died once because Windows hit its
rem 1-hour idle sleep timer -- the idle clock counts user input, not work, so a
rem background job does nothing to hold it off.
rem ---------------------------------------------------------------------------
powercfg -change standby-timeout-ac 0 >nul 2>&1

if "%~1"=="" goto :nothing_dropped

echo.
echo   VisualResearcher
echo   ----------------
echo   Leave this window open until it finishes. Closing it stops the run.
echo.

set /a COUNT=0
for %%F in (%*) do set /a COUNT+=1
echo   %COUNT% file(s) to process.
echo.

:next_file
if "%~1"=="" goto :all_done
echo ===========================================================
echo   %~nx1
echo ===========================================================
"%PY%" -m visualresearcher.cli run "%~1"
if errorlevel 1 (
  echo.
  echo   ^>^> FAILED on %~nx1  ^(see the job.log in its project folder^)
  echo.
) else (
  echo.
  echo   ^>^> done: %~nx1
  echo.
)
shift
goto :next_file

:all_done
echo.
echo   All files processed. Opening the projects folder.
start "" "%~dp0projects"
echo.
pause
exit /b 0

:nothing_dropped
echo.
echo   VisualResearcher
echo   ================
echo.
echo   Nothing was dropped on me, so here is how to use me:
echo.
echo     1. Drag one or more audio files onto this file
echo        ^(or onto its Desktop shortcut^) and let go.
echo.
echo     2. Or put files in the drop folder and leave the
echo        watcher running:  "Watch Folder.cmd"
echo.
echo   Audio formats: wav mp3 m4a flac ogg opus aac
echo.
echo   Output lands in:  %~dp0projects\^<name^>\
echo     selected\          the pick for each segment
echo     all_candidates\    every ranked option, in time order
echo     sources.csv        credits for every file in selected\
echo     shotlist.md        segments where nothing good was found
echo.
pause
exit /b 0
