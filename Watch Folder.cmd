@echo off
setlocal
title VisualResearcher - watching drop folder
cd /d "%~dp0"

set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo   Cannot find %PY%
  echo   Run "uv sync" in this folder first.
  pause
  exit /b 1
)

if not exist "%~dp0inbox" mkdir "%~dp0inbox"

rem A watched folder is a long-lived job by definition, so the same idle-sleep
rem trap applies here as in the drag-and-drop path.
powercfg -change standby-timeout-ac 0 >nul 2>&1

echo.
echo   VisualResearcher - drop folder
echo   ==============================
echo.
echo   Drop audio files into the folder that just opened.
echo   Each one becomes its own project automatically.
echo.
echo   Leave this window open. Closing it stops the watcher.
echo   Ctrl-C to stop.
echo.

start "" "%~dp0inbox"
"%PY%" -m visualresearcher.cli watch

echo.
echo   Watcher stopped.
pause
