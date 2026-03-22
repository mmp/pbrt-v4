@echo off
REM Run from repo root: compare-skipmip.bat "<scene.pbrt>" [<spp>]
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0compare-skipmip.ps1" %*
exit /b %ERRORLEVEL%
