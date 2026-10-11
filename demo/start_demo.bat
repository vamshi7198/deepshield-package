@echo off
REM DeepShield local demo: starts the analysis server (port 8765) and the web interface (port 8080),
REM then opens the browser. Close the two black windows to stop the demo.
cd /d "%~dp0"
set PORT=8765
start "DeepShield server" "%~dp0..\.venv\Scripts\python.exe" app.py
start "DeepShield web" "%~dp0..\.venv\Scripts\python.exe" -m http.server 8080 --directory web-dist
timeout /t 15 /nobreak >nul
start "" http://localhost:8080
