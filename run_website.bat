@echo off
cd /d "%~dp0"
set RUNTIME_MODE=hf_fruit_classifier
set PYTHONUTF8=1
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
