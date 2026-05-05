@echo off
cd /d "%~dp0"
set YOLO_CONFIG_DIR=%CD%
set RUNTIME_MODE=minimal
set YOLO_FALLBACK=yolov8n.pt
set PYTHONUTF8=1
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
