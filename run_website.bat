@echo off
cd /d "%~dp0"
set YOLO_CONFIG_DIR=%CD%
set CLIP_MODELS=ViT-L/14,ViT-B/32
set PYTHONUTF8=1
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
