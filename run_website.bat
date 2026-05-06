@echo off
cd /d "%~dp0"
set YOLO_CONFIG_DIR=%CD%
set CLIP_MODELS=ViT-L/14,ViT-B/32
set PYTHONUTF8=1

rem Quality threshold controls. Increase values to make the app stricter.
set QUALITY_MIN_CONFIDENCE=0.40
set QUALITY_THRESHOLD_FRESH=0.40
set QUALITY_THRESHOLD_ADULTERATED=0.40
set QUALITY_THRESHOLD_ROTTEN=0.40

rem Real-fruit gate. Keep false for normal demo mode; set true to reject weak/no-fruit inputs.
set FRUIT_GATE_ENABLED=false
set FRUIT_GATE_MODE=any
set FRUIT_MIN_CONFIDENCE=0.12
set YOLO_MIN_REAL_FRUIT_CONFIDENCE=0.18
set FRUIT_NAMING_ENABLED=true

.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
