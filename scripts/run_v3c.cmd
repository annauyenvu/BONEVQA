@echo off
setlocal
set HF_HOME=D:\BaoYen_work\hf_cache
set TMP=D:\BaoYen_work\tmp
set TEMP=D:\BaoYen_work\tmp
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d C:\Users\Admin\Music\2026\BaoYen
echo [%TIME%] === BAT DAU v3c ===
D:\BaoYen_work\venv\Scripts\python.exe -m src.train --config configs\stage3_vqa_rad_v3c.yaml >> D:\BaoYen_work\logs\v3c.log 2>&1
echo [%TIME%] === KET THUC v3c (exit %ERRORLEVEL%) ===
