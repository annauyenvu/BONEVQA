@echo off
setlocal
set HF_HOME=D:\BaoYen_work\hf_cache
set TMP=D:\BaoYen_work\tmp
set TEMP=D:\BaoYen_work\tmp
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set PY=D:\BaoYen_work\venv\Scripts\python.exe
set LOG=D:\BaoYen_work\logs
cd /d C:\Users\Admin\Music\2026\BaoYen

call :step v3a -m src.train --config configs\stage3_vqa_rad_v3a.yaml
call :step v3b -m src.train --config configs\stage3_vqa_rad_v3b.yaml
echo [%TIME%] === HOAN TAT V3 ===
exit /b 0

:step
set NAME=%1
shift
set ARGS=
:collect
if [%1]==[] goto run
set ARGS=%ARGS% %1
shift
goto collect
:run
echo [%TIME%] === BAT DAU %NAME% ===
%PY% %ARGS% >> %LOG%\%NAME%.log 2>&1
echo [%TIME%] === KET THUC %NAME% (exit %ERRORLEVEL%) ===
exit /b 0
