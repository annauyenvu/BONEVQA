@echo off
setlocal
set HF_HOME=D:\BaoYen_work\hf_cache
set TMP=D:\BaoYen_work\tmp
set TEMP=D:\BaoYen_work\tmp
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set PY=D:\BaoYen_work\venv\Scripts\python.exe
set CK=D:\BaoYen_work\checkpoints
set LG=D:\BaoYen_work\logs
cd C:\Users\Admin\Music\2026\BaoYen

echo [%TIME%] CHO ABL8 XONG
:cho
findstr /C:"HOAN TAT ABL8" %LG%\abl8_wrap.log >nul 2>&1
if errorlevel 1 (
  timeout /t 120 /nobreak >nul
  goto cho
)

echo [%DATE% %TIME%] GIAI DOAN 2 QUY MO LON
%PY% -m src.train --config configs\stage2_pmcvqa_lon.yaml >> %LG%\stage2lon.log 2>&1
echo [%DATE% %TIME%] XONG GIAI DOAN 2 (exit %ERRORLEVEL%)

echo [%DATE% %TIME%] GIAI DOAN 3 FRACATLAS TU DONG
%PY% -m src.train --config configs\stage3_fracatlas_lon_tudong.yaml >> %LG%\stage3lon.log 2>&1
echo [%DATE% %TIME%] GIAI DOAN 3 FRACATLAS HO TRO
%PY% -m src.train --config configs\stage3_fracatlas_lon_hotro.yaml >> %LG%\stage3lon.log 2>&1
echo [%DATE% %TIME%] GIAI DOAN 3 VQA-RAD
%PY% -m src.train --config configs\stage3_vqa_rad_lon.yaml >> %LG%\stage3lon.log 2>&1

echo [%DATE% %TIME%] DANH GIA
%PY% -m src.evaluate --checkpoint %CK%\stage3_fracatlas_lon_tudong\best.pt --dataset fracatlas --tag lon_tu_dong --max_samples 1200 --khong_dung_mask_that --khong_dung_region >> %LG%\lon_eval.log 2>&1
%PY% -m src.evaluate --checkpoint %CK%\stage3_fracatlas_lon_hotro\best.pt --dataset fracatlas --tag lon_ho_tro --max_samples 1200 --che_do_ho_tro --khong_dung_region >> %LG%\lon_eval.log 2>&1
%PY% -m src.evaluate --checkpoint %CK%\stage3_vqa_rad_lon\best.pt --dataset vqa_rad --tag lon_vqa_rad --khong_dung_region >> %LG%\lon_eval.log 2>&1
%PY% -m src.evaluate --checkpoint %CK%\stage3_vqa_rad_lon\best.pt --dataset vqa_rad --tag lon_vqa_rad_vote --khong_dung_region --closed_llm_weight 0.3 >> %LG%\lon_eval.log 2>&1
echo [%DATE% %TIME%] HOAN TAT QUY MO LON
