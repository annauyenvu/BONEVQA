@echo off
setlocal
set HF_HOME=D:\BaoYen_work\hf_cache
set TMP=D:\BaoYen_work\tmp
set TEMP=D:\BaoYen_work\tmp
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set PY=D:\BaoYen_work\venv\Scripts\python.exe
set CK=D:\BaoYen_work\checkpoints
set CFG=configs\ablation_fracatlas_detector.yaml
set LG=D:\BaoYen_work\logs
set BOX=D:\BaoYen_work\data\fracatlas\box_bo_phat_hien.json
set EF=--dataset fracatlas --max_samples 1200 --khong_dung_region --box_bo_phat_hien %BOX%
cd C:\Users\Admin\Music\2026\BaoYen

echo [%TIME%] CHO DA SEED XONG
:cho
findstr /C:"HOAN TAT TIEP THEO" %LG%\tieptheo_wrap.log >nul 2>&1
if errorlevel 1 (
  timeout /t 60 /nobreak >nul
  goto cho
)
echo [%TIME%] SINH BOX BO PHAT HIEN
%PY% scripts\sinh_box_bo_phat_hien.py >> %LG%\sinhbox.log 2>&1
echo [%TIME%] XONG SINH BOX (exit %ERRORLEVEL%)

echo [%TIME%] TRAIN abl7_du3
%PY% -m src.train --config %CFG% --run_name abl7_du3 >> %LG%\abl7_train.log 2>&1
echo [%TIME%] TRAIN abl7_khong_visual
%PY% -m src.train --config %CFG% --run_name abl7_khong_visual --ablation visual_prompt >> %LG%\abl7_train.log 2>&1
echo [%TIME%] TRAIN abl7_khong_lens
%PY% -m src.train --config %CFG% --run_name abl7_khong_lens --ablation lens >> %LG%\abl7_train.log 2>&1
echo [%TIME%] TRAIN abl7_khong_latent
%PY% -m src.train --config %CFG% --run_name abl7_khong_latent --ablation latent_prompt >> %LG%\abl7_train.log 2>&1
echo [%TIME%] TRAIN abl7_khong_prompt
%PY% -m src.train --config %CFG% --run_name abl7_khong_prompt --ablation "visual_prompt,latent_prompt,lens" >> %LG%\abl7_train.log 2>&1

echo [%TIME%] EVAL abl7
%PY% -m src.evaluate --checkpoint %CK%\abl7_du3\best.pt --tag abl7_du3 %EF% >> %LG%\abl7_eval.log 2>&1
%PY% -m src.evaluate --checkpoint %CK%\abl7_khong_visual_no_visual_prompt\best.pt --tag abl7_khong_visual %EF% --ablation visual_prompt >> %LG%\abl7_eval.log 2>&1
%PY% -m src.evaluate --checkpoint %CK%\abl7_khong_lens_no_lens\best.pt --tag abl7_khong_lens %EF% --ablation lens >> %LG%\abl7_eval.log 2>&1
%PY% -m src.evaluate --checkpoint %CK%\abl7_khong_latent_no_latent_prompt\best.pt --tag abl7_khong_latent %EF% --ablation latent_prompt >> %LG%\abl7_eval.log 2>&1
%PY% -m src.evaluate --checkpoint %CK%\abl7_khong_prompt_no_visual_prompt_latent_prompt_lens\best.pt --tag abl7_khong_prompt %EF% --ablation "visual_prompt,latent_prompt,lens" >> %LG%\abl7_eval.log 2>&1
echo [%TIME%] HOAN TAT ABL7
