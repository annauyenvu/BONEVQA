@echo off
setlocal
set HF_HOME=D:\BaoYen_work\hf_cache
set TMP=D:\BaoYen_work\tmp
set TEMP=D:\BaoYen_work\tmp
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set PY=D:\BaoYen_work\venv\Scripts\python.exe
set CK=D:\BaoYen_work\checkpoints
set CFG=configs\ablation_fracatlas_sach.yaml
set LG=D:\BaoYen_work\logs
set EF=--dataset fracatlas --max_samples 1200 --khong_dung_mask_that --khong_dung_region
cd C:\Users\Admin\Music\2026\BaoYen

echo [%TIME%] CHO ABL5 XONG
:cho
findstr /C:"HOAN TAT ABL5" %LG%\abl5_wrap.log >nul 2>&1
if errorlevel 1 (
  timeout /t 60 /nobreak >nul
  goto cho
)
echo [%TIME%] ABL5 DA XONG, BAT DAU

echo [%TIME%] BO PHAT HIEN GAY XUONG
%PY% scripts\huan_luyen_bo_phat_hien.py --epochs 12 >> %LG%\detector.log 2>&1
echo [%TIME%] XONG BO PHAT HIEN (exit %ERRORLEVEL%)

echo [%TIME%] SEED 1337
%PY% -m src.train --config %CFG% --run_name abl6_du3_s1337 --seed 1337 >> %LG%\seed.log 2>&1
%PY% -m src.train --config %CFG% --run_name abl6_khong_prompt_s1337 --seed 1337 --ablation "visual_prompt,latent_prompt,lens" >> %LG%\seed.log 2>&1
echo [%TIME%] SEED 2024
%PY% -m src.train --config %CFG% --run_name abl6_du3_s2024 --seed 2024 >> %LG%\seed.log 2>&1
%PY% -m src.train --config %CFG% --run_name abl6_khong_prompt_s2024 --seed 2024 --ablation "visual_prompt,latent_prompt,lens" >> %LG%\seed.log 2>&1

echo [%TIME%] DANH GIA SEED
%PY% -m src.evaluate --checkpoint %CK%\abl6_du3_s1337\best.pt --tag abl6_du3_s1337 %EF% >> %LG%\seed_eval.log 2>&1
%PY% -m src.evaluate --checkpoint %CK%\abl6_khong_prompt_s1337_no_visual_prompt_latent_prompt_lens\best.pt --tag abl6_khong_prompt_s1337 %EF% --ablation "visual_prompt,latent_prompt,lens" >> %LG%\seed_eval.log 2>&1
%PY% -m src.evaluate --checkpoint %CK%\abl6_du3_s2024\best.pt --tag abl6_du3_s2024 %EF% >> %LG%\seed_eval.log 2>&1
%PY% -m src.evaluate --checkpoint %CK%\abl6_khong_prompt_s2024_no_visual_prompt_latent_prompt_lens\best.pt --tag abl6_khong_prompt_s2024 %EF% --ablation "visual_prompt,latent_prompt,lens" >> %LG%\seed_eval.log 2>&1
echo [%TIME%] HOAN TAT TIEP THEO
