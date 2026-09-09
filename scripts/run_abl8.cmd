@echo off
setlocal
set HF_HOME=D:\BaoYen_work\hf_cache
set TMP=D:\BaoYen_work\tmp
set TEMP=D:\BaoYen_work\tmp
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set PY=D:\BaoYen_work\venv\Scripts\python.exe
set CK=D:\BaoYen_work\checkpoints
set CFG=configs\ablation_fracatlas_hotro.yaml
set LG=D:\BaoYen_work\logs
set EF=--dataset fracatlas --max_samples 1200 --che_do_ho_tro --khong_dung_region
cd C:\Users\Admin\Music\2026\BaoYen

echo [%TIME%] CHO ABL7 XONG
:cho
findstr /C:"HOAN TAT ABL7" %LG%\abl7_wrap.log >nul 2>&1
if errorlevel 1 (
  timeout /t 60 /nobreak >nul
  goto cho
)
echo [%TIME%] SEED HO TRO 1337
%PY% -m src.train --config %CFG% --run_name abl8_du3_s1337 --seed 1337 >> %LG%\abl8.log 2>&1
%PY% -m src.train --config %CFG% --run_name abl8_khong_prompt_s1337 --seed 1337 --ablation "visual_prompt,latent_prompt,lens" >> %LG%\abl8.log 2>&1
echo [%TIME%] SEED HO TRO 2024
%PY% -m src.train --config %CFG% --run_name abl8_du3_s2024 --seed 2024 >> %LG%\abl8.log 2>&1
%PY% -m src.train --config %CFG% --run_name abl8_khong_prompt_s2024 --seed 2024 --ablation "visual_prompt,latent_prompt,lens" >> %LG%\abl8.log 2>&1

echo [%TIME%] EVAL SEED HO TRO
%PY% -m src.evaluate --checkpoint %CK%\abl8_du3_s1337\best.pt --tag abl8_du3_s1337 %EF% >> %LG%\abl8_eval.log 2>&1
%PY% -m src.evaluate --checkpoint %CK%\abl8_khong_prompt_s1337_no_visual_prompt_latent_prompt_lens\best.pt --tag abl8_khong_prompt_s1337 %EF% --ablation "visual_prompt,latent_prompt,lens" >> %LG%\abl8_eval.log 2>&1
%PY% -m src.evaluate --checkpoint %CK%\abl8_du3_s2024\best.pt --tag abl8_du3_s2024 %EF% >> %LG%\abl8_eval.log 2>&1
%PY% -m src.evaluate --checkpoint %CK%\abl8_khong_prompt_s2024_no_visual_prompt_latent_prompt_lens\best.pt --tag abl8_khong_prompt_s2024 %EF% --ablation "visual_prompt,latent_prompt,lens" >> %LG%\abl8_eval.log 2>&1
echo [%TIME%] HOAN TAT ABL8
