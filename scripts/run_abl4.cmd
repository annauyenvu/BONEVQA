@echo off
setlocal
set HF_HOME=D:\BaoYen_work\hf_cache
set TMP=D:\BaoYen_work\tmp
set TEMP=D:\BaoYen_work\tmp
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set PY=D:\BaoYen_work\venv\Scripts\python.exe
set CK=D:\BaoYen_work\checkpoints
set LG=D:\BaoYen_work\logs\abl4.log
set FLAGS=--dataset fracatlas --max_samples 1200 --khong_dung_mask_that --khong_dung_region
cd C:\Users\Admin\Music\2026\BaoYen
echo [%TIME%] BAT DAU ABL4
%PY% -m src.evaluate --checkpoint %CK%\abl_du3\best.pt --tag abl4_du3 %FLAGS% >> %LG% 2>&1
%PY% -m src.evaluate --checkpoint %CK%\abl_khong_visual_no_visual_prompt\best.pt --tag abl4_khong_visual %FLAGS% --ablation visual_prompt >> %LG% 2>&1
%PY% -m src.evaluate --checkpoint %CK%\abl_khong_lens_no_lens\best.pt --tag abl4_khong_lens %FLAGS% --ablation lens >> %LG% 2>&1
%PY% -m src.evaluate --checkpoint %CK%\abl_khong_latent_no_latent_prompt\best.pt --tag abl4_khong_latent %FLAGS% --ablation latent_prompt >> %LG% 2>&1
%PY% -m src.evaluate --checkpoint %CK%\abl_khong_prompt_no_visual_prompt_latent_prompt_lens\best.pt --tag abl4_khong_prompt %FLAGS% --ablation "visual_prompt,latent_prompt,lens" >> %LG% 2>&1
echo [%TIME%] HOAN TAT ABL4
