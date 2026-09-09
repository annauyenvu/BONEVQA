@echo off
setlocal
set HF_HOME=D:\BaoYen_work\hf_cache
set TMP=D:\BaoYen_work\tmp
set TEMP=D:\BaoYen_work\tmp
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set PY=D:\BaoYen_work\venv\Scripts\python.exe
set LOG=D:\BaoYen_work\logs
set CK=D:\BaoYen_work\checkpoints
cd /d C:\Users\Admin\Music\2026\BaoYen

call :step t1_abl_khong_latent -m src.train --config configs\ablation_fracatlas.yaml --run_name abl_khong_latent --ablation latent_prompt
call :step t1_abl_khong_lens -m src.train --config configs\ablation_fracatlas.yaml --run_name abl_khong_lens --ablation lens
call :step t1_abl_khong_prompt -m src.train --config configs\ablation_fracatlas.yaml --run_name abl_khong_prompt --ablation "visual_prompt,latent_prompt,lens"

call :step t2_eval_fracatlas -m src.evaluate --checkpoint %CK%\stage3_fracatlas_full\best.pt --dataset fracatlas --tag full --max_samples 1200
call :step t2_eval_vqa_rad -m src.evaluate --checkpoint %CK%\stage3_vqa_rad_full\best.pt --dataset vqa_rad --tag full
call :step t2_eval_abl_du3 -m src.evaluate --checkpoint %CK%\abl_du3\best.pt --dataset fracatlas --tag abl_du3 --max_samples 1200
call :step t2_eval_abl_khong_visual -m src.evaluate --checkpoint %CK%\abl_khong_visual_no_visual_prompt\best.pt --dataset fracatlas --tag abl_khong_visual --max_samples 1200 --ablation visual_prompt
call :step t2_eval_abl_khong_latent -m src.evaluate --checkpoint %CK%\abl_khong_latent_no_latent_prompt\best.pt --dataset fracatlas --tag abl_khong_latent --max_samples 1200 --ablation latent_prompt
call :step t2_eval_abl_khong_lens -m src.evaluate --checkpoint %CK%\abl_khong_lens_no_lens\best.pt --dataset fracatlas --tag abl_khong_lens --max_samples 1200 --ablation lens
call :step t2_eval_abl_khong_prompt -m src.evaluate --checkpoint %CK%\abl_khong_prompt_no_visual_prompt_latent_prompt_lens\best.pt --dataset fracatlas --tag abl_khong_prompt --max_samples 1200 --ablation "visual_prompt,latent_prompt,lens"

copy /Y %CK%\stage3_fracatlas_full\best.pt %CK%\best.pt >nul
echo [%TIME%] === HOAN TAT PHAN CON LAI ===
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
