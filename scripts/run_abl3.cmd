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
set LG=D:\BaoYen_work\logs\abl3
set EF=--dataset fracatlas --max_samples 1200 --khong_dung_mask_that --khong_dung_region
cd C:\Users\Admin\Music\2026\BaoYen

echo [%TIME%] TRAIN abl3_du3
%PY% -m src.train --config %CFG% --run_name abl3_du3 >> %LG%_train.log 2>&1
echo [%TIME%] TRAIN abl3_khong_visual
%PY% -m src.train --config %CFG% --run_name abl3_khong_visual --ablation visual_prompt >> %LG%_train.log 2>&1
echo [%TIME%] TRAIN abl3_khong_lens
%PY% -m src.train --config %CFG% --run_name abl3_khong_lens --ablation lens >> %LG%_train.log 2>&1
echo [%TIME%] TRAIN abl3_khong_latent
%PY% -m src.train --config %CFG% --run_name abl3_khong_latent --ablation latent_prompt >> %LG%_train.log 2>&1
echo [%TIME%] TRAIN abl3_khong_prompt
%PY% -m src.train --config %CFG% --run_name abl3_khong_prompt --ablation "visual_prompt,latent_prompt,lens" >> %LG%_train.log 2>&1

echo [%TIME%] EVAL
%PY% -m src.evaluate --checkpoint %CK%\abl3_du3\best.pt --tag abl3_du3 %EF% >> %LG%_eval.log 2>&1
%PY% -m src.evaluate --checkpoint %CK%\abl3_khong_visual_no_visual_prompt\best.pt --tag abl3_khong_visual %EF% --ablation visual_prompt >> %LG%_eval.log 2>&1
%PY% -m src.evaluate --checkpoint %CK%\abl3_khong_lens_no_lens\best.pt --tag abl3_khong_lens %EF% --ablation lens >> %LG%_eval.log 2>&1
%PY% -m src.evaluate --checkpoint %CK%\abl3_khong_latent_no_latent_prompt\best.pt --tag abl3_khong_latent %EF% --ablation latent_prompt >> %LG%_eval.log 2>&1
%PY% -m src.evaluate --checkpoint %CK%\abl3_khong_prompt_no_visual_prompt_latent_prompt_lens\best.pt --tag abl3_khong_prompt %EF% --ablation "visual_prompt,latent_prompt,lens" >> %LG%_eval.log 2>&1
echo [%TIME%] HOAN TAT ABL3
