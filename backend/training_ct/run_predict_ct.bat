@echo off
setlocal

REM Usage:
REM run_predict_ct.bat -Checkpoint path\to\best_model.pth -Config path\to\config.yaml -Image path\to\image.nii.gz
REM run_predict_ct.bat -Checkpoint path\to\best_model.pth -Config path\to\config.yaml -DataDir path\to\images\folder

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_predict_ct.ps1" %*

if errorlevel 1 (
  echo.
  echo Prediction failed.
  exit /b 1
)

echo.
echo Prediction finished.
exit /b 0
