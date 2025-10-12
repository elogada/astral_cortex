@echo off
:: Require admin privileges
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo [!] Please run this script as Administrator.
    pause
    exit /b
)
echo Turning OFF Hugging Face offline mode...
setx HF_HUB_OFFLINE "0" /M
setx TRANSFORMERS_OFFLINE "0" /M
setx HF_DATASETS_OFFLINE "0" /M
setx HF_HUB_DISABLE_TELEMETRY "0" /M
pause
