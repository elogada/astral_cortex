@echo off
echo ==================================================
echo   Configuring Airgap Mode
echo ==================================================
echo.

:: Require admin privileges
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo [!] Please run this script as Administrator.
    pause
    exit /b
)

:: Set system-wide environment variables
setx HF_HUB_OFFLINE "1" /M
setx TRANSFORMERS_OFFLINE "1" /M
setx HF_DATASETS_OFFLINE "1" /M
setx HF_HUB_DISABLE_TELEMETRY "1" /M

echo.
echo [✓] Airgap mode variables set successfully.
echo.
echo You may need to restart your computer for them to take effect.
pause
