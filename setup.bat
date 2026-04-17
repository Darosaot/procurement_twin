@echo off
:: ─────────────────────────────────────────────────────────────────
::  Procurement Digital Twin — Local Setup (Windows)
:: ─────────────────────────────────────────────────────────────────
::  Run once:  setup.bat
::  Then:      python run.py
:: ─────────────────────────────────────────────────────────────────

echo.
echo ========================================================
echo   Procurement Digital Twin — Local Setup
echo ========================================================
echo.

:: ── 1. Check Python ──────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.10 or later.
    echo         https://www.python.org/downloads/
    pause
    exit /b 1
)
for /f "tokens=2" %%v in ('python --version 2^>^&1') do echo [OK] Python %%v found.

:: ── 2. Create virtual environment ────────────────────────────────
if not exist ".venv" (
    echo.
    echo Creating virtual environment (.venv^)...
    python -m venv .venv
    echo [OK] Virtual environment created.
) else (
    echo [OK] Virtual environment already exists.
)

:: ── 3. Activate and install ───────────────────────────────────────
echo.
echo Installing dependencies (this takes ~2 minutes on first run^)...
call .venv\Scripts\activate.bat
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt
echo [OK] All dependencies installed.

:: ── 4. Done ──────────────────────────────────────────────────────
echo.
echo ========================================================
echo   Setup complete!
echo ========================================================
echo.
echo   To launch the dashboard:
echo     .venv\Scripts\activate.bat
echo     python run.py
echo.
echo   Then open http://localhost:8050 in your browser.
echo.
pause
