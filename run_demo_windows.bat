@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe python -m venv .venv
if not exist .venv\Scripts\python.exe (
  echo Install Python 3.11 or 3.12 first.
  pause
  exit /b 1
)
.venv\Scripts\python.exe -m pip install -r requirements-demo.txt
if errorlevel 1 (
  pause
  exit /b 1
)
.venv\Scripts\python.exe -m streamlit run app.py
pause
