@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  py -3.11 -m venv .venv
  if errorlevel 1 python -m venv .venv
)
if not exist .venv\Scripts\python.exe (
  echo Install Python 3.11 or 3.12 and rerun this file.
  pause
  exit /b 1
)
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 (
  echo CPU AI dependency installation failed. Check your internet connection.
  pause
  exit /b 1
)
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
  echo Package installation failed. Check the internet connection and pip output.
  pause
  exit /b 1
)
.venv\Scripts\python.exe -m streamlit run app.py
pause
