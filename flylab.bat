@echo off
rem flylab: double-click to install (first time only) and run the whole demo.
rem The first run downloads PyTorch and the model weights and takes 10-20 minutes.
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 (
  echo Nie znaleziono Pythona. Zainstaluj Python 3.12 z python.org albo: winget install Python.Python.3.12
  goto :fail
)
py -3.12 -c "import sys" >nul 2>nul
if errorlevel 1 (
  echo Brak Pythona 3.12 - flyvis i flygym nie dzialaja na 3.13 ani 3.14.
  echo Zainstaluj go poleceniem: winget install Python.Python.3.12
  goto :fail
)

if not exist ".venv\Scripts\python.exe" (
  echo == Tworze srodowisko .venv
  py -3.12 -m venv .venv
  if errorlevel 1 goto :fail
)
set "PY=.venv\Scripts\python.exe"

if not exist ".venv\flylab-installed.txt" (
  echo == Instaluje pakiety - to potrwa kilka minut
  "%PY%" -m pip install --upgrade pip
  if errorlevel 1 goto :fail
  "%PY%" -m pip install -e ".[body,lab]"
  if errorlevel 1 goto :fail
  echo == Pobieram wagi modelu wzroku muszki
  "%PY%" -m flyvis_cli.download_pretrained_models --skip_large_files
  if errorlevel 1 goto :fail
  echo installed> ".venv\flylab-installed.txt"
)

echo == Sprawdzam instalacje
"%PY%" -m wingloop.lab check
if errorlevel 1 goto :fail

echo == Uruchamiam demo: lot muszki, wzrok, uczenie, dron (okolo 10 minut)
"%PY%" -m wingloop.lab demo
if errorlevel 1 goto :fail

echo == Gotowe. Otwieram raport.
start "" "flylab-run\flylab.html"
pause
exit /b 0

:fail
echo.
echo Cos poszlo nie tak. Skopiuj tekst powyzej i wklej go w czacie.
pause
exit /b 1
