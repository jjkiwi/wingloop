@echo off
rem Fly the fly yourself in a 3-D window. Run flylab.bat once first: it installs everything.
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Najpierw uruchom flylab.bat - instaluje wszystko, co potrzebne.
  pause
  exit /b 1
)
echo Sterowanie w oknie:
echo   strzalki gora/dol   - szybciej / wolniej
echo   strzalki lewo/prawo - skret o 15 stopni
echo   W / S               - w gore / w dol o 3 mm
echo   spacja              - pauza
echo   [ / ]               - wolniej / szybciej (zwolnione tempo)
echo   H                   - zawis
echo   C                   - kamera: za mucha / swobodna (mysz)
echo.
".venv\Scripts\python.exe" -m wingloop.lab watch
if errorlevel 1 (
  echo.
  echo Cos poszlo nie tak. Skopiuj tekst powyzej i wklej go w czacie.
)
pause
