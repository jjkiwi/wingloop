@echo off
rem The fly flies to its targets on its own. Run flylab.bat once first: it installs everything.
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Najpierw uruchom flylab.bat - instaluje wszystko, co potrzebne.
  pause
  exit /b 1
)
echo Muszka poleci sama do celow, ktorych ja nauczysz.
echo.
echo   1. Trening na uwiezi: muszka oglada kule, plytki i slupki, dostaje
echo      dopamine za cel i kare za reszte (pierwszy raz 2-4 minuty, potem od razu).
echo   2. Lot: rozglada sie, ocenia kazdy obiekt tym, czego sie nauczyla,
echo      leci do najlepszego, hamuje, szuka nastepnego.
echo.
set CEL=
set /p CEL=Do czego ma leciec? kula / plytka / slupek [kula]: 
if "%CEL%"=="" set CEL=kula
set KTO=
set /p KTO=Kto leci: mucha / dron (dron z mozgiem muchy) [mucha]: 
set DRON=
if /i "%KTO%"=="dron" set DRON=--dron
set ARENA=
set /p ARENA=Ktora arena (liczba, kazda inaczej rozstawiona) [0]: 
if "%ARENA%"=="" set ARENA=0
echo.
echo W oknie 3D: spacja - pauza, C - kamera za mucha / swobodna (mysz).
echo W przegladarce: mapa areny, oceny obiektow i mozg muchy na zywo.
echo.
".venv\Scripts\python.exe" -m wingloop.lab cel %CEL% --seed %ARENA% %DRON%
if errorlevel 1 (
  echo.
  echo Cos poszlo nie tak. Skopiuj tekst powyzej i wklej go w czacie.
)
pause
