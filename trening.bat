@echo off
rem Practice flights: the fly (or the drone) learns from every landing, and remembers.
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Najpierw uruchom flylab.bat - instaluje wszystko, co potrzebne.
  pause
  exit /b 1
)
echo Trening w locie: muszka (albo dron) leci misje na nowych arenach.
echo Po kazdym ladowaniu dostaje nagrode (cel) albo kare (nie cel)
echo i uczy sie na widokach z prawdziwego lotu. Pamiec zapisuje sie na dysku,
echo a cel.bat sam ja potem wczytuje. Kolejny trening doklada do tego, co juz umie.
echo Kazdy cel jest najpierw ogladany z bliska; na treningu muszka laduje mimo watpliwosci,
echo zeby sie przekonac. Na koncu egzamin na 6 nowych arenach i ocena.
echo.
set CEL=
set /p CEL=Do czego ma sie uczyc leciec? kula / plytka / slupek [kula]: 
if "%CEL%"=="" set CEL=kula
set KTO=
set /p KTO=Kto trenuje: mucha / dron [mucha]: 
set DRON=
if /i "%KTO%"=="dron" set DRON=--dron
set ILE=
set /p ILE=Ile misji treningowych [20]: 
if "%ILE%"=="" set ILE=20
echo.
".venv\Scripts\python.exe" -m wingloop.lab trening %CEL% --misje %ILE% %DRON%
if errorlevel 1 (
  echo.
  echo Cos poszlo nie tak. Skopiuj tekst powyzej i wklej go w czacie.
)
pause
