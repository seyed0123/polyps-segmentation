@echo off
setlocal
cd /d "%~dp0\.."

if not exist model.pth (
  echo ERROR: Copy model.pth into the project root before building.
  exit /b 1
)

python -m pip install --upgrade pyinstaller
if errorlevel 1 exit /b 1

python -m PyInstaller --clean --noconfirm packaging\PolypMonitor.spec
if errorlevel 1 exit /b 1

echo.
echo Build complete: dist\PolypMonitor\PolypMonitor.exe
echo Test the entire dist\PolypMonitor folder before creating the installer.
endlocal
