@echo off
setlocal

cd /d "%~dp0"

echo Stopping any existing Omega backend...
taskkill /F /IM omega-acad.exe >nul 2>&1

echo Building omega-acad (fast if nothing changed)...
cargo build -p omega-acad
if errorlevel 1 (
    echo Build failed - not starting.
    pause
    exit /b 1
)

echo Starting Omega backend in its own window...
start "Omega Backend" cmd /k "target\debug\omega-acad.exe"

echo Waiting for the backend to come up...
timeout /t 3 /nobreak >nul

echo Launching the Mind Map visualization...
set "GODOT_DIR=%LOCALAPPDATA%\Microsoft\WinGet\Packages\GodotEngine.GodotEngine_Microsoft.Winget.Source_8wekyb3d8bbwe"
set "GODOT_EXE="
for %%F in ("%GODOT_DIR%\Godot_*_win64.exe") do set "GODOT_EXE=%%F"
if not defined GODOT_EXE (
    echo Godot executable not found under %GODOT_DIR%
    echo A winget upgrade likely changed the version-suffixed filename - the backend is still running; launch Godot manually with: --path "viz"
    pause
    exit /b 1
)
start "" "%GODOT_EXE%" --path "viz"

endlocal
