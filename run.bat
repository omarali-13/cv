@echo off
setlocal EnableDelayedExpansion
title CV for Fit - AI Fitness Coach
cd /d "%~dp0"

echo.
echo    ====================================================
echo                   CV   FOR   FIT
echo         Real-Time Fitness Form Analysis Engine
echo    ====================================================
echo.

where py >nul 2>&1
if %errorlevel% neq 0 (
    where python >nul 2>&1
    if %errorlevel% neq 0 (
        echo    [ERROR] Python is not installed or not in PATH.
        echo    Please install Python 3.10+ from https://www.python.org
        echo.
        pause
        exit /b 1
    )
    set "PYTHON_EXE=python"
) else (
    set "PYTHON_EXE=py"
)

echo    [INFO] Checking dependencies...
%PYTHON_EXE% -c "import cv2, numpy, mediapipe, fastdtw, scipy, pyttsx3, fastapi, uvicorn" >nul 2>&1
if %errorlevel% neq 0 (
    echo    [!] Installing missing packages...
    %PYTHON_EXE% -m pip install mediapipe opencv-python numpy fastdtw scipy pyttsx3 uvicorn fastapi python-multipart
    if %errorlevel% neq 0 (
        echo    [ERROR] Failed to install packages.
        pause
        exit /b 1
    )
)
echo    [OK] Environment is ready.
echo.

:choose_mode
echo    ------------------------------------
echo       Choose Mode / Select Launch Mode
echo    ------------------------------------
echo       [1] Web Dashboard (Modern Browser UI) - RECOMMENDED
echo       [2] Desktop OpenCV Camera Window
echo    ------------------------------------
echo.

:ask_mode
set /p "mode_choice=    Enter choice (1 or 2): "
if "%mode_choice%"=="1" goto launch_web
if "%mode_choice%"=="2" goto choose_exercise
echo    [!] Invalid choice. Please enter 1 or 2.
goto ask_mode

:launch_web
echo.
echo    [OK] Starting CV for Fit Web Server...
echo    [INFO] Opening dashboard at http://localhost:8000
echo.
start http://localhost:8000
%PYTHON_EXE% -m uvicorn app.main:app --host 0.0.0.0 --port 8000
pause
exit /b 0

:choose_exercise
echo.
echo    ------------------------------------
echo       Choose an Exercise
echo    ------------------------------------
echo       [1]  Bicep Curl
echo       [2]  Squat
echo       [3]  Lunge
echo       [4]  Push-up
echo       [5]  Superman
echo       [6]  Shoulder Press
echo       [7]  Chair Dips
echo       [8]  Plank
echo    ------------------------------------
echo.

:ask_exercise
set /p "exercise_choice=    Enter exercise (1-8): "

if "%exercise_choice%"=="1" set "exercise=bicep_curl" & goto choose_source
if "%exercise_choice%"=="2" set "exercise=squat" & goto choose_source
if "%exercise_choice%"=="3" set "exercise=lunge" & goto choose_source
if "%exercise_choice%"=="4" set "exercise=push_up" & goto choose_source
if "%exercise_choice%"=="5" set "exercise=superman" & goto choose_source
if "%exercise_choice%"=="6" set "exercise=shoulder_press" & goto choose_source
if "%exercise_choice%"=="7" set "exercise=dips" & goto choose_source
if "%exercise_choice%"=="8" set "exercise=plank" & goto choose_source

echo    [!] Invalid choice. Please enter 1 to 8.
goto ask_exercise

:choose_source
echo.
echo    ------------------------------------
echo       Choose Video Source
echo    ------------------------------------
echo       [1]  Webcam (Live Camera)
echo       [2]  Video file (.mp4)
echo    ------------------------------------
echo.

:ask_source
set /p "source_choice=    Enter source (1 or 2): "
if "%source_choice%"=="1" set "source=0" & goto launch_engine
if "%source_choice%"=="2" goto ask_file
echo    [!] Invalid choice. Please enter 1 or 2.
goto ask_source

:ask_file
set /p "source=    Enter video file path: "
if not exist "%source%" (
    echo    [!] File not found: %source%
    goto ask_source
)

:launch_engine
echo.
echo    ====================================
echo       Starting live analysis for: %exercise%
echo       Press 'q' or Esc to stop.
echo    ====================================
echo.

%PYTHON_EXE% "%~dp0cv_engine.py" --source "%source%" --exercise %exercise%

echo.
echo    ====================================
echo       Session finished.
echo    ====================================
echo.
pause
exit /b 0
