@echo off
chcp 65001 >nul 2>&1
title CV for Fit - Fitness Tracker

echo.
echo    ====================================================
echo                   CV   FOR   FIT
echo         Real-Time Fitness Form Analysis Engine
echo    ====================================================
echo.

:: -------------------------------------------
:: Check Python is available
:: -------------------------------------------
where py >nul 2>&1
if %errorlevel% neq 0 (
    echo    [ERROR] Python not found on this system.
    echo    Please install Python 3.10+ from https://www.python.org
    echo.
    pause
    exit /b 1
)

:: -------------------------------------------
:: Check dependencies one by one
:: -------------------------------------------
echo    Checking dependencies...
echo.

py -c "import mediapipe" >nul 2>&1
if %errorlevel% neq 0 goto install_deps

py -c "import cv2" >nul 2>&1
if %errorlevel% neq 0 goto install_deps

py -c "import numpy" >nul 2>&1
if %errorlevel% neq 0 goto install_deps

py -c "import fastdtw" >nul 2>&1
if %errorlevel% neq 0 goto install_deps

py -c "import scipy" >nul 2>&1
if %errorlevel% neq 0 goto install_deps

py -c "import pyttsx3" >nul 2>&1
if %errorlevel% neq 0 goto install_deps

echo    [OK] All dependencies found.
echo.
goto choose_exercise

:install_deps
echo    [!] Some dependencies are missing. Installing now...
echo.
py -m pip install mediapipe opencv-python numpy fastdtw scipy pyttsx3
if %errorlevel% neq 0 (
    echo.
    echo    [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)
echo.
echo    [OK] Dependencies installed successfully.
echo.

:: -------------------------------------------
:: Choose exercise
:: -------------------------------------------
:choose_exercise
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
set /p "exercise_choice=    Enter your choice (1-8): "

if "%exercise_choice%"=="1" (
    set "exercise=bicep_curl"
    echo.
    echo    [OK] Selected: Bicep Curl
    echo.
    goto choose_source
)
if "%exercise_choice%"=="2" (
    set "exercise=squat"
    echo.
    echo    [OK] Selected: Squat
    echo.
    goto choose_source
)
if "%exercise_choice%"=="3" (
    set "exercise=lunge"
    echo.
    echo    [OK] Selected: Lunge
    echo.
    goto choose_source
)
if "%exercise_choice%"=="4" (
    set "exercise=push_up"
    echo.
    echo    [OK] Selected: Push-up
    echo.
    goto choose_source
)
if "%exercise_choice%"=="5" (
    set "exercise=superman"
    echo.
    echo    [OK] Selected: Superman
    echo.
    goto choose_source
)
if "%exercise_choice%"=="6" (
    set "exercise=shoulder_press"
    echo.
    echo    [OK] Selected: Shoulder Press
    echo.
    goto choose_source
)
if "%exercise_choice%"=="7" (
    set "exercise=dips"
    echo.
    echo    [OK] Selected: Chair Dips
    echo.
    goto choose_source
)
if "%exercise_choice%"=="8" (
    set "exercise=plank"
    echo.
    echo    [OK] Selected: Plank
    echo.
    goto choose_source
)

echo    [!] Invalid choice. Please enter 1 to 8.
goto ask_exercise

:: -------------------------------------------
:: Choose source
:: -------------------------------------------
:choose_source
echo    ------------------------------------
echo       Choose Video Source
echo    ------------------------------------
echo       [1]  Webcam (live)
echo       [2]  Video file
echo    ------------------------------------
echo.

:ask_source
set /p "source_choice=    Enter your choice (1 or 2): "

if "%source_choice%"=="1" (
    set "source=0"
    echo.
    echo    [OK] Using: Webcam
    echo.
    goto launch
)
if "%source_choice%"=="2" (
    goto ask_file
)

echo    [!] Invalid choice. Please enter 1 or 2.
goto ask_source

:ask_file
set /p "source=    Enter video file path: "
if not exist "%source%" (
    echo    [!] File not found: %source%
    echo    Please check the path and try again.
    echo.
    goto ask_source
)
echo.
echo    [OK] Using: %source%
echo.

:: -------------------------------------------
:: Launch the engine
:: -------------------------------------------
:launch
echo    ====================================
echo       Starting live analysis...
echo       Press 'q' or Esc to stop.
echo    ====================================
echo.

py "%~dp0cv_engine.py" --source "%source%" --exercise %exercise%

echo.
echo    ====================================
echo       Session ended.
echo       Thanks for using CV for Fit!
echo    ====================================
echo.
pause
