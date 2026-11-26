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
%PYTHON_EXE% -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
pause
exit /b 0

:choose_exercise
echo.
echo    ----------------------------------------------------------------------
echo       Choose an Exercise / اختر التمرين من التمارين الـ 16
echo    ----------------------------------------------------------------------
echo       [ Biceps / الذراع الأمامي ]
echo         [1]  Bicep Curl             [2]  Hammer Curl
echo.
echo       [ Chest / الصدر ]
echo         [3]  Push-up                [4]  Chest Fly
echo.
echo       [ Shoulders / الأكتاف ]
echo         [5]  Shoulder Press         [6]  Lateral Raise
echo.
echo       [ Triceps / الذراع الخلفي ]
echo         [7]  Chair Dips             [8]  Overhead Tricep
echo.
echo       [ Back / الظهر ]
echo         [9]  Superman               [10] Bent-Over Row
echo.
echo       [ Quads / الفخذ الأمامي ]
echo         [11] Squat                  [12] Wall Sit (Isometric Hold)
echo.
echo       [ Glutes & Hamstrings / المؤخرة والخلفي ]
echo         [13] Lunge                  [14] Glute Bridge
echo.
echo       [ Core & Abs / البطن والجذع ]
echo         [15] Plank (Isometric Hold) [16] Leg Raise
echo    ----------------------------------------------------------------------
echo.

:ask_exercise
set /p "exercise_choice=    Enter exercise (1-16): "

if "%exercise_choice%"=="1" set "exercise=bicep_curl" & goto choose_source
if "%exercise_choice%"=="2" set "exercise=hammer_curl" & goto choose_source
if "%exercise_choice%"=="3" set "exercise=push_up" & goto choose_source
if "%exercise_choice%"=="4" set "exercise=chest_fly" & goto choose_source
if "%exercise_choice%"=="5" set "exercise=shoulder_press" & goto choose_source
if "%exercise_choice%"=="6" set "exercise=lateral_raise" & goto choose_source
if "%exercise_choice%"=="7" set "exercise=dips" & goto choose_source
if "%exercise_choice%"=="8" set "exercise=overhead_tricep" & goto choose_source
if "%exercise_choice%"=="9" set "exercise=superman" & goto choose_source
if "%exercise_choice%"=="10" set "exercise=bent_over_row" & goto choose_source
if "%exercise_choice%"=="11" set "exercise=squat" & goto choose_source
if "%exercise_choice%"=="12" set "exercise=wall_sit" & goto choose_source
if "%exercise_choice%"=="13" set "exercise=lunge" & goto choose_source
if "%exercise_choice%"=="14" set "exercise=glute_bridge" & goto choose_source
if "%exercise_choice%"=="15" set "exercise=plank" & goto choose_source
if "%exercise_choice%"=="16" set "exercise=leg_raise" & goto choose_source

echo    [!] Invalid choice. Please enter 1 to 16.
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
if "%source_choice%"=="1" set "source=0" & goto choose_coach_lang
if "%source_choice%"=="2" goto ask_file
echo    [!] Invalid choice. Please enter 1 or 2.
goto ask_source

:ask_file
set /p "source=    Enter video file path: "
if not exist "%source%" (
    echo    [!] File not found: %source%
    goto ask_source
)

:choose_coach_lang
echo.
echo    ------------------------------------
echo       Choose Coach Voice Language
echo    ------------------------------------
echo       [1]  العربية (صوت كابتن مصري رجالي) - الافتراضي
echo       [2]  English (American Male Coach)
echo    ------------------------------------
echo.
set "coach_lang=ar"
set /p "lang_choice=    Enter choice (1 or 2, default 1): "
if "%lang_choice%"=="2" set "coach_lang=en"

:launch_engine
echo.
echo    ====================================================
echo       Starting live analysis for: %exercise%
echo       Voice Coach: %coach_lang% (100%% Male)
echo       Press 'q' or Esc to stop workout.
echo    ====================================================
echo.

%PYTHON_EXE% "%~dp0cv_engine.py" --source "%source%" --exercise %exercise% --lang %coach_lang%

echo.
echo    ====================================================
echo               WORKOUT SESSION FINISHED
echo    ====================================================
echo.
echo    [OK] Camera analysis session ended successfully.
echo    [INFO] Repetition analysis and form scores displayed above.
echo.
echo    ----------------------------------------------------
echo       [1] Choose another exercise
echo       [2] Return to main menu
echo       [3] Exit
echo    ----------------------------------------------------
echo.

:ask_after
set /p "after_choice=    Enter choice (1, 2 or 3): "
if "%after_choice%"=="1" goto choose_exercise
if "%after_choice%"=="2" goto choose_mode
if "%after_choice%"=="3" exit /b 0
echo    [!] Invalid choice. Please enter 1, 2 or 3.
goto ask_after
