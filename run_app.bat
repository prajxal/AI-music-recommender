@echo off
echo ======================================
echo Emotion Music Recommender Application
echo ======================================
echo.

REM Check if Python is installed
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo Python is not installed or not in PATH.
    echo Please install Python 3.10 or newer.
    pause
    exit /b
)

REM Check if setup has been run
if not exist requirements.txt (
    echo ERROR: requirements.txt not found.
    echo Please make sure you're running this from the correct directory.
    pause
    exit /b
)

REM Check if packages are installed
echo Checking requirements...
pip list | findstr "streamlit" >nul
if %errorlevel% neq 0 (
    echo Installing required packages...
    python setup.py
)

REM Run the application
echo.
echo Starting the application...
echo If this is your first time running the app, your browser will open automatically.
echo.
echo Press Ctrl+C in this window to stop the application.
echo.
streamlit run emotion_music_app.py

pause