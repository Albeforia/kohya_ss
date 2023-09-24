@echo off

:: Deactivate the virtual environment
call .\venv\Scripts\deactivate.bat

:: Calling external python program to check for local modules
python .\setup\check_local_modules.py --no_question

:: Activate the virtual environment
call .\venv\Scripts\activate.bat
set PATH=%PATH%;%~dp0venv\Lib\site-packages\torch\lib

:: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#lazy-loading
set CUDA_MODULE_LOADING=LAZY

:: Validate requirements
@REM python.exe .\setup\validate_requirements.py

:: If the exit code is 0, run the kohya_gui.py script with the command-line arguments
if %errorlevel% equ 0 (
    REM Check if the batch was started via double-click
    IF /i "%comspec% /c %~0 " equ "%cmdcmdline:"=%" (
        REM echo This script was started by double clicking.
        cmd /k python.exe kohya_gui.py %*
    ) ELSE (
        REM echo This script was started from a command prompt.
        python.exe kohya_gui.py %*
    )
)
