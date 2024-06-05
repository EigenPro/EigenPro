@echo off
REM Get the directory path of the script
set "script_dir=%~dp0"

REM Remove the trailing backslash from script_dir
set "script_dir=%script_dir:~0,-1%"

REM Get the parent directory of script_dir
for %%i in ("%script_dir%") do set "parent_dir=%%~dpi"

REM Remove the trailing backslash from parent_dir
set "parent_dir=%parent_dir:~0,-1%"

REM Add the parent directory to PYTHONPATH
set "PYTHONPATH=%parent_dir%;%PYTHONPATH%"

REM Run the Python script
python "%script_dir%\run_example.py" --dataset=fmnist
