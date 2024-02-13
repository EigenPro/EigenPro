@echo off
REM Get the directory path of the script
set "script_dir=%~dp0"

REM Run the Python script
python "%script_dir%run_example.py" --dataset=fmnist
