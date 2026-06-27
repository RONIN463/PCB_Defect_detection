@echo off
setlocal EnableDelayedExpansion

set "NEWPATH="
for %%p in ("%PATH:;=";"%") do (
    set "p_lower=%%~p"
    if "!p_lower:mingw=!"=="!p_lower!" (
        if "!p_lower:msys=!"=="!p_lower!" (
            if "!NEWPATH!"=="" (
                set "NEWPATH=%%~p"
            ) else (
                set "NEWPATH=!NEWPATH!;%%~p"
            )
        )
    )
)

set "PATH=!NEWPATH!"
echo MinGW/msys paths removed from PATH
echo Launching PCB Defect Detection GUI...
python MainProgram.py
pause
