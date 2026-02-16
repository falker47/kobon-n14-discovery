@echo off
setlocal
cd /d "%~dp0"

echo Compiling main.tex...
pdflatex -interaction=nonstopmode main.tex

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Compilation failed!
    pause
    exit /b %errorlevel%
)

echo.
echo [SUCCESS] PDF generated successfully.
timeout /t 3
endlocal
