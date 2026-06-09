@echo off
chcp 65001 >nul

title 微博情感分析系统

echo =====================================
echo      微博情感分析系统启动中...
echo =====================================
echo.

echo [1/3] 检查 Ollama...

tasklist | findstr ollama.exe >nul

if errorlevel 1 (
    echo 未检测到 Ollama，正在启动...
    start "" ollama serve
    timeout /t 5 >nul
) else (
    echo Ollama 已运行
)

echo.
echo [2/3] 检查模型...

ollama list | findstr qwen2.5:14b >nul

if errorlevel 1 (
    echo.
    echo 未找到 qwen2.5:14b
    echo 请先执行：
    echo ollama pull qwen2.5:14b
    pause
    exit
)

echo 模型检查通过

echo.
echo [3/3] 启动 Streamlit...

start "" http://localhost:8501

streamlit run app.py

pause