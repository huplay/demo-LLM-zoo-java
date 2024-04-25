@echo off
CHCP 65001 >nul
if exist modelConfig/%1/setup.bat call modelConfig/%1/setup.bat
java %GPT_JAVA_ARGS% -jar app/target/demo-llm-zoo.jar %*
