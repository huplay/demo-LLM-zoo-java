@echo off
CHCP 65001 >nul
java %GPT_JAVA_ARGS% -jar app/target/demo-llm-zoo.jar %*
