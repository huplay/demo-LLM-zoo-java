@echo off
CHCP 65001 >nul
java %GPT_JAVA_ARGS% -jar --add-modules=jdk.incubator.vector app/target/demo-llm-zoo.jar %*