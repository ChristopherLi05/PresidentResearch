@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem Compare one President agent against every other built-in agent.
rem Usage: compare_agent.bat AgentName [trials-per-configuration] [seed] [workers] [parallel-matchups]
rem Example: compare_agent.bat ExperiencedAgent3 1000 42 3 2

pushd "%~dp0" || exit /b 1

if /I "%~1"=="--run-match" goto run_match

set "AGENT=%~1"
if "%AGENT%"=="" (
    set /p "AGENT=Agent class to compare (for example, ExperiencedAgent3): "
)
if "%AGENT%"=="" (
    echo No agent was supplied.
    echo Usage: %~nx0 AgentName [trials-per-configuration] [seed] [workers] [parallel-matchups]
    popd
    exit /b 1
)

rem A short class name is resolved from src.agent; fully-qualified specs also work.
if "%AGENT::=%"=="%AGENT%" set "AGENT=src.agent:%AGENT%"

set "TRIALS=%~2"
if "%TRIALS%"=="" set "TRIALS=1000"
set "SEED=%~3"
if "%SEED%"=="" set "SEED=42"
set "WORKERS=%~4"
if "%WORKERS%"=="" set "WORKERS=6"
set "PARALLEL_MATCHUPS=%~5"
if "%PARALLEL_MATCHUPS%"=="" set "PARALLEL_MATCHUPS=2"

if exist ".venv\Scripts\python.exe" (
    set "PYTHON=.venv\Scripts\python.exe"
) else (
    set "PYTHON=python"
)

if not exist "results\comparisons" mkdir "results\comparisons"
set "STATUS_DIR=results\comparisons\.batch-status-%RANDOM%%RANDOM%"
mkdir "!STATUS_DIR!" || (
    echo Could not create the temporary status directory.
    popd
    exit /b 1
)

echo.
echo Comparing %AGENT% against all other built-in agents
echo Trials per seating: %TRIALS%  ^|  seed: %SEED%  ^|  worker threads per matchup: %WORKERS%
echo Concurrent matchups: %PARALLEL_MATCHUPS%
echo.

set /a JOB_COUNT=0
for %%O in (
    "src.agent:RandomAgent"
    "src.agent:BasicAgent"
    "src.agent:BasicAgentPlus"
    "src.agent:ExperiencedAgent"
    "src.agent:ExperiencedAgent2"
    "src.agent:ExperiencedAgent3"
    "src.agent:SwitchPAgent"
) do (
    if /I not "%AGENT%"=="%%~O" (
        call :wait_for_slot
        set /a JOB_COUNT+=1
        start "" /b "%ComSpec%" /d /c call "%~f0" --run-match "!AGENT!" "%%~O" "!TRIALS!" "!SEED!" "!WORKERS!" "!STATUS_DIR!\!JOB_COUNT!"
    )
)

call :wait_for_all
set "HAD_FAILURE=0"
for /L %%I in (1,1,!JOB_COUNT!) do (
    set "JOB_STATUS="
    set /p "JOB_STATUS=" < "!STATUS_DIR!\%%I.done"
    type "!STATUS_DIR!\%%I.log"
    if not "!JOB_STATUS!"=="0" set "HAD_FAILURE=1"
)
rd /s /q "!STATUS_DIR!"
popd
if "!HAD_FAILURE!"=="1" exit /b 1
echo All comparisons finished. JSON reports are in results\comparisons\
exit /b 0

:wait_for_slot
call :count_finished
set /a ACTIVE_MATCHUPS=JOB_COUNT-FINISHED_JOBS
if !ACTIVE_MATCHUPS! GEQ !PARALLEL_MATCHUPS! (
    ping 127.0.0.1 -n 2 > nul
    goto wait_for_slot
)
exit /b 0

:wait_for_all
call :count_finished
if !FINISHED_JOBS! LSS !JOB_COUNT! (
    ping 127.0.0.1 -n 2 > nul
    goto wait_for_all
)
exit /b 0

:count_finished
set /a FINISHED_JOBS=0
for %%F in ("!STATUS_DIR!\*.done") do if exist "%%~fF" set /a FINISHED_JOBS+=1
exit /b 0

:run_match
set "INPUT_AGENT=%~2"
set "OPPONENT=%~3"
set "TRIALS=%~4"
set "SEED=%~5"
set "WORKERS=%~6"
set "STATUS_BASE=%~7"
set "INPUT_FILE=!INPUT_AGENT::=_!"
set "OPPONENT_FILE=!OPPONENT::=_!"
set "OUTPUT=results\comparisons\!INPUT_FILE!_vs_!OPPONENT_FILE!_seed!SEED!.json"
if exist ".venv\Scripts\python.exe" (set "PYTHON=.venv\Scripts\python.exe") else (set "PYTHON=python")
"!PYTHON!" arena_multithread.py "!INPUT_AGENT!" "!OPPONENT!" --trials-per-configuration !TRIALS! --seed !SEED! --workers !WORKERS! --output "!OUTPUT!" > "!STATUS_BASE!.log" 2>&1
set "MATCH_EXIT_CODE=!ERRORLEVEL!"
> "!STATUS_BASE!.done" echo !MATCH_EXIT_CODE!
exit /b !MATCH_EXIT_CODE!
