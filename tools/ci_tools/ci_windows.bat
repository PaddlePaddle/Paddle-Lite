@echo off
setlocal
setlocal enabledelayedexpansion

set branch=%1%
set agile_pull_id=%2%
set cmd="main"

set python_bin=C:\Python27\python.exe
set home_path=C:\xly\workspace
set code_path=%home_path%\py27\Paddle-Lite
set work_path=%AGILE_PULL_ID%_%AGILE_REVISION%
set python_bin_name=%python_bin%

:round
@echo off
if /I "%1"=="build" (
    goto main
) else if /I "%1"=="run_full_demo" (
    goto run_full_demo
) else if /I  "%1"=="run_light_demo" (
    goto run_light_demo
) else if /I  "%1"=="run_python_demo" (
    goto run_python_demo
) else if /I  "%1"=="without_avx" (
    set WITH_AVX=OFF
) else if /I  "%1"=="help" (
      call:print_usage
      goto:eof
) else (
    goto main
)
shift
goto round



:main
if NOT EXIST "%home_path%" (
    md "%home_path%"
)
cd %home_path%

echo path: %work_path%
if exist "%work_path%" (
    del /f /s /q "%work_path%"  >nul 2>&1
)
md %work_path%

cd %code_path%
echo %code_path%
del /f /s /q "build.lite.x86"  >nul 2>&1
del /f /s /q "third-party"  >nul 2>&1

git checkout %branch%
git pull
git branch -D test

git fetch origin pull/%agile_pull_id%/head
git checkout -b test FETCH_HEAD
git branch
git log --pretty=oneline -10

rem echo %python_bin_name% | lite\tools\build_windows.bat

set whl_path=%code_path%\build.lite.x86\inference_lite_lib\python\install\dist
if exist %whl_path% (
    exit 0
) else (
    echo ".whl is not exist"
    exit 1
)

rem xcopy  %code_path%\build.lite.x86\inference_lite_lib %home_path%\%work_path% /s/h/e/k/f/c

if %ERRORLEVEL% NEQ 0 set EXCODE=%ERRORLEVEL%

rem rmdir C:\Users\administrator\Downloads\workspace /s/q

echo EXCODE: %EXCODE%
exit /b %EXCODE%
goto:eof



:run_full_demo
set cxx_demo_path=%home_path%\%work_path%\inference_lite_lib\demo\cxx
set cxx_full_demo_path=%cxx_demo_path%\mobilenetv1_full
cd %cxx_full_demo_path%
build.bat

set cxx_bin_name=%cxx_demo_path%\mobilenetv1_full\build\Release\mobilenet_full_api.exe
if exist %cxx_bin_name% (
    echo "demo full api exist"
) else (
    echo "build demo api fail,mobilenet_full_api.exe is not exist"
    exit 1
)
if %ERRORLEVEL% NEQ 0 set EXCODE=%ERRORLEVEL%
echo EXCODE: %EXCODE%
exit /b %EXCODE%
goto:eof



:run_light_demo
set cxx_demo_path=%home_path%\%work_path%\inference_lite_lib\demo\cxx
set cxx_light_demo_path=%cxx_demo_path%\mobilenetv1_light
cd %cxx_light_demo_path%
build.bat

set cxx_bin_name=%cxx_demo_path%\mobilenetv1_light\build\Release\mobilenet_light_api.exe
if exist %cxx_bin_name% (
    echo "demo light api exist"
) else (
    echo "build demo api fail,mobilenet_light_api.exe is not exist"
    exit 1
)

cd %home_path%
echo "============clean==========="
echo work_path: %work_path%
if exist "%work_path%" (
    del /f /s /q "%work_path%"  >nul 2>&1
)
if %ERRORLEVEL% NEQ 0 set EXCODE=%ERRORLEVEL%
echo EXCODE: %EXCODE%
exit /b %EXCODE%
goto:eof