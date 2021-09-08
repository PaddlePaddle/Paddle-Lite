@echo off
setlocal
setlocal enabledelayedexpansion

set branch=%1%
set agile_pull_id=%2%
set agile_revision_id=%3%
set cmd="main"

set python_bin=C:\Python27\python.exe
set home_path=C:\xly\workspace
set xly_root=C:\xly
set code_path=%xly_root%\py27\Paddle-Lite
set work_path=%agile_pull_id%_%agile_revision_id%
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
    mkdir "%home_path%"
)
cd %home_path%

echo path: %work_path%
if exist "%work_path%" (
    rmdir "%work_path%" /s/q
)
mkdir %work_path%

cd %code_path%
echo %code_path%
rmdir  "build.lite.x86" /s /q
rmdir  "third-party" /s /q

git checkout %branch%
git pull
git branch -D test

git fetch origin pull/%agile_pull_id%/head
git checkout -b test FETCH_HEAD
git branch

:: Merge the latest branch
git config --local user.name "PaddleCI"
git config --local user.email "paddle_ci@example.com"
git merge --no-edit origin/%branch%
git log --pretty=oneline -10

rem echo %python_bin_name% | lite\tools\build_windows.bat

set whl_path=%code_path%\build.lite.x86\inference_lite_lib\python\install\dist
if exist %whl_path% (
    exit 0
) else (
    echo ".whl is not exist"
    exit 1
)

xcopy  %code_path%\build.lite.x86\inference_lite_lib %home_path%\%work_path% /s/h/e/k/f/c

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

if %ERRORLEVEL% NEQ 0 set EXCODE=%ERRORLEVEL%
echo EXCODE: %EXCODE%
exit /b %EXCODE%
goto:eof

:run_python_demo
set ce_tools=C:\xiaowen01
set opt_model_path=%ce_tools%\auto_test_ce\script
set light_api_model=%opt_model_path%\mobilenet_v1.nb
set full_api_model=%opt_model_path%\..\fluid_models_uncombined\mobilenet_v1

set whl_path=%home_path%\%work_path%\inference_lite_lib\python\install\dist
:: install lite lib
cd %whl_path%
for /F %%i in ('dir /B') do ( set whl_name=%%i)
echo y | %python_bin_name% -m pip uninstall %whl_name%
%python_bin_name% -m pip install %whl_name%

cd %opt_model_path%
%python_bin_name% run_opt_model_use_python.py
for /F %%i in ('dir /b /a-d . ^| find /v /c "::"') do set file_num=%%i
echo %file_num%
if %file_num% LSS 22 (
    echo "exist model opt fail"
    exit 1
)
:: for /F %%i in (dir /S/B | findstr /I nb | find /v /c "&#@") do ( set opted_model_num=%%i) 
:: echo opted_model_num

set python_demo_path=%home_path%\%work_path%\inference_lite_lib\demo\python
%python_bin_name% %python_demo_path%\mobilenetv1_light_api.py --model_dir=%light_api_model%
%python_bin_name% %python_demo_path%\mobilenetv1_full_api.py --model_dir=%full_api_model%

set cxx_demo_path=%home_path%\%work_path%\inference_lite_lib\demo\cxx
set cxx_light_demo_path=%cxx_demo_path%\mobilenetv1_light
cd %cxx_light_demo_path%
set cxx_bin_name=mobilenet_light_api.exe
%cxx_light_demo_path%\build\Release\%cxx_bin_name% %light_api_model%

set cxx_full_demo_path=%cxx_demo_path%\mobilenetv1_full
cd %cxx_full_demo_path%
set cxx_bin_name=mobilenet_full_api.exe
%cxx_full_demo_path%\build\Release\%cxx_bin_name% %full_api_model%

cd %home_path%
echo "============clean==========="
echo work_path: %work_path%
if exist "%work_path%" (
    rmdir "%work_path%" /s /q
)
if %ERRORLEVEL% NEQ 0 set EXCODE=%ERRORLEVEL%
echo EXCODE: %EXCODE%
exit /b %EXCODE%
goto:eof
