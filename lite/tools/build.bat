@echo off
setlocal
setlocal enabledelayedexpansion

set source_path=%~dp0\\..\\..\\
rem  global variables
set BUILD_EXTRA=OFF
set BUILD_JAVA=ON
set BUILD_PYTHON=OFF
set BUILD_DIR=%source_path%
set OPTMODEL_DIR=""
set BUILD_TAILOR=OFF
set BUILD_CV=OFF
set WITH_LOG=ON  

set THIRDPARTY_TAR=https://paddle-inference-dist.bj.bcebos.com/PaddleLite/third-party-05b862.tar.gz

set workspace=%source_path%

:set_vcvarsall_dir
SET /P vcvarsall_dir="Please input the path of visual studio command Prompt, such as C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat   =======>"
set tmp_var=!vcvarsall_dir!
call:remove_space
set vcvarsall_dir=!tmp_var!   
IF NOT EXIST "%vcvarsall_dir%" (
    echo "------------%vcvarsall_dir% not exist------------"
    goto set_vcvarsall_dir
)

call:prepare_thirdparty

if EXIST "%build_directory%" (
    call:rm_rebuild_dir "%build_directory%"
    md "%build_directory%"
) 

set root_dir=%workspace%
set build_directory=%BUILD_DIR%\build.lite.x86
set GEN_CODE_PATH_PREFIX=%build_directory%\lite\gen_code
set DEBUG_TOOL_PATH_PREFIX=%build_directory%\lite\tools\debug

rem for code gen, a source file is generated after a test, but is dependended by some targets in cmake.
rem here we fake an empty file to make cmake works.
if NOT EXIST "%GEN_CODE_PATH_PREFIX%" (
    md "%GEN_CODE_PATH_PREFIX%"
)

type nul >"%GEN_CODE_PATH_PREFIX%\__generated_code__.cc"

if NOT EXIST "%DEBUG_TOOL_PATH_PREFIX%" (
     md "%DEBUG_TOOL_PATH_PREFIX%"
)

copy "%root_dir%\lite\tools\debug\analysis_tool.py" "%DEBUG_TOOL_PATH_PREFIX%\"

cd "%build_directory%"

  cmake ..   -G "Visual Studio 14 2015 Win64" -T host=x64  -DWITH_MKL=ON      ^
            -DWITH_MKLDNN=OFF   ^
            -DLITE_WITH_X86=ON  ^
            -DLITE_WITH_PROFILE=OFF ^
            -DWITH_LITE=ON ^
            -DLITE_WITH_LIGHT_WEIGHT_FRAMEWORK=OFF ^
            -DLITE_WITH_ARM=OFF ^
            -DWITH_GPU=OFF ^
            -DLITE_BUILD_EXTRA=ON ^
            -DLITE_WITH_PYTHON=ON ^
            -DPYTHON_EXECUTABLE="%python_path%"

call "%vcvarsall_dir%" amd64

msbuild /m /p:Configuration=Release lite\publish_inference.vcxproj >mylog.txt 2>&1
goto:eof

:prepare_thirdparty 
    SET /P python_path="Please input the path of python.exe, such as C:\Python35\python.exe, C:\Python35\python3.exe   =======>"
    set tmp_var=!python_path!
    call:remove_space
    set python_path=!tmp_var!   
    if "!python_path!"=="" (
      set python_path=python.exe
    ) else (
      if NOT exist "!python_path!" (
        echo "------------!python_path! not exist------------" 
        goto:eof
      )  
    )

    if  EXIST "%workspace%\third-party" (
        if NOT EXIST "%workspace%\third-party-05b862.tar.gz" (
            echo "The directory of third_party exists, the third-party-05b862.tar.gz not exists."            
        ) else (
               echo "The directory of third_party exists, the third-party-05b862.tar.gz exists."
               call:rm_rebuild_dir "%workspace%\third-party"
               !python_path! %workspace%\lite\tools\untar.py %source_path%\third-party-05b862.tar.gz %workspace%
        )
    ) else (
        if NOT EXIST "%workspace%\third-party-05b862.tar.gz" (
            echo "The directory of third_party not exists, the third-party-05b862.tar.gz not exists."
            call:download_third_party
            !python_path! %workspace%\lite\tools\untar.py %source_path%\third-party-05b862.tar.gz %workspace%
        ) else (
            echo "The directory of third_party not exists, the third-party-05b862.tar.gz exists."
               !python_path! %workspace%\lite\tools\untar.py %source_path%\third-party-05b862.tar.gz %workspace%
        )

    )
    git submodule update --init --recursive
goto:eof

:download_third_party
powershell.exe (new-object System.Net.WebClient).DownloadFile('https://paddle-inference-dist.bj.bcebos.com/PaddleLite/third-party-05b862.tar.gz', ^
'%workspace%third-party-05b862.tar.gz')
goto:eof

:rm_rebuild_dir
    del /f /s /q "%~1\*.*"  >nul 2>&1
    rd /s /q  "%~1" >nul 2>&1
goto:eof


:remove_space
:remove_left_space
if "%tmp_var:~0,1%"==" " (
    set "tmp_var=%tmp_var:~1%"
    goto remove_left_space
)

:remove_right_space
if "%tmp_var:~-1%"==" " (
    set "tmp_var=%tmp_var:~0,-1%"
    goto remove_left_space
)
goto:eof
