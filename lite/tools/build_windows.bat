@echo off
setlocal
setlocal enabledelayedexpansion

set source_path=%~dp0\\..\\..\\
set BUILD_EXTRA=OFF
set WITH_PYTHON=ON
set BUILD_DIR=%source_path%
set WITH_LOG=ON
set WITH_PROFILE=OFF
set WITH_PRECISION_PROFILE=OFF
set WITH_TESTING=OFF
set BUILD_FOR_CI=OFF
set BUILD_PLATFORM=x64
set MSVC_STATIC_CRT=ON
set WITH_STATIC_MKL=OFF
set WITH_OPENCL=OFF
set WITH_AVX=ON
set WITH_KUNLUNXIN_XPU=OFF
set KUNLUNXIN_XPU_SDK_ROOT=""
set BUILD_TYPE=Release
set CMAKE_GENERATOR=Visual Studio 14 2015
set ARCH=""
set WITH_STRIP=OFF
set OPTMODEL_DIR=""
set THIRDPARTY_URL=https://paddlelite-data.bj.bcebos.com/third_party_libs/
set THIRDPARTY_TAR=third-party-91a9ab3.tar.gz

set workspace=%source_path%
set /a cores=%number_of_processors%-2 > null

:round
@echo off
if /I "%1"=="with_extra" (
    set BUILD_EXTRA=ON
) else if /I "%1"=="without_python" (
    set WITH_PYTHON=OFF
) else if /I  "%1"=="with_profile" (
    set WITH_PROFILE=ON
) else if /I  "%1"=="with_precision_profile" (
    set WITH_PRECISION_PROFILE=ON
) else if /I  "%1"=="without_log" (
    set WITH_LOG=OFF
) else if /I  "%1"=="with_strip" (
    set WITH_STRIP=ON
    set OPTMODEL_DIR="%2"
) else if /I  "%1"=="build_x86" (
    set BUILD_PLATFORM=Win32
) else if /I  "%1"=="use_ninja" (
    set CMAKE_GENERATOR=Ninja
) else if /I  "%1"=="use_vs2017" (
    set CMAKE_GENERATOR=Visual Studio 15 2017
) else if /I  "%1"=="use_vs2019" (
    set CMAKE_GENERATOR=Visual Studio 16 2019
) else if /I  "%1"=="with_dynamic_crt" (
    set MSVC_STATIC_CRT=OFF
) else if /I  "%1"=="with_static_mkl" (
    set WITH_STATIC_MKL=ON
) else if /I  "%1"=="with_opencl" (
    set WITH_OPENCL=ON
) else if /I  "%1"=="without_avx" (
    set WITH_AVX=OFF
) else if /I  "%1"=="with_kunlunxin_xpu" (
    set WITH_KUNLUNXIN_XPU=ON
) else if /I  "%1"=="kunlunxin_xpu_sdk_root" (
    set KUNLUNXIN_XPU_SDK_ROOT="%2"
) else if /I  "%1"=="build_for_ci" (
    set BUILD_FOR_CI=ON
    set WITH_TESTING=ON
    set BUILD_EXTRA=ON
    set WITH_PROFILE=ON
) else if /I  "%1"=="help" (
      call:print_usage
      goto:eof
) else (
    goto main
)
shift
goto round

:main
if "%WITH_PYTHON%"=="ON" (
    set BUILD_EXTRA=ON
)

cd "%workspace%"

echo "------------------------------------------------------------------------------------------------------|"
echo "|  WITH_LOG=%WITH_LOG%                                                                                |"
echo "|  BUILD_EXTRA=%BUILD_EXTRA%                                                                          |"
echo "|  WITH_PYTHON=%WITH_PYTHON%                                                                          |"
echo "|  LITE_WITH_PROFILE=%WITH_PROFILE%                                                                   |"
echo "|  LITE_WITH_PRECISION_PROFILE=%WITH_PRECISION_PROFILE%                                               |"
echo "|  WITH_TESTING=%WITH_TESTING%                                                                        |"
echo "|  WITH_STRIP=%WITH_STRIP%                                                                            |"
echo "|  OPTMODEL_DIR=%OPTMODEL_DIR%                                                                        |"
echo "|  BUILD_PLATFORM=%BUILD_PLATFORM%                                                                    |"
echo "|  WITH_STATIC_MKL=%WITH_STATIC_MKL%                                                                  |"
echo "|  MSVC_STATIC_CRT=%MSVC_STATIC_CRT%                                                                  |"
echo "|  WITH_OPENCL=%WITH_OPENCL%                                                                          |"
echo "|  WITH_AVX=%WITH_AVX%                                                                                |"
echo "|  WITH_KUNLUNXIN_XPU=%WITH_KUNLUNXIN_XPU%                                                            |"
echo "|  KUNLUNXIN_XPU_SDK_ROOT=%KUNLUNXIN_XPU_SDK_ROOT%                                                    |"
echo "------------------------------------------------------------------------------------------------------|"


set vcvarsall_dir=C:\Program Files ^(x86^)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat
if "%CMAKE_GENERATOR%"=="Visual Studio 14 2015" (
  set vcvarsall_dir=C:\Program Files ^(x86^)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat
) else if "%CMAKE_GENERATOR%"=="Visual Studio 15 2017" (
  set vcvarsall_dir=C:\Program Files ^(x86^)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvarsall.bat
) else if "%CMAKE_GENERATOR%"=="Visual Studio 16 2019" (
  set vcvarsall_dir=C:\Program Files ^(x86^)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat
)
IF NOT EXIST "%vcvarsall_dir%" (
  goto set_vcvarsall_dir
)

call:set_python_path

call:prepare_thirdparty

set root_dir=%workspace%
set build_directory=%BUILD_DIR%\build.lite.x86
if "%WITH_OPENCL%"=="ON" (
    set "build_directory=%build_directory%.opencl"
    call:prepare_opencl_source_code
)
set GEN_CODE_PATH_PREFIX=%build_directory%\lite\gen_code
set DEBUG_TOOL_PATH_PREFIX=%build_directory%\lite\tools\debug
set Test_FILE="%build_directory%\lite_tests.txt"

REM "Clean the build directory."
if EXIST "%build_directory%" (
    call:rm_rebuild_dir "%build_directory%"
    md "%build_directory%"
)

REM "for code gen, a source file is generated after a test, but is dependended by some targets in cmake."
REM "here we fake an empty file to make cmake works."
if NOT EXIST "%GEN_CODE_PATH_PREFIX%" (
    md "%GEN_CODE_PATH_PREFIX%"
)

type nul >"%GEN_CODE_PATH_PREFIX%\__generated_code__.cc"

if NOT EXIST "%DEBUG_TOOL_PATH_PREFIX%" (
     md "%DEBUG_TOOL_PATH_PREFIX%"
)

copy "%root_dir%\lite\tools\debug\analysis_tool.py" "%DEBUG_TOOL_PATH_PREFIX%\"

cd "%build_directory%"

if "%CMAKE_GENERATOR%"=="Ninja" (
    pip install ninja
    if %errorlevel% NEQ 0 (
        echo pip install ninja failed!
        exit /b 7
    )
    goto ninja_build
)

    cmake %root_dir%  -G "%CMAKE_GENERATOR%" -A %BUILD_PLATFORM% ^
            -DMSVC_STATIC_CRT=%MSVC_STATIC_CRT% ^
            -DWITH_MKL=ON      ^
            -DWITH_MKLDNN=OFF   ^
            -DWITH_AVX=%WITH_AVX% ^
            -DCMAKE_BUILD_TYPE=%BUILD_TYPE% ^
            -DTHIRD_PARTY_BUILD_TYPE=Release ^
            -DLITE_WITH_X86=ON  ^
            -DLITE_WITH_PROFILE=%WITH_PROFILE% ^
            -DLITE_WITH_PRECISION_PROFILE=%WITH_PRECISION_PROFILE% ^
            -DWITH_LITE=ON ^
            -DLITE_WITH_XPU=%WITH_KUNLUNXIN_XPU% ^
            -DXPU_SDK_ROOT=%KUNLUNXIN_XPU_SDK_ROOT%  ^
            -DLITE_WITH_LIGHT_WEIGHT_FRAMEWORK=OFF ^
            -DLITE_WITH_ARM=OFF ^
            -DLITE_WITH_OPENCL=%WITH_OPENCL% ^
            -DLITE_BUILD_EXTRA=%BUILD_EXTRA% ^
            -DLITE_WITH_PYTHON=%WITH_PYTHON% ^
            -DWITH_TESTING=%WITH_TESTING%    ^
            -DLITE_WITH_LOG=%WITH_LOG%       ^
            -DWITH_STATIC_MKL=%WITH_STATIC_MKL%  ^
            -DLITE_BUILD_TAILOR=%WITH_STRIP%  ^
            -DLITE_OPTMODEL_DIR=%OPTMODEL_DIR%  ^
            -DPYTHON_EXECUTABLE="%python_path%"

if "%BUILD_FOR_CI%"=="ON" (
    call "%vcvarsall_dir%" amd64
    msbuild /m:%cores% /p:Configuration=%BUILD_TYPE% lite\lite_compile_deps.vcxproj
    call:test_server
    cmake ..   -G "%CMAKE_GENERATOR% Win64" -T host=x64 -DWITH_LITE=ON -DLITE_ON_MODEL_OPTIMIZE_TOOL=ON -DWITH_TESTING=OFF -DLITE_BUILD_EXTRA=ON
    msbuild /m:%cores% /p:Configuration=%BUILD_TYPE% lite\api\opt.vcxproj
) else if "%BUILD_PLATFORM%"=="x64" (
    call "%vcvarsall_dir%" amd64
    msbuild /maxcpucount:%cores% /p:Configuration=%BUILD_TYPE% lite\publish_inference.vcxproj
) else (
    call "%vcvarsall_dir%" x86
    msbuild /maxcpucount:%cores% /p:Configuration=%BUILD_TYPE% lite\publish_inference.vcxproj
)
goto:eof

:ninja_build
    if "%BUILD_PLATFORM%"=="x64" (
        call "%vcvarsall_dir%" amd64
        set ARCH="amd64"
    ) else (
        call "%vcvarsall_dir%" x86
        set ARCH="i386"
    )

    cmake %root_dir%  -G Ninja -DARCH=%ARCH% ^
            -DMSVC_STATIC_CRT=%MSVC_STATIC_CRT% ^
            -DWITH_MKL=ON      ^
            -DWITH_MKLDNN=OFF   ^
            -DWITH_AVX=%WITH_AVX% ^
            -DLITE_WITH_X86=ON  ^
            -DLITE_WITH_PROFILE=%WITH_PROFILE% ^
            -DLITE_WITH_PRECISION_PROFILE=%WITH_PRECISION_PROFILE% ^
            -DWITH_LITE=ON ^
            -DLITE_WITH_XPU=%WITH_KUNLUNXIN_XPU% ^
            -DXPU_SDK_ROOT=%KUNLUNXIN_XPU_SDK_ROOT%  ^
            -DCMAKE_BUILD_TYPE=%BUILD_TYPE% ^
            -DTHIRD_PARTY_BUILD_TYPE=Release ^
            -DLITE_WITH_LIGHT_WEIGHT_FRAMEWORK=OFF ^
            -DLITE_WITH_ARM=OFF ^
            -DLITE_WITH_OPENCL=%WITH_OPENCL% ^
            -DLITE_BUILD_EXTRA=%BUILD_EXTRA% ^
            -DLITE_WITH_PYTHON=%WITH_PYTHON% ^
            -DWITH_TESTING=%WITH_TESTING%    ^
            -DLITE_WITH_LOG=%WITH_LOG%       ^
            -DWITH_STATIC_MKL=%WITH_STATIC_MKL%  ^
            -DLITE_BUILD_TAILOR=%WITH_STRIP%  ^
            -DLITE_OPTMODEL_DIR=%OPTMODEL_DIR%  ^
            -DPYTHON_EXECUTABLE="%python_path%"

    ninja extern_mklml
    ninja publish_inference -j %cores%
goto:eof


:prepare_thirdparty
    if  EXIST "%workspace%\third-party" (
        if NOT EXIST "%workspace%\%THIRDPARTY_TAR%" (
            echo "The directory of third_party exists, %THIRDPARTY_TAR% not exists."
            git submodule update --init --recursive
            call:rm_rebuild_dir "%workspace%\third-party\glog\src\extern_glog-build"
            call:rm_rebuild_dir "%workspace%\third-party\protobuf-host\src\extern_protobuf-build"

        ) else (
               echo "The directory of third_party exists, the %THIRDPARTY_TAR% exists."
               call:rm_rebuild_dir "%workspace%\third-party"
               !python_path! %workspace%\lite\tools\untar.py %source_path%\%THIRDPARTY_TAR% %workspace%
        )
    ) else (
        if NOT EXIST "%workspace%\%THIRDPARTY_TAR%" (
            echo "The directory of third_party not exists, the %THIRDPARTY_TAR% not exists."
            call:download_third_party
            if EXIST "%workspace%\%THIRDPARTY_TAR%" (
                !python_path! %workspace%\lite\tools\untar.py %source_path%\%THIRDPARTY_TAR% %workspace%
            ) else (
                echo "------------Can't download the %THIRDPARTY_TAR%!------"
            )
        ) else (
            echo "The directory of third_party not exists, the %THIRDPARTY_TAR% exists."
            !python_path! %workspace%\lite\tools\untar.py %source_path%\%THIRDPARTY_TAR% %workspace%
        )
    )
goto:eof

:download_third_party
powershell.exe (new-object System.Net.WebClient).DownloadFile('%THIRDPARTY_TAR%', ^
'%workspace%\%THIRDPARTY_TAR%')
goto:eof

:prepare_opencl_source_code
    set GEN_CODE_PATH_OPENCL=%root_dir%\lite\backends\opencl
    del /f /s /q "%GEN_CODE_PATH_OPENCL%\opencl_kernels_source.cc" >nul 2>&1
    set OPENCL_KERNELS_PATH=%root_dir%\lite\backends\opencl\cl_kernel
    md "%GEN_CODE_PATH_OPENCL%"
    type nul > "%GEN_CODE_PATH_OPENCL%\opencl_kernels_source.cc"
    !python_path! %root_dir%\lite\tools\cmake_tools/gen_opencl_code.py %OPENCL_KERNELS_PATH% %GEN_CODE_PATH_OPENCL%\opencl_kernels_source.cc
goto:eof


:rm_rebuild_dir
    del /f /s /q "%~1\*.*"  >nul 2>&1
    rd /s /q  "%~1" >nul 2>&1
goto:eof

:set_vcvarsall_dir
SET /P vcvarsall_dir="Please input the path of visual studio command Prompt, such as %vcvarsall_dir%   =======>"
set tmp_var=!vcvarsall_dir!
call:remove_space
set vcvarsall_dir=!tmp_var!
IF NOT EXIST "%vcvarsall_dir%" (
    echo "------------%vcvarsall_dir% not exist------------"
    goto:eof
)
goto:eof

:set_python_path
set python_path=C:\Python35\python.exe
IF NOT EXIST "%python_path%" (
    goto input_python_path
)
SET /P answer="We checked that %python_path% exists. Using this python path[yes/no]? Type yes will use the python path, while type no will let you input your prefer python path:"
set tmp_var=%answer%
call:remove_space
if "%tmp_var%"=="yes" (
    goto:eof
) else (
:input_python_path
    SET /P python_path="Please input the path of python.exe, such as C:\Python35\python.exe, C:\Python35\python3.exe  ======>"
    set tmp_var=%python_path%
    call:remove_space
    set python_path=%tmp_var%
    if "%python_path%"=="" (
        set python_path=python.exe
    ) else (
        IF NOT EXIST "%python_path%" (
            echo "------------%python_path% not exist---------------"
        )
    )
)
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

:print_usage
echo "------------------------------------------------------------------------------------------------------|"
echo "|  Methods of compiling Paddle-lite Windows library:                                                  |"
echo "|-----------------------------------------------------------------------------------------------------|"
echo "|  compile windows library: ( x86 )                                                                   |"
echo "|      build_windows.bat                                                                              |"
echo "|  print help information:                                                                            |"
echo "|      build_windows.bat help                                                                         |"
echo "|                                                                                                     |"
echo "|  optional argument:                                                                                 |"
echo "|      without_log: Disable print log information. Default  ON.                                       |"
echo "|      without_python: Disable Python api lib in lite mode. Default ON.                               |"
echo "|      with_profile: Enable profile mode in lite framework. Default  OFF.                             |"
echo "|      with_extra: Enable extra algorithm support in Lite, both kernels and operators. Default OFF.   |"
echo "|      with_strip: Enable tailoring library according to model. Default OFF.                          |"
echo "|      build_x86: Enable building for Windows x86 platform. Default is x64.                           |"
echo "|      with_dynamic_crt: Enable building for MSVC Dynamic Runtime. Default is Static.                 |"
echo "|      with_static_mkl: Enable Static linking Intel(R) MKL. Default is Dynamic.                       |"
echo "|      with_opencl: Enable OpenCL for GPU accelerator. Default OFF.                                   |"
echo "|      without_avx: Enable AVX or SSE for X86 kernels. Default is ON.                                 |"
echo "|      use_ninja: Enable ninja build. Default is OFF.                                                 |"
echo "|      use_vs2017: Enable visual studio 2017 build. Default is OFF.                                   |"
echo "|      use_vs2019: Enable visual studio 2019 build. Default is OFF.                                   |"
echo "|  for example:                                                                                       |"
echo "|      build_windows.bat with_log with_profile with_python with_extra                                 |"
echo "|      build_windows.bat build_x86 with_strip D:\Paddle-Lite\opt_model_dir                            |"
echo "------------------------------------------------------------------------------------------------------|"
goto:eof

:test_server
    rem Due to the missing of x86 kernels, we skip the following tests temporarily.
    rem TODO(xxx) clear the skip list latter
    set skip_list=("test_paddle_api" "test_cxx_api" "test_light_api" "test_apis" "test_model_bin")

    for /f %%a in ('type %test_file%') do (
        set to_skip=0
        for %%b in %skip_list% do (
            if "%%a"==%%b (
                set to_skip=1
                echo "to skip %%a"
            )
        )
        if !to_skip! EQU 0 (
            echo "Run the test of %%a"
            ctest -C Release -R %%a

        )
    )
goto:eof
