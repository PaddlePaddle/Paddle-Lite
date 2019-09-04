#!/usr/bin/env bash
NETS=""
declare -a supportedNets=("googlenet" "mobilenet" "yolo" "squeezenet" "resnet" "mobilenetssd" "nlp" "mobilenetfssd" "genet" "super" "op")

# merge cl to so
merge_cl_to_so=1
opencl_kernels="opencl_kernels.cpp"
cd ../src/operators/kernel/cl
if [[ -f "${opencl_kernels}" ]]; then
    rm "${opencl_kernels}"
fi
python gen_code.py "${merge_cl_to_so}" > "${opencl_kernels}"
cd -

build_for_mac() {
    if [ ! `which brew` ]; then
        echo "building failed! homebrew not found, please install homebrew."
        return
    fi
    if [ ! `which cmake` ]; then
        echo "installing cmake."
        brew install cmake
        if [ ! $? ]; then
            echo "cmake install failed."
            return
        fi
    fi
    PLATFORM="x86"
    MODE="Release"
    BUILD_DIR=../build/release/"${PLATFORM}"
    mkdir -p ${BUILD_DIR}/build

    mkdir -p ${BUILD_DIR}/test
    cp -r ../test/models ${BUILD_DIR}/test/models

    cmake .. \
        -B"${BUILD_DIR}" \
    	-DCMAKE_BUILD_TYPE="${MODE}" \
    	-DIS_MAC=true

    cd ${BUILD_DIR}
    make -j 8
}

build_for_android() {
    # rm -rf "../build"
    if [ -z "${NDK_ROOT}" ]; then
        echo "NDK_ROOT not found!"
        exit -1
    fi

    if [ -z "$PLATFORM" ]; then
        PLATFORM="arm-v7a"  # Users could choose "arm-v8a" platform.
        # PLATFORM="arm-v8a"
    fi

    if [ "${PLATFORM}" = "arm-v7a" ]; then
        ABI="armeabi-v7a with NEON"
        ARM_PLATFORM="V7"
        CXX_FLAGS="-march=armv7-a -mfpu=neon -mfloat-abi=softfp -pie -fPIE -w -Wno-error=format-security"
    elif [ "${PLATFORM}" = "arm-v8a" ]; then
        ABI="arm64-v8a"
        ARM_PLATFORM="V8"
        CXX_FLAGS="-march=armv8-a  -pie -fPIE -w -Wno-error=format-security -llog"
    else
        echo "unknown platform!"
        exit -1
    fi


    MODE="Release"
    ANDROID_PLATFORM_VERSION="android-19"
    TOOLCHAIN_FILE="./tools/android-cmake/android.toolchain.cmake"
    ANDROID_ARM_MODE="arm"

    if [ "${#NETS}" -gt 1 ]; then
    cmake .. \
        -B"../build/release/${PLATFORM}" \
        -DANDROID_ABI="${ABI}" \
        -DCMAKE_BUILD_TYPE="${MODE}" \
        -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_FILE}" \
        -DANDROID_PLATFORM="${ANDROID_PLATFORM_VERSION}" \
        -DCMAKE_CXX_FLAGS="${CXX_FLAGS}" \
        -DANDROID_STL=c++_static \
        -DANDROID=true \
        -DNET="${NETS}" \
        -D"${ARM_PLATFORM}"=true
    else

    cmake .. \
        -B"../build/release/${PLATFORM}" \
        -DANDROID_ABI="${ABI}" \
        -DCMAKE_BUILD_TYPE="${MODE}" \
        -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_FILE}" \
        -DANDROID_PLATFORM="${ANDROID_PLATFORM_VERSION}" \
        -DCMAKE_CXX_FLAGS="${CXX_FLAGS}" \
        -DANDROID_STL=c++_static \
        -DANDROID=true \
        -D"${ARM_PLATFORM}"=true
    fi
    cd "../build/release/${PLATFORM}"
    make -j 8
    mkdir ./build/cl_kernel
    cp ../../../src/operators/kernel/cl/cl_kernel/*  ./build/cl_kernel/
}

build_for_arm_linux() {
    MODE="Release"
    ARM_LINUX="arm-linux"

    if [ "${#NETS}" -gt 1 ]; then
        cmake .. \
            -B"../build/release/arm-linux" \
            -DCMAKE_BUILD_TYPE="${MODE}" \
            -DCMAKE_TOOLCHAIN_FILE="./tools/toolchains/arm-linux-gnueabihf.cmake" \
            -DCMAKE_CXX_FLAGS="-std=c++14 -mcpu=cortex-a53 -mtune=cortex-a53 -ftree-vectorize -funsafe-math-optimizations  -pipe -mlittle-endian " \
            -DNET="${NETS}" \
            -D"V7"=true
    else
        cmake .. \
            -B"../build/release/arm-linux" \
            -DCMAKE_BUILD_TYPE="${MODE}" \
            -DCMAKE_TOOLCHAIN_FILE="./tools/toolchains/arm-linux-gnueabihf.cmake" \
            -DCMAKE_CXX_FLAGS="-std=c++14 -mcpu=cortex-a53 -mtune=cortex-a53 -ftree-vectorize -funsafe-math-optimizations -pipe -mlittle-endian " \
            -DNET="${NETS}" \
            -D"V7"=true
    fi

    cd "../build/release/arm-linux"
    make -j 2

    cd "../../../test/"
    DIRECTORY="models"
    if [ "`ls -A $DIRECTORY`" = "" ]; then
        echo "$DIRECTORY is indeed empty pull images"
        wget http://mms-graph.bj.bcebos.com/paddle-mobile%2FmodelsAndImages.zip
        unzip paddle-mobile%2FmodelsAndImages.zip
        mv modelsAndImages/images/ images
        mv modelsAndImages/models/ models
        rm -rf paddle-mobile%2FmodelsAndImages.zip
        rm -rf __MACOS
    else
        echo "$DIRECTORY is indeed not empty, DONE!"
    fi

}

build_for_ios() {
#    rm -rf "../build"
    PLATFORM="ios"
    MODE="Release"
    BUILD_DIR=../build/release/"${PLATFORM}"/
    TOOLCHAIN_FILE="./tools/ios-cmake/ios.toolchain.cmake"
    mkdir -p "${BUILD_DIR}"
    if [ "${#NETS}" -gt 1 ]; then
        cmake .. \
            -B"${BUILD_DIR}" \
            -DCMAKE_BUILD_TYPE="${MODE}" \
            -DIOS_PLATFORM=OS \
            -DIOS_ARCH="${IOS_ARCH}" \
            -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_FILE}" \
            -DNET="${NETS}" \
            -DIS_IOS="true"
    else
        cmake .. \
            -B"${BUILD_DIR}" \
            -DCMAKE_BUILD_TYPE="${MODE}" \
            -DIOS_PLATFORM=OS \
            -DIOS_ARCH="${IOS_ARCH}" \
            -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_FILE}" \
            -DIS_IOS="true"
    fi
    cd "${BUILD_DIR}"
    make -j 8
    cp ../../../src/io/ios_io/PaddleMobileCPU.h ./build/PaddleMobileCPU.h
    cd ./build
    # 生成符号表
    ranlib *.a
}

build_error() {
    echo "unknown target : $1"
}

if [ $# -lt 1 ]; then
    echo "error: target missing!"
    echo "available targets: ios|android"
    echo "sample usage: ./build.sh android"
else
    params=($@)
    for(( i=1; i<$#; i++ )); do
        if [ ${i} != 1 ]; then
            NETS=$NETS$";"
        fi
        NETS=$NETS$"${params[i]}"
    done
    params=${@:2}

    supported=false
    for name in ${params[@]}; do
        for net in ${supportedNets[@]}; do
            match=false
            if [ "$name"x = "$net"x ];then
                supported=true
                match=true
                break 1
            fi
        done
        if [ "$match" = false ];then
            echo "${name} not supported!"
            echo "supported nets are: ${supportedNets[@]}"
            exit -1
        fi
    done

    if [ $1 = "android" ]; then
        build_for_android
    elif [ $1 = "arm_linux" ]; then
        build_for_arm_linux
    elif [ $1 = "ios" ]; then
        build_for_ios
    else
        build_error "$1"
    fi
fi
