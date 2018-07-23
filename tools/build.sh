#!/usr/bin/env bash

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
    #rm -rf "../build"
    if [ -z "${NDK_ROOT}" ]; then
        echo "NDK_ROOT not found!"
        exit -1
    fi

    if [ -z "$PLATFORM" ]; then
        PLATFORM="arm-v7a"  # Users could choose "arm-v8a" platform.
#        PLATFORM="arm-v8a"
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
    ANDROID_PLATFORM_VERSION="android-22"
    TOOLCHAIN_FILE="./tools/android-cmake/android.toolchain.cmake"
    ANDROID_ARM_MODE="arm"
    if [ $# -eq 1 ]; then
    cmake .. \
        -B"../build/release/${PLATFORM}" \
        -DANDROID_ABI="${ABI}" \
        -DCMAKE_BUILD_TYPE="${MODE}" \
        -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_FILE}" \
        -DANDROID_PLATFORM="${ANDROID_PLATFORM_VERSION}" \
        -DCMAKE_CXX_FLAGS="${CXX_FLAGS}" \
        -DANDROID_STL=c++_static \
        -DANDROID=true \
        -DNET=$1 \
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
}

build_for_ios() {
#    rm -rf "../build"
    PLATFORM="ios"
    MODE="Release"
#    IOS_ARCH="armv7"
#    IOS_ARCH="armv7s"
    IOS_ARCH="arm64"    # Users could choose "armv7" or "armv7s" platforms.
    BUILD_DIR=../build/release/"${PLATFORM}"/"${IOS_ARCH}"
    TOOLCHAIN_FILE="./tools/ios-cmake/ios.toolchain.cmake"
    mkdir -p "${BUILD_DIR}"
    if [ $# -eq 1 ]; then
        cmake .. \
            -B"${BUILD_DIR}" \
            -DCMAKE_BUILD_TYPE="${MODE}" \
            -DIOS_PLATFORM=OS \
            -DIOS_ARCH="${IOS_ARCH}" \
            -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_FILE}" \
            -DNET=$1 \
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
    cp ../../../src/ios_io/PaddleMobile.h ./build/PaddleMobile.h
    cd ./build
    # 生成符号表
    ranlib *.a
}

build_error() {
    echo "unknown argument"
}

if [ $# -lt 1 ]; then
	echo "error: target missing!"
    echo "available targets: ios|android"
    echo "sample usage: ./build.sh android"
else
    if [ $# -eq 2 ]; then
        if [ $2 != "googlenet" -a $2 != "mobilenet" -a $2 != "yolo" -a $2 != "squeezenet" -a $2 != "resnet" ]; then
	        if [ $1 = "android" ]; then
		        build_for_android
	        elif [ $1 = "ios" ]; then
		        build_for_ios
	        else
		        build_error
	        fi
        else
	        if [ $1 = "android" ]; then
		        build_for_android $2
	        elif [ $1 = "ios" ]; then
		        build_for_ios $2
	        else
		        build_error
	        fi
        fi
    else
	    if [ $1 = "android" ]; then
		    build_for_android
	    elif [ $1 = "ios" ]; then
		    build_for_ios
	    else
		    build_error
	    fi
	fi
fi