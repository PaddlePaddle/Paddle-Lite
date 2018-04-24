#!/bin/bash

build_for_linux() {
    echo "linux"
}

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
    CXX_FLAGS="-std=c++11 -O3 -s"
    BUILD_DIR=build/release/"${PLATFORM}"
    mkdir -p ${BUILD_DIR}/build

    cmake . \
        -B"${BUILD_DIR}" \
    	-DCMAKE_BUILD_TYPE="${MODE}" \
    	-DCMAKE_CXX_FLAGS="${CXX_FLAGS}" \
    	-DIS_MAC=true

    cd ${BUILD_DIR}
    make -j 8
}

build_for_android() {
    if [ -z "${NDK_ROOT}" ]; then
        echo "NDK_ROOT not found!"
        exit -1
    fi

#    PLATFORM="arm-v7a"
    PLATFORM="arm-v8a"

    if [ "${PLATFORM}" = "arm-v7a" ]; then
        ABI="armeabi-v7a with NEON"
        ARM_PLATFORM="V7"
        CXX_FLAGS="-O3 -std=c++11 -s"
    elif [ "${PLATFORM}" = "arm-v8a" ]; then
        ABI="arm64-v8a"
        ARM_PLATFORM="V8"
        CXX_FLAGS="-O3 -std=c++11 -s"
    else
        echo "unknown platform!"
        exit -1
    fi

    MODE="Release"
    ANDROID_PLATFORM_VERSION="android-15"
    TOOLCHAIN_FILE="./android-cmake/android.toolchain.cmake"
    ANDROID_ARM_MODE="arm"

    cmake . \
        -B"build/release/${PLATFORM}" \
        -DANDROID_ABI="${ABI}" \
        -DCMAKE_BUILD_TYPE="${MODE}" \
        -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_FILE}" \
        -DANDROID_PLATFORM="${ANDROID_PLATFORM_VERSION}" \
        -DCMAKE_CXX_FLAGS="${CXX_FLAGS}" \
        -DANDROID_STL=c++_static \
        -DANDROID=true \
        -D"${ARM_PLATFORM}"=true

    cd "./build/release/${PLATFORM}"
    make -j 8
}

build_for_ios() {
    PLATFORM="ios"
    MODE="Release"
    BUILD_DIR=build/release/"${PLATFORM}"
    TOOLCHAIN_FILE="./ios-cmake/ios.toolchain.cmake"
    C_FLAGS="-fobjc-abi-version=2 -fobjc-arc -isysroot ${CMAKE_OSX_SYSROOT}"
    CXX_FLAGS="-fobjc-abi-version=2 -fobjc-arc -std=gnu++11 -stdlib=libc++ -isysroot ${CMAKE_OSX_SYSROOT}"
    mkdir -p "${BUILD_DIR}"

    cmake . \
        -B"${BUILD_DIR}" \
        -DCMAKE_BUILD_TYPE="${MODE}" \
        -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_FILE}" \
        -DIOS_PLATFORM=OS \
        -DCMAKE_C_FLAGS="${C_FLAGS}" \
        -DCMAKE_CXX_FLAGS="${CXX_FLAGS}" \
        -DIS_IOS="true" \

    cd "${BUILD_DIR}"
    make -j 8
}

build_error() {
    echo "unknown argument"
}

if [ $# -lt 1 ]; then
	echo "error: target missing!"
    echo "available targets: mac|linux|ios|android"
    echo "sample usage: ./build.sh mac"
else
	if [ $1 = "mac" ]; then
		build_for_mac
	elif [ $1 = "linux" ]; then
		build_for_linux
	elif [ $1 = "android" ]; then
		build_for_android
	elif [ $1 = "ios" ]; then
		build_for_ios
	else
		build_error
	fi
fi