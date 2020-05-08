#!/usr/bin/env bash

# merge cl to so
merge_cl_to_so=1
opencl_kernels="opencl_kernels.cpp"
cd ../src/operators/kernel/cl
if [[ -f "${opencl_kernels}" ]]; then
    rm "${opencl_kernels}"
fi
python gen_code.py "${merge_cl_to_so}" >"${opencl_kernels}"
cd -

# get cl headers
opencl_header_dir="../third_party/opencl/OpenCL-Headers"
commit_id="320d7189b3e0e7b6a8fc5c10334c79ef364b5ef6"
if [[ -d "$opencl_header_dir" && -d "$opencl_header_dir/.git" ]]; then
    echo "pulling opencl headers"
    cd $opencl_header_dir
    git stash
    git pull
    git checkout $commit_id
    cd -
else
    echo "cloning opencl headers"
    rm -rf $opencl_header_dir
    git clone https://github.com/KhronosGroup/OpenCL-Headers $opencl_header_dir
    git checkout $commit_id
fi

build_for_android() {
    # rm -rf "../build"
    if [ -z "${NDK_ROOT}" ]; then
        echo "NDK_ROOT not found!"
        exit -1
    fi

    if [ -z "$PLATFORM" ]; then
        # PLATFORM="arm-v7a" # Users could choose "arm-v8a" platform.
        PLATFORM="arm-v8a"
    fi

    if [ "${PLATFORM}" = "arm-v7a" ]; then
        ABI="armeabi-v7a with NEON"
        ARM_PLATFORM="V7"
        CXX_FLAGS="-march=armv7-a -mfpu=neon -mfloat-abi=softfp -pie -fPIE -w -Wno-error=format-security"
    elif [ "${PLATFORM}" = "arm-v8a" ]; then
        ABI="arm64-v8a"
        ARM_PLATFORM="V8"
        CXX_FLAGS="-march=armv8-a  -pie -fPIE -w -Wno-error=format-security -llog -fuse-ld=gold"
    else
        echo "unknown platform!"
        exit -1
    fi

    MODE="Release"
    ANDROID_PLATFORM_VERSION="android-19"
    TOOLCHAIN_FILE="./tools/android-cmake/android.toolchain.cmake"
    ANDROID_ARM_MODE="arm"

    cmake .. \
        -B"../buildreleasev8/${PLATFORM}" \
        -DANDROID_ABI="${ABI}" \
        -DCMAKE_BUILD_TYPE="${MODE}" \
        -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_FILE}" \
        -DANDROID_PLATFORM="${ANDROID_PLATFORM_VERSION}" \
        -DCMAKE_CXX_FLAGS="${CXX_FLAGS}" \
        -DANDROID_STL=c++_static \
        -DANDROID=true \
        -DWITH_LOGGING=OFF \
        -DWITH_PROFILE=OFF \
        -DWITH_TEST=OFF \
        -D"${ARM_PLATFORM}"=true

    cd "../buildreleasev8/${PLATFORM}"
    make -j 8
}

build_for_android
