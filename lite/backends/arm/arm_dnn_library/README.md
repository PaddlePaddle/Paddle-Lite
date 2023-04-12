# arm_dnn_library

arm_dnn_library is a Arm NEON accelerated library for neural network inference for Paddle and Paddle-Lite.

## Supported platforms

- Armv7 and Armv8 on Android, iOS, macOS, Linux and Windows.

## Supported operators

- relu
- concat

## How to build ?
### For Android,
- Preparing build environment.
  - On x86+Linux,
  ```
  1) Install build-essentials.
    $ apt update
    $ apt-get install -y --no-install-recommends gcc g++ git make wget python unzip adb curl

  2) Install CMake 3.10.3.
    $ wget -c https://mms-res.cdn.bcebos.com/cmake-3.10.3-Linux-x86_64.tar.gz && \
      tar xzf cmake-3.10.3-Linux-x86_64.tar.gz && \
      mv cmake-3.10.3-Linux-x86_64 /opt/cmake-3.10 &&
      ln -s /opt/cmake-3.10/bin/cmake /usr/bin/cmake && \
      ln -s /opt/cmake-3.10/bin/ccmake /usr/bin/ccmake

  3) Install Android NDK.
    $ cd /tmp && curl -O https://dl.google.com/android/repository/android-ndk-r17c-linux-x86_64.zip
    $ cd /opt && unzip /tmp/android-ndk-r17c-linux-x86_64.zip
    $ echo "export ANDROID_NDK=/opt/android-ndk-r17c" >> ~/.bashrc
    $ source ~/.bashrc

  4) Remove debug flags in android.toolchain.cmake, please refer to https://github.com/android/ndk/issues/243.
    Edit $ANDROID_NDK/build/cmake/android.toolchain.cmake or $ANDROID_NDK/build/cmake/android-legacy.toolchain.cmake after Android NDK r23c.
    list(APPEND ANDROID_COMPILER_FLAGS
      # Remove debug flags -g
      -DANDROID
      ...
  ```
  - On x86+macOS,
  ```
  1) Install build-essentials.
    $ brew install curl gcc git make unzip wget

  2) Install CMake 3.10.3.
    $ mkdir /usr/local/Cellar/cmake/ && cd /usr/local/Cellar/cmake/
    $ wget https://cmake.org/files/v3.10/cmake-3.10.2-Darwin-x86_64.tar.gz
    $ tar zxf ./cmake-3.10.2-Darwin-x86_64.tar.gz
    $ mv cmake-3.10.2-Darwin-x86_64/CMake.app/Contents/ ./3.10.2
    $ ln -s /usr/local/Cellar/cmake/3.10.2/bin/cmake /usr/local/bin/cmake

  3) Install Android NDK.
    $ cd ~/Documents && curl -O https://dl.google.com/android/repository/android-ndk-r17c-darwin-x86_64.zip
    $ cd ~/Library && unzip ~/Documents/android-ndk-r17c-darwin-x86_64.zip
    $ echo "export ANDROID_NDK=~/Library/android-ndk-r17c" >> ~/.bash_profile
    $ source ~/.bash_profile

  4) Remove debug flags in android.toolchain.cmake, please refer to https://github.com/android/ndk/issues/243.
    Edit $ANDROID_NDK/build/cmake/android.toolchain.cmake or $ANDROID_NDK/build/cmake/android-legacy.toolchain.cmake after Android NDK r23c.
    list(APPEND ANDROID_COMPILER_FLAGS
      # Remove debug flags -g
      -DANDROID
      ...
  ```
- Build for armeabi-v7a,
  ```
  $ ./scripts/build_android.sh --arch=armv7
  ```

- Build for arm64-v8a,
  ```
  $ ./scripts/build_android.sh
  ```

### For Linux,
- Preparing build environment.
  - On x86+Linux,
  ```
  1) Install build-essentials.
    $ apt update
    $ apt-get install -y --no-install-recommends gcc g++ git make wget python unzip g++-arm-linux-gnueabi gcc-arm-linux-gnueabi g++-arm-linux-gnueabihf gcc-arm-linux-gnueabihf gcc-aarch64-linux-gnu g++-aarch64-linux-gnu

  2) Install CMake 3.10.3.
    $ wget -c https://mms-res.cdn.bcebos.com/cmake-3.10.3-Linux-x86_64.tar.gz && \
      tar xzf cmake-3.10.3-Linux-x86_64.tar.gz && \
      mv cmake-3.10.3-Linux-x86_64 /opt/cmake-3.10 &&
      ln -s /opt/cmake-3.10/bin/cmake /usr/bin/cmake && \
      ln -s /opt/cmake-3.10/bin/ccmake /usr/bin/ccmake
  ```
  - on Arm+Linux,
  ```
  1) Install build-essentials.
    $ apt update
    $ apt-get install -y --no-install-recommends gcc g++ make wget python unzip patchelf

  2) Install CMake 3.10.3 or above.
    $ wget https://www.cmake.org/files/v3.10/cmake-3.10.3.tar.gz
    $ tar -zxvf cmake-3.10.3.tar.gz
    $ cd cmake-3.10.3
    $ ./configure
    $ make
    $ sudo make install
  ```
- Build for armhf,
  ```
  $ ./scripts/build_linux.sh --arch=armv7
  ```

- Build for arm64,
  ```
  $ ./scripts/build_linux.sh
  ```
