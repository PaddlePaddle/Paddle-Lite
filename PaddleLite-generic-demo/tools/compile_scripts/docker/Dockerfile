# A image to cross compile the libraries for PaddleLite-generic-demo
FROM ubuntu:16.04
MAINTAINER PaddlePaddle Authors <paddle-dev@baidu.com>

RUN cp -a /etc/apt/sources.list /etc/apt/sources.list.bak \
    && sed -i "s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g" /etc/apt/sources.list \
    && apt-get -y update
# Cross compiler for Arm Linux
RUN apt-get install -y software-properties-common
RUN add-apt-repository -y ppa:ubuntu-toolchain-r/test
RUN apt-get update
RUN apt-get install -y --no-install-recommends \
        build-essential \
        cmake-curses-gui \
        curl \
        fish \
        gawk \
        gcc-7 \
        g++-7 \
        git \
        graphviz \
        less \
        make \
        patch \
        python \
        python-pip \
        python-setuptools \
        unzip \
        vim \
        wget \
        openssh-client \
        sshpass \
        openjdk-8-jdk \
        g++-arm-linux-gnueabi \
        gcc-arm-linux-gnueabi \
        g++-arm-linux-gnueabihf \
        gcc-arm-linux-gnueabihf \
        gcc-aarch64-linux-gnu \
        g++-aarch64-linux-gnu

RUN ln -fs /usr/bin/gcc-7 /usr/bin/gcc
RUN ln -fs /usr/bin/g++-7 /usr/bin/g++

# Fix the installation of clang-format-3.8 in Ubuntu 18.04
# RUN apt-get install -y --no-install-recommends gpg-agent
# RUN wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add -
# RUN apt-add-repository "deb http://apt.llvm.org/trusty/ llvm-toolchain-trusty-3.8 main" && apt-get -y update
RUN apt-get install -y --no-install-recommends clang-format-3.8 
RUN ln -s clang-format-3.8 /usr/bin/clang-format
RUN pip install --upgrade pip==20.3.4
RUN pip install wheel
RUN pip install pre-commit

# Set timezone
RUN DEBIAN_FRONTEND=noninteractive apt install -y tzdata
RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/timezone && echo 'Asia/Shanghai' >/etc/timezone
RUN dpkg-reconfigure -f noninteractive tzdata

# Download both cmake 3.10 and 3.22.3, but cmake 3.10 is used by default
RUN curl -L http://paddlelite-demo.bj.bcebos.com/devices/generic/tools/cmake-3.10.3-Linux-x86_64.tar.gz -o - | tar -xz -C /opt/
RUN curl -L http://paddlelite-demo.bj.bcebos.com/devices/generic/tools/cmake-3.22.3-linux-x86_64.tar.gz -o - | tar -xz -C /opt/
RUN rm /usr/bin/cmake && ln -s /opt/cmake-3.10.3-Linux-x86_64/bin/cmake /usr/bin/cmake && \
    rm /usr/bin/ccmake && ln -s /opt/cmake-3.10.3-Linux-x86_64/bin/ccmake /usr/bin/ccmake

# Cross compiler for Android
# Download both android ndk r17c and r20b, but r17c is used by default
RUN curl -O http://paddlelite-demo.bj.bcebos.com/devices/generic/tools/android-ndk-r17c-linux-x86_64.zip && unzip android-ndk-r17c-linux-x86_64.zip -d /opt/
RUN curl -O http://paddlelite-demo.bj.bcebos.com/devices/generic/tools/android-ndk-r20b-linux-x86_64.zip && unzip android-ndk-r20b-linux-x86_64.zip -d /opt/
ENV NDK_ROOT /opt/android-ndk-r17c

# Android platform tools and Android-24 SDK
RUN curl -O http://paddlelite-demo.bj.bcebos.com/devices/generic/tools/sdk-tools-linux-4333796-android-24.zip && unzip sdk-tools-linux-4333796-android-24.zip -d /opt/
ENV ANDROID_HOME /opt/android_sdk
ENV PATH $PATH:${ANDROID_HOME}/tools:${ANDROID_HOME}/platform-tools:${ANDROID_HOME}/tools/bin
RUN mkdir /root/.android && touch /root/.android/repositories.cfg

# Clean
RUN apt-get autoremove -y && apt-get clean
RUN rm -rf /android-ndk-r17c-linux-x86_64.zip /android-ndk-r20b-linux-x86_64.zip /sdk-tools-linux-4333796-android-24.zip
