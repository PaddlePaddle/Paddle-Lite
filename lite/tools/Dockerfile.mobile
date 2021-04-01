# A image for paddle lite mobile cross compile and simulator on android

FROM ubuntu:16.04
MAINTAINER PaddlePaddle Authors <paddle-dev@baidu.com>

RUN echo '\
deb <mirror> <version> main restricted universe multiverse\n\
deb <mirror> <version>-updates main restricted universe multiverse\n\
deb <mirror> <version>-backports main restricted universe multiverse\n\
deb <mirror> <version>-security main restricted universe multiverse\n'\
> /etc/apt/sources.list
RUN sed -ie 's|<mirror>|http://mirrors.tuna.tsinghua.edu.cn/ubuntu/|' /etc/apt/sources.list
RUN sed -ie 's|<version>|xenial|' /etc/apt/sources.list

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y --no-install-recommends \
        clang-format-3.8 \
        curl \
        gcc \
        g++ \
        git \
        make \
        patchelf \
        python \
        android-tools-adb \
        python-dev \
        python-pip \
        python-setuptools \
        unzip \
        vim \
        wget

# for android simulator
RUN apt-get install -y --no-install-recommends \
        libc6-i386 \
        lib32stdc++6 \
        redir \
        iptables \
        openjdk-8-jre \
        default-jdk

# for cmake 3.10
RUN curl -O https://mms-res.cdn.bcebos.com/cmake-3.10.3-Linux-x86_64.tar.gz && \
        tar xzf cmake-3.10.3-Linux-x86_64.tar.gz && \
        mv cmake-3.10.3-Linux-x86_64 /opt/cmake-3.10 && \
        ln -s /opt/cmake-3.10/bin/cmake /usr/bin/cmake && \
        ln -s /opt/cmake-3.10/bin/ccmake /usr/bin/ccmake

# for arm linux compile
RUN apt-get install -y --no-install-recommends \
        g++-arm-linux-gnueabi \
        gcc-arm-linux-gnueabi \
        g++-arm-linux-gnueabihf \
        gcc-arm-linux-gnueabihf \
        gcc-aarch64-linux-gnu \
        g++-aarch64-linux-gnu 

# for android ndk17c and ndk20b
RUN cd /tmp && curl -O https://dl.google.com/android/repository/android-ndk-r17c-linux-x86_64.zip
RUN cd /tmp && curl -O https://dl.google.com/android/repository/android-ndk-r20b-linux-x86_64.zip
ENV NDK_ROOT /opt/android-ndk-r17c
ENV NDK_ROOT_R20B /opt/android-ndk-r20b
RUN cd /opt && unzip /tmp/android-ndk-r17c-linux-x86_64.zip
RUN cd /opt && unzip /tmp/android-ndk-r20b-linux-x86_64.zip

# VNC port
EXPOSE 5900

# clean
RUN ln -s clang-format-3.8 /usr/bin/clang-format
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple wheel
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pre-commit
RUN apt-get autoremove -y && apt-get clean
RUN rm -rf /sdk-tools-linux-4333796.zip /tmp/android-ndk-r17c-linux-x86_64.zip /cmake-3.10.3-Linux-x86_64.tar.gz
