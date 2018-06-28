FROM ubuntu:16.04

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
        curl \
        unzip \
        git \
        make \
        cmake \
        cmake-curses-gui \
        python \
        python-pip \
        python-setuptools \
        clang-format-5.0 \
        graphviz \
        g++-arm-linux-gnueabi \
        gcc-arm-linux-gnueabi
RUN apt-get autoremove -y && apt-get clean
RUN pip install --upgrade pip
RUN pip install wheel && pip install pre-commit
RUN ln -s clang-format-5.0 /usr/bin/clang-format
# RUN cd /tmp && curl -O http://mirrors.neusoft.edu.cn/android/repository/android-ndk-r17b-linux-x86_64.zip
# RUN cd /opt && unzip /tmp/android-ndk-r17b-linux-x86_64.zip
# ENV NDK_ROOT /opt/android-ndk-r17b
