FROM ubuntu:18.04

RUN echo '\
deb <mirror> bionic main restricted universe multiverse\n\
deb <mirror> bionic-updates main restricted universe multiverse\n\
deb <mirror> bionic-backports main restricted universe multiverse\n\
deb <mirror> bionic-security main restricted universe multiverse\n'\
> /etc/apt/sources.list
RUN sed -ie 's|<mirror>|http://mirrors.tuna.tsinghua.edu.cn/ubuntu/|' /etc/apt/sources.list

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y --no-install-recommends \
        curl \
        unzip \
        git \
        cmake \
        python \
        python-pip \
        python-setuptools \
        clang-format-5.0 \
        graphviz
RUN apt-get autoremove -y && apt-get clean
RUN pip install wheel pre-commit
RUN pre-commit autoupdate
RUN ln -s clang-format-5.0 /usr/bin/clang-format
RUN cd /tmp && curl -O http://mirrors.neusoft.edu.cn/android/repository/android-ndk-r17b-linux-x86_64.zip
RUN cd /opt && unzip /tmp/android-ndk-r17b-linux-x86_64.zip