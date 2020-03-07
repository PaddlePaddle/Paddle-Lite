#!/usr/bin/env bash

apt-get update
apt-get install -y gcc g++ cmake

cd /workspace && mkdir build
cd build && cmake .. -DCPU=OFF -DGPU_CL=OFF -DFPGA=ON && make -j4
