#!/bin/sh
# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

tools/codegen.py src/operators/relu/fp32_neon.cc.in src/operators/relu/codegen/fp32_aarch64_neon_x16.cc -D BATCH_TILE=16 -D ARCH=aarch64 &
tools/codegen.py src/operators/relu/fp32_neon.cc.in src/operators/relu/codegen/fp32_aarch32_neon_x8.cc -D BATCH_TILE=8 -D ARCH=aarch32 &

wait
