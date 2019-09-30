// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// This file is modified according to
// https://github.com/ARM-software/ComputeLibrary
// * Copyright (c) 2017-2018 ARM Limited.
// *
// * SPDX-License-Identifier: MIT
// *
// * Permission is hereby granted, free of charge, to any person obtaining a
// copy
// * of this software and associated documentation files (the "Software"), to
// * deal in the Software without restriction, including without limitation the
// * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// * sell copies of the Software, and to permit persons to whom the Software is
// * furnished to do so, subject to the following conditions:
// *
// * The above copyright notice and this permission notice shall be included in
// all
// * copies or substantial portions of the Software.
// *
// * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM,
// * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE
// * SOFTWARE.

#pragma once

#define _DECLARE_SDOT_ELEMENT                                                \
  ".altmacro\n"                                                              \
  ".macro sdot opd:req, opn:req, opm:req\n"                                  \
  "local vd, vn, vm, h, l\n"                                                 \
  ".irp "                                                                    \
  "reg,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25," \
  "26,27,28,29,30,31\n"                                                      \
  ".ifeqs \"\\opd\",\"v\\reg\\.4s\"\n"                                       \
  ".set vd,\\reg\n"                                                          \
  ".endif\n"                                                                 \
  ".ifeqs \"\\opn\",\"v\\reg\\.16b\"\n"                                      \
  ".set vn,\\reg\n"                                                          \
  ".endif\n"                                                                 \
  ".irp idx,0,1,2,3\n"                                                       \
  ".ifeqs \"\\opm\",\"v\\reg\\.4b[\\idx\\]\"\n"                              \
  ".set vm,\\reg\n"                                                          \
  ".set h,\\idx / 2\n"                                                       \
  ".set l,\\idx %% 2\n"                                                      \
  ".endif\n"                                                                 \
  ".endr\n"                                                                  \
  ".endr\n"                                                                  \
  ".ifndef vd\n"                                                             \
  ".error \"Bad operand \\opd\"\n"                                           \
  ".exitm\n"                                                                 \
  ".endif\n"                                                                 \
  ".ifndef vn\n"                                                             \
  ".error \"Bad operand \\opn\"\n"                                           \
  ".exitm\n"                                                                 \
  ".endif\n"                                                                 \
  ".ifndef vm\n"                                                             \
  ".error \"Bad operand \\opm\"\n"                                           \
  ".exitm\n"                                                                 \
  ".endif\n"                                                                 \
  ".ifndef h\n"                                                              \
  ".error \"Bad operand \\opm\"\n"                                           \
  ".exitm\n"                                                                 \
  ".endif\n"                                                                 \
  ".ifndef l\n"                                                              \
  ".error \"Bad operand \\opm\"\n"                                           \
  ".exitm\n"                                                                 \
  ".endif\n"                                                                 \
  ".int  0x4f80e000 | vd | (vn << 5) | (vm << 16) | (l << 21) | (h << 11)\n" \
  ".endm\n"

#define _DECLARE_SDOT_VECTOR                                                 \
  ".altmacro\n"                                                              \
  ".macro sdot opd:req, opn:req, opm:req\n"                                  \
  "local vd, vn, vm\n"                                                       \
  ".irp "                                                                    \
  "reg,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25," \
  "26,27,28,29,30,31\n"                                                      \
  ".ifeqs \"\\opd\",\"v\\reg\\.4s\"\n"                                       \
  ".set vd,\\reg\n"                                                          \
  ".endif\n"                                                                 \
  ".ifeqs \"\\opn\",\"v\\reg\\.16b\"\n"                                      \
  ".set vn,\\reg\n"                                                          \
  ".endif\n"                                                                 \
  ".ifeqs \"\\opm\",\"v\\reg\\.16b\"\n"                                      \
  ".set vm,\\reg\n"                                                          \
  ".endif\n"                                                                 \
  ".endr\n"                                                                  \
  ".endr\n"                                                                  \
  ".ifndef vd\n"                                                             \
  ".error \"Bad operand \\opd\"\n"                                           \
  ".exitm\n"                                                                 \
  ".endif\n"                                                                 \
  ".ifndef vn\n"                                                             \
  ".error \"Bad operand \\opn\"\n"                                           \
  ".exitm\n"                                                                 \
  ".endif\n"                                                                 \
  ".ifndef vm\n"                                                             \
  ".error \"Bad operand \\opm\"\n"                                           \
  ".exitm\n"                                                                 \
  ".endif\n"                                                                 \
  ".int  0x4e809400 | vd | (vn << 5) | (vm << 16)\n"                         \
  ".endm\n"

#define _DECLARE_SDOT_VECTOR_2s                                              \
  ".altmacro\n"                                                              \
  ".macro sdot opd:req, opn:req, opm:req\n"                                  \
  "local vd, vn, vm\n"                                                       \
  ".irp "                                                                    \
  "reg,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25," \
  "26,27,28,29,30,31\n"                                                      \
  ".ifeqs \"\\opd\",\"v\\reg\\.2s\"\n"                                       \
  ".set vd,\\reg\n"                                                          \
  ".endif\n"                                                                 \
  ".ifeqs \"\\opn\",\"v\\reg\\.8b\"\n"                                       \
  ".set vn,\\reg\n"                                                          \
  ".endif\n"                                                                 \
  ".ifeqs \"\\opm\",\"v\\reg\\.8b\"\n"                                       \
  ".set vm,\\reg\n"                                                          \
  ".endif\n"                                                                 \
  ".endr\n"                                                                  \
  ".endr\n"                                                                  \
  ".ifndef vd\n"                                                             \
  ".error \"Bad operand \\opd\"\n"                                           \
  ".exitm\n"                                                                 \
  ".endif\n"                                                                 \
  ".ifndef vn\n"                                                             \
  ".error \"Bad operand \\opn\"\n"                                           \
  ".exitm\n"                                                                 \
  ".endif\n"                                                                 \
  ".ifndef vm\n"                                                             \
  ".error \"Bad operand \\opm\"\n"                                           \
  ".exitm\n"                                                                 \
  ".endif\n"                                                                 \
  ".int  0x0e809400 | vd | (vn << 5) | (vm << 16)\n"                         \
  ".endm\n"

#define _DECLARE_SDOT_ELEMENT_2s                                               \
  ".altmacro\n"                                                                \
  ".macro sdot opd:req, opn:req, opm:req\n"                                    \
  "local vd, vn, vm, h, l\n"                                                   \
  ".irp "                                                                      \
  "reg,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,"   \
  "26,27,28,29,30,31\n"                                                        \
  ".ifeqs \"\\opd\",\"v\\reg\\.2s\"\n"                                         \
  ".set vd,\\reg\n"                                                            \
  ".endif\n"                                                                   \
  ".ifeqs \"\\opn\",\"v\\reg\\.8b\"\n"                                         \
  ".set vn,\\reg\n"                                                            \
  ".endif\n"                                                                   \
  ".irp idx,0,1,2,3\n"                                                         \
  ".ifeqs \"\\opm\",\"v\\reg\\.4b[\\idx\\]\"\n"                                \
  ".set vm,\\reg\n"                                                            \
  ".set h,\\idx / 2\n"                                                         \
  ".set l,\\idx %% 2\n"                                                        \
  ".endif\n"                                                                   \
  ".endr\n"                                                                    \
  ".endr\n"                                                                    \
  ".ifndef vd\n"                                                               \
  ".error \"Bad operand \\opd\"\n"                                             \
  ".exitm\n"                                                                   \
  ".endif\n"                                                                   \
  ".ifndef vn\n"                                                               \
  ".error \"Bad operand \\opn\"\n"                                             \
  ".exitm\n"                                                                   \
  ".endif\n"                                                                   \
  ".ifndef vm\n"                                                               \
  ".error \"Bad operand \\opm\"\n"                                             \
  ".exitm\n"                                                                   \
  ".endif\n"                                                                   \
  ".ifndef h\n"                                                                \
  ".error \"Bad operand \\opm\"\n"                                             \
  ".exitm\n"                                                                   \
  ".endif\n"                                                                   \
  ".ifndef l\n"                                                                \
  ".error \"Bad operand \\opm\"\n"                                             \
  ".exitm\n"                                                                   \
  ".endif\n"                                                                   \
  ".int    0x0f80e000 | vd | (vn << 5) | (vm << 16) | (l << 21) | (h << 11)\n" \
  ".endm\n"
