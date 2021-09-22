/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License. */

#include <metal_stdlib>
using namespace metal;

#define CONCAT2(a, b) a##b
#define CONCAT2_(a, b) a##_##b
#define CONCAT3_(a, b, c) a##_##b##_##c
#define CONCAT4_(a, b, c, d) a##_##b##_##c##_##d
#define CONCAT5_(a, b, c, d, e) a##_##b##_##c##_##d##_##e

#define FUNC(f, r, n, v, p) CONCAT5_(f, r, n, v, p)
#define VECTOR(p, n) CONCAT2(p, n)
#define FUNC2_(a, b) CONCAT2_(a, b)
#define FUNC3_(a, b, c) CONCAT3_(a, b, c)
