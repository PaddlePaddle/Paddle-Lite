/* Copyright (c) 2020 AWCloud. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <cstdint>
#include <cstdlib>
#include <cwchar>
#include <vector>
#include <sys/time.h>

namespace paddle {
namespace lite {
namespace mma {

inline int64_t get_tminus( void ) 
{
	struct timeval time;
	
	gettimeofday(&time, NULL);
	
	return 1000000LL * (int64_t)time.tv_sec + (int64_t)time.tv_usec;
}

float find_max(const float* data, int size);

void quantize_s8 (const float* src, int8_t* dst, int size, float factor);
void quantize_s32(const float* src, int32_t* dst, int size, float factor);

}  // namespace mma
}  // namespace lite
}  // namespace paddle
