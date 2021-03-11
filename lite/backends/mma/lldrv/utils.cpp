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

#include <memory.h>
#include <algorithm>
#include <fstream>
#include <string>

#include "lite/backends/mma/lldrv/utils.h"

namespace paddle {
namespace lite {
namespace mma {

//---------------------------------------------------------------------------

float find_max(const float* data, int size) 
{
	float max = 0.0;
	
	for (size_t i=0; i<size; ++i) {
		float value = data[i];
		float abs = value>0.0 ? value : -value;
		
		max = std::max(max, abs);
	}
	
	return max;
}

void quantize_s8(const float* src, int8_t* dst, int size, float factor) 
{
	float fdata;
	
	for (size_t i=0; i<size; i++) {
		fdata = src[i] * factor;
		
		if (fdata<0.0) {
			fdata -= 0.5;
		} else {
			fdata += 0.5;
		}
		
		dst[i] = (int8_t)fdata;
	}
}

void quantize_s32(const float* src, int32_t* dst, int size, float factor) 
{
	float fdata;
	
	for (size_t i=0; i<size; i++) {
		fdata = src[i] * factor;
		
		if (fdata<0.0) {
			fdata -= 0.5;
		} else {
			fdata += 0.5;
		}
		
		dst[i] = (int32_t)fdata;
	}
}

//---------------------------------------------------------------------------

}  // namespace mma
}  // namespace lite
}  // namespace paddle
