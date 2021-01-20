// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/backends/metal/metal_debug.h"
#include "lite/utils/cp_logging.h"

namespace paddle {
namespace lite {

void metal_debug::dump_image(std::string name, metal_image* image, int length, dump_mode mode) {
  float* buf = (float*)malloc(sizeof(float) * length);
  std::string filename = name + ".txt";
  FILE* fp = fopen(filename.c_str(), "w");
  image->to_nchw<float>(buf);
  for (int i = 0; i < length; ++i) {
    if (mode == dump_mode::TO_FILE || mode == dump_mode::TO_BOTH) fprintf(fp, "%f\n", buf[i]);
    if (mode == dump_mode::TO_STDOUT || mode == dump_mode::TO_BOTH) VLOG(4) << buf[i];
  }
  free(buf);
  fclose(fp);
  fp = NULL;
}

void metal_debug::dump_image(std::string name,
                             const metal_image* image,
                             int length,
                             dump_mode mode) {
  dump_image(name, const_cast<metal_image*>(image), length, mode);
}

void metal_debug::dump_buffer(std::string name, metal_buffer* buffer, int length, dump_mode mode) {
  float* buf = (float*)malloc(sizeof(float) * length);
  std::string filename = name + ".txt";
  FILE* fp = fopen(filename.c_str(), "w");
  buffer->to_nchw<float>(buf);
  for (int i = 0; i < length; ++i) {
    if (mode == dump_mode::TO_FILE || mode == dump_mode::TO_BOTH) fprintf(fp, "%f\n", buf[i]);
    if (mode == dump_mode::TO_STDOUT || mode == dump_mode::TO_BOTH) VLOG(4) << buf[i];
  }
  free(buf);
  fclose(fp);
  fp = NULL;
}

void metal_debug::dump_buffer(std::string name,
                              const metal_buffer* buf,
                              int length,
                              dump_mode mode) {
  dump_buffer(name, const_cast<metal_buffer*>(buf), length, mode);
}

void metal_debug::dump_nchw_float(std::string name, float* data, int length, dump_mode mode) {
  std::string filename = name + ".txt";
  FILE* fp = fopen(filename.c_str(), "w");
  for (int i = 0; i < length; ++i) {
    if (mode == dump_mode::TO_FILE || mode == dump_mode::TO_BOTH) fprintf(fp, "%f\n", data[i]);
    if (mode == dump_mode::TO_STDOUT || mode == dump_mode::TO_BOTH) VLOG(4) << data[i];
  }
  free(data);
  fclose(fp);
  fp = NULL;
}

}
}