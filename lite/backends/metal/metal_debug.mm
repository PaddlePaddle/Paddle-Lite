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

void MetalDebug::DumpImage(std::string name, MetalImage* image, int length, DumpMode mode) {
  float* buf = (float*)malloc(sizeof(float) * length);
  std::string filename = name + ".txt";
  FILE* fp = fopen(filename.c_str(), "w");
  image->CopyToNCHW<float>(buf);
  for (int i = 0; i < length; ++i) {
    if (mode == DumpMode::kFile || mode == DumpMode::kBoth) fprintf(fp, "%f\n", buf[i]);
    if (mode == DumpMode::kStd || mode == DumpMode::kBoth) VLOG(4) << buf[i];
  }
  free(buf);
  fclose(fp);
  fp = NULL;
}

void MetalDebug::DumpImage(std::string name,
                             const MetalImage* image,
                             int length,
                             DumpMode mode) {
  DumpImage(name, const_cast<MetalImage*>(image), length, mode);
}

void MetalDebug::DumpBuffer(std::string name, MetalBuffer* buffer, int length, DumpMode mode) {
  float* buf = (float*)malloc(sizeof(float) * length);
  std::string filename = name + ".txt";
  FILE* fp = fopen(filename.c_str(), "w");
  buffer->CopyToNCHW<float>(buf);
  for (int i = 0; i < length; ++i) {
    if (mode == DumpMode::kFile || mode == DumpMode::kBoth) fprintf(fp, "%f\n", buf[i]);
    if (mode == DumpMode::kStd || mode == DumpMode::kBoth) VLOG(4) << buf[i];
  }
  free(buf);
  fclose(fp);
  fp = NULL;
}

void MetalDebug::DumpBuffer(std::string name,
                              const MetalBuffer* buf,
                              int length,
                              DumpMode mode) {
  DumpBuffer(name, const_cast<MetalBuffer*>(buf), length, mode);
}

void MetalDebug::DumpNCHWFloat(std::string name, float* data, int length, DumpMode mode) {
  std::string filename = name + ".txt";
  FILE* fp = fopen(filename.c_str(), "w");
  for (int i = 0; i < length; ++i) {
    if (mode == DumpMode::kFile || mode == DumpMode::kBoth) fprintf(fp, "%f\n", data[i]);
    if (mode == DumpMode::kStd || mode == DumpMode::kBoth) VLOG(4) << data[i];
  }
  free(data);
  fclose(fp);
  fp = NULL;
}

}
}