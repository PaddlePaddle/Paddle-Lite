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

#include <cstdio>
#include <cstdlib>

#include "lite/backends/metal/metal_debug.h"
#include "lite/utils/cp_logging.h"

namespace paddle {
namespace lite {

void MetalDebug::DumpImage(const std::string& name, MetalImage* image, DumpMode mode) {
  int length = image->tensor_dim_.production();
  auto buf = (float*)malloc(sizeof(float) * length);
  image->CopyToNCHW<float>(buf);
  std::string OUTPUT_BASE_PATH = "";
#ifdef TARGET_IOS
  OUTPUT_BASE_PATH = std::string([[NSSearchPathForDirectoriesInDomains(
                         NSDocumentDirectory, NSUserDomainMask, YES) objectAtIndex:0] UTF8String]) +
                     "/";
#endif
  std::string filename = OUTPUT_BASE_PATH + name + ".txt";
  FILE* fp = fopen(filename.c_str(), "w");
  for (int i = 0; i < length; ++i) {
    if (mode == DumpMode::kFile || mode == DumpMode::kBoth) fprintf(fp, "%f\n", buf[i]);
    if (mode == DumpMode::kStd || mode == DumpMode::kBoth) VLOG(4) << buf[i];
  }
  free(buf);
  fclose(fp);
}

void MetalDebug::DumpImage(const std::string& name, const MetalImage* image, DumpMode mode) {
  DumpImage(name, const_cast<MetalImage*>(image), mode);
}

void MetalDebug::DumpImage(const std::string& name,
                           std::shared_ptr<MetalImage> image,
                           DumpMode mode) {
  int length = image->tensor_dim_.production();
  auto buf = (float*)malloc(sizeof(float) * length);
  std::string OUTPUT_BASE_PATH = "";
#ifdef TARGET_IOS
  OUTPUT_BASE_PATH = std::string([[NSSearchPathForDirectoriesInDomains(
                         NSDocumentDirectory, NSUserDomainMask, YES) objectAtIndex:0] UTF8String]) +
                     "/";
#endif
  std::string filename = OUTPUT_BASE_PATH + name + ".txt";
  FILE* fp = fopen(filename.c_str(), "w");
  image->CopyToNCHW<float>(buf);
  for (int i = 0; i < length; ++i) {
    if (mode == DumpMode::kFile || mode == DumpMode::kBoth) fprintf(fp, "%f\n", buf[i]);
    if (mode == DumpMode::kStd || mode == DumpMode::kBoth) VLOG(4) << buf[i];
  }
  free(buf);
  fclose(fp);
}

void MetalDebug::DumpBuffer(const std::string& name, MetalBuffer* buffer, DumpMode mode) {
  std::string OUTPUT_BASE_PATH = "";
#ifdef TARGET_IOS
  OUTPUT_BASE_PATH = std::string([[NSSearchPathForDirectoriesInDomains(
                         NSDocumentDirectory, NSUserDomainMask, YES) objectAtIndex:0] UTF8String]) +
                     "/";
#endif
  if (buffer->type() == MetalBuffer::TYPE::kTensorBuffer) {
    int length = buffer->tensor_dim().production();
    auto buf = (float*)malloc(sizeof(float) * length);
    std::string filename = OUTPUT_BASE_PATH + name + ".txt";
    FILE* fp = fopen(filename.c_str(), "w");
    buffer->CopyToNCHW<float>(buf);
    for (int i = 0; i < length; ++i) {
      if (mode == DumpMode::kFile || mode == DumpMode::kBoth) fprintf(fp, "%f\n", buf[i]);
      if (mode == DumpMode::kStd || mode == DumpMode::kBoth) VLOG(4) << buf[i];
    }
    free(buf);
    fclose(fp);
  } else if (buffer->type() == MetalBuffer::TYPE::kCommonBuffer) {
    int length = buffer->data_length();
    auto buf = (float*)malloc(length);
    std::string filename = OUTPUT_BASE_PATH + name + ".txt";
    FILE* fp = fopen(filename.c_str(), "w");
    memcpy(buf, buffer->buffer().contents, length);
    for (int i = 0; i < length; ++i) {
      if (mode == DumpMode::kFile || mode == DumpMode::kBoth) fprintf(fp, "%f\n", buf[i]);
      if (mode == DumpMode::kStd || mode == DumpMode::kBoth) VLOG(4) << buf[i];
    }
    free(buf);
    fclose(fp);
  }
}

void MetalDebug::DumpBuffer(const std::string& name,
                            const MetalBuffer* buf,
                            int length,
                            DumpMode mode) {
  DumpBuffer(name, const_cast<MetalBuffer*>(buf), length, mode);
}

void MetalDebug::DumpNCHWFloat(const std::string& name, float* data, int length, DumpMode mode) {
  std::string OUTPUT_BASE_PATH = "";
#ifdef TARGET_IOS
  OUTPUT_BASE_PATH = std::string([[NSSearchPathForDirectoriesInDomains(
                         NSDocumentDirectory, NSUserDomainMask, YES) objectAtIndex:0] UTF8String]) +
                     "/";
#endif
  std::string filename = OUTPUT_BASE_PATH + name + ".txt";
  FILE* fp = fopen(filename.c_str(), "w");
  for (int i = 0; i < length; ++i) {
    if (mode == DumpMode::kFile || mode == DumpMode::kBoth) fprintf(fp, "%f\n", data[i]);
    if (mode == DumpMode::kStd || mode == DumpMode::kBoth) VLOG(4) << data[i];
  }
  free(data);
  fclose(fp);
}

void MetalDebug::DumpBuffer(const std::string& name,
                            std::shared_ptr<MetalBuffer> buffer,
                            int length,
                            DumpMode mode) {
  std::string OUTPUT_BASE_PATH = "";
#ifdef TARGET_IOS
  OUTPUT_BASE_PATH = std::string([[NSSearchPathForDirectoriesInDomains(
                         NSDocumentDirectory, NSUserDomainMask, YES) objectAtIndex:0] UTF8String]) +
                     "/";
#endif
  auto buf = (float*)malloc(sizeof(float) * length);
  std::string filename = OUTPUT_BASE_PATH + name + ".txt";
  FILE* fp = fopen(filename.c_str(), "w");
  buffer->CopyToNCHW<float>(buf);
  for (int i = 0; i < length; ++i) {
    if (mode == DumpMode::kFile || mode == DumpMode::kBoth) fprintf(fp, "%f\n", buf[i]);
    if (mode == DumpMode::kStd || mode == DumpMode::kBoth) VLOG(4) << buf[i];
  }
  free(buf);
  fclose(fp);
}

LITE_THREAD_LOCAL std::map<std::string, int> MetalDebug::op_stats_ = {};
LITE_THREAD_LOCAL bool MetalDebug::enable_ = false;
LITE_THREAD_LOCAL int MetalDebug::layer_count_ = 0;

}  // namespace lite
}  // namespace paddle