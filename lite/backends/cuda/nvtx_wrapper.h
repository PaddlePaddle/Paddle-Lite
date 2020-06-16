// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include "nvtx3/nvToolsExt.h"

namespace paddle {
namespace lite {

enum class Color : uint32_t {
  Engine = 0xFFD2691E,
  Runner = 0xFFFFD700,
};

// Generate an NVTX range that is started when `generate` is called
// and closed when the object is destroyed.
class NVTXRangeAnnotation {
 public:
  explicit NVTXRangeAnnotation(nvtxDomainHandle_t domain);
  NVTXRangeAnnotation(NVTXRangeAnnotation&& other);
  NVTXRangeAnnotation(const NVTXRangeAnnotation&) = delete;
  NVTXRangeAnnotation& operator=(const NVTXRangeAnnotation&) = delete;
  ~NVTXRangeAnnotation();
  void generate(nvtxStringHandle_t stringHandle, Color color);

 private:
  nvtxDomainHandle_t domain_;
  bool isGenerating_;
};

class NVTXAnnotator {
 public:
  static const NVTXAnnotator& Global();

 public:
  bool IsEnabled() const;
  NVTXRangeAnnotation AnnotateBlock() const;
  nvtxStringHandle_t RegisterString(const char*) const;

 private:
  // Only a global instance of that object is allowed.
  // It can be accessed by call `NVTXAnnotator::Global()` function.
  explicit NVTXAnnotator(const char* domainName);

 private:
  nvtxDomainHandle_t domain_;
};

}  // namespace lite
}  // namespace paddle
