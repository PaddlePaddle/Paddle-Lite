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

#ifdef LITE_WITH_NVTX
#include "lite/backends/cuda/nvtx_wrapper.h"

#include <cuda.h>
#include <cuda_runtime.h>

namespace paddle {
namespace lite {

NVTXRangeAnnotation::NVTXRangeAnnotation(nvtxDomainHandle_t domain)
    : domain_(domain), isGenerating_(false) {}

NVTXRangeAnnotation::NVTXRangeAnnotation(NVTXRangeAnnotation&& other)
    : domain_(other.domain_), isGenerating_(other.isGenerating_) {
  other.isGenerating_ = false;
}

NVTXRangeAnnotation::~NVTXRangeAnnotation() {
  if (isGenerating_) {
    nvtxDomainRangePop(domain_);
  }
}

void NVTXRangeAnnotation::generate(nvtxStringHandle_t stringHandle,
                                   Color color) {
  nvtxEventAttributes_t attributes = nvtxEventAttributes_t();
  attributes.version = NVTX_VERSION;
  attributes.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  attributes.colorType = NVTX_COLOR_ARGB;
  attributes.color = static_cast<decltype(attributes.color)>(color);
  attributes.messageType = NVTX_MESSAGE_TYPE_REGISTERED;
  attributes.message.registered = stringHandle;

  nvtxDomainRangePushEx(domain_, &attributes);
  isGenerating_ = true;
}

const NVTXAnnotator& NVTXAnnotator::Global() {
  static const NVTXAnnotator annotator("Paddle-Lite");
  return annotator;
}

bool NVTXAnnotator::IsEnabled() const { return domain_ != nullptr; }

NVTXRangeAnnotation NVTXAnnotator::AnnotateBlock() const {
  return NVTXRangeAnnotation(domain_);
}

nvtxStringHandle_t NVTXAnnotator::RegisterString(const char* str) const {
  return nvtxDomainRegisterStringA(domain_, str);
}

NVTXAnnotator::NVTXAnnotator(const char* domainName)
    : domain_(nvtxDomainCreateA(domainName)) {}

}  // namespace lite
}  // namespace paddle
#endif
