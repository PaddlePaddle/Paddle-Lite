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

#include "lite/api/paddle_place.h"
#include "lite/backends/fpga/KD/float16.hpp"
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
#include "lite/core/target_wrapper.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace fpga {

using float16 = zynqmp::float16;

template <typename T>
void convert_to_hwc(
    T* chw_data, T* hwc_data, int num, int channel, int height, int width) {
  int chw = channel * height * width;
  int wc = width * channel;
  int index = 0;
  for (int n = 0; n < num; n++) {
    for (int c = 0; c < channel; c++) {
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          hwc_data[n * chw + h * wc + w * channel + c] = chw_data[index];
          index++;
        }
      }
    }
  }
}

template <typename T>
void hwc_to_chw(
    T* chw_data, T* hwc_data, int num, int channel, int height, int width) {
  int chw = channel * height * width;
  int wc = width * channel;
  int wh = width * height;
  int index = 0;
  for (int n = 0; n < num; n++) {
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        for (int c = 0; c < channel; c++) {
          chw_data[n * chw + c * wh + h * width + w] = hwc_data[index];
          index++;
        }
      }
    }
  }
}

void TransHwcToChw(Tensor* dest, const Tensor* src) {
  if (src->ZynqTensor()->dataType() == zynqmp::FP32) {
    float* chw = dest->mutable_data<float>();
    float* hwc = const_cast<float*>(src->data<float>());
    int num = dest->dims()[0];
    int channel = dest->dims()[1];
    int height = 1;
    if (dest->dims().size() > 2) {
      height = dest->dims()[2];
    }
    int width = 1;
    if (dest->dims().size() > 3) {
      width = dest->dims()[3];
    }

    hwc_to_chw<float>(chw, hwc, num, channel, height, width);
  }

  if (src->ZynqTensor()->dataType() == zynqmp::FP16) {
    float16* chw = dest->mutable_data<float16>();
    float16* hwc = const_cast<float16*>(src->data<float16>());
    int num = dest->dims()[0];
    int channel = dest->dims()[1];
    int height = 1;
    if (dest->dims().size() > 2) {
      height = dest->dims()[2];
    }
    int width = 1;
    if (dest->dims().size() > 3) {
      width = dest->dims()[3];
    }

    hwc_to_chw<float16>(chw, hwc, num, channel, height, width);
  }
}
void TransChwToHwc(Tensor* dest, const Tensor* src) {
  std::cout << "chw to hwc \n";
  exit(-1);
}

class TransHwcToChwCompute
    : public KernelLite<TARGET(kFPGA), PRECISION(kAny), DATALAYOUT(kNHWC)> {
 public:
  void Run() override {
    auto& param = Param<operators::LayoutParam>();
    param.x->ZynqTensor()->syncToCPU();
    TransHwcToChw(param.y, param.x);
    param.y->ZynqTensor()->flush();
    param.y->ZynqTensor()->copyScaleFrom(param.x->ZynqTensor());

    auto out_lod = param.y->mutable_lod();
    *out_lod = param.x->lod();
  }

  std::unique_ptr<type_infer_handler_t> GetTypeInferHandler() override {
    std::unique_ptr<type_infer_handler_t> res(new type_infer_handler_t);
    *res = [](const std::map<std::string, const Type*>& inputs,
              const std::string& out) -> const Type* {
      CHECK(!inputs.empty());
      auto* type = inputs.at("Input");
      CHECK(type->layout() == DATALAYOUT(kNHWC));

      auto out_place = type->place();
      out_place.layout = DATALAYOUT(kNHWC);
      auto* out_type = Type::Get(type->id(),
                                 out_place.target,
                                 out_place.precision,
                                 out_place.layout,
                                 out_place.device);
      return out_type;
    };
    return res;
  }

  std::string doc() const override { return "Trans Layout from NHWC to NCHW"; }
};

/*
 * This kernel copies a tensor from FPGA to host space.
 */
class TransChwToHwcCompute
    : public KernelLite<TARGET(kFPGA), PRECISION(kAny), DATALAYOUT(kNHWC)> {
 public:
  void Run() override {
    auto& param = Param<operators::LayoutParam>();
    auto out_data = param.y->mutable_data<float16>(TARGET(kFPGA));
    TransChwToHwc(param.y, param.x);
  }

  std::string doc() const override { return "Trans Layout from NHWC to NCHW"; }
};

}  // namespace fpga
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(layout,
                     kFPGA,
                     kAny,
                     kNHWC,
                     paddle::lite::kernels::fpga::TransHwcToChwCompute,
                     hwc_to_chw_fpga_fp16)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kFPGA),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNHWC))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kFPGA),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kNCHW))})
    .Finalize();

REGISTER_LITE_KERNEL(layout,
                     kFPGA,
                     kAny,
                     kNHWC,
                     paddle::lite::kernels::fpga::TransHwcToChwCompute,
                     hwc_to_chw_arm_float)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kARM),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNHWC))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kARM),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNCHW))})
    .Finalize();

REGISTER_LITE_KERNEL(layout,
                     kFPGA,
                     kAny,
                     kNHWC,
                     paddle::lite::kernels::fpga::TransChwToHwcCompute,
                     chw_to_hwc_fpga_fp16)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kFPGA),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNCHW))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kFPGA),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kNHWC))})
    .Finalize();

REGISTER_LITE_KERNEL(layout_once,
                     kFPGA,
                     kAny,
                     kNHWC,
                     paddle::lite::kernels::fpga::TransHwcToChwCompute,
                     hwc_to_chw_fpga_fp16)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kFPGA),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNHWC))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kFPGA),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kNCHW))})
    .Finalize();

REGISTER_LITE_KERNEL(layout_once,
                     kFPGA,
                     kAny,
                     kNHWC,
                     paddle::lite::kernels::fpga::TransChwToHwcCompute,
                     chw_to_hwc_fpga_fp16)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kFPGA),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNCHW))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kFPGA),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kNHWC))})
    .Finalize();
